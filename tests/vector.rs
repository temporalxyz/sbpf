#![allow(clippy::arithmetic_side_effects)]
//! Tests of the 512-bit vector ops: verifier, interpreter vs JIT, faults, host calls.
//!
//! The comparison of the reference semantics against native AVX-512 is in `vector_native.rs`.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use solana_sbpf::{
    declare_builtin_function, ebpf,
    elf::Executable,
    error::EbpfError,
    memory_region::{AccessType, AccessViolationHandler, MemoryRegion},
    program::{BuiltinFunctionDefinition, BuiltinProgram, FunctionRegistry, SBPFVersion},
    vector::{encode, encode_with_offset, op, op_info, VectorRegister, OPS, VECTOR_REGISTERS},
    verifier::{RequisiteVerifier, VerifierError},
    vm::{CallFrame, Config, ExecutionMode},
};
use std::{convert::TryInto, sync::Arc};
use test_utils::{create_vm, TestContextObject};

// Program construction

fn insn(opc: u8, dst: u8, src: u8, off: i16, imm: i32) -> u64 {
    opc as u64
        | ((dst | src << 4) as u64) << 8
        | (off as u16 as u64) << 16
        | (imm as u32 as u64) << 32
}
fn mov64_imm(dst: u8, imm: i32) -> u64 {
    insn(ebpf::MOV64_IMM, dst, 0, 0, imm)
}
fn mov64_reg(dst: u8, src: u8) -> u64 {
    insn(ebpf::MOV64_REG, dst, src, 0, 0)
}
fn add64_imm(dst: u8, imm: i32) -> u64 {
    insn(ebpf::ADD64_IMM, dst, 0, 0, imm)
}
fn lddw(dst: u8, imm: u64) -> [u64; 2] {
    [
        insn(ebpf::LD_DW_IMM, dst, 0, 0, imm as u32 as i32),
        (imm >> 32) << 32,
    ]
}
fn exit() -> u64 {
    insn(ebpf::EXIT, 0, 0, 0, 0)
}
fn vop(op: u8, imm8: u8, out: u8, in0: u8, in1: u8) -> u64 {
    encode(op, imm8, out, in0, in1)
}

const SYSCALL_CLOBBER: &str = "clobber_vector_registers";

/// Overwrites every vector and mask register of the host (if it has AVX-512)
fn clobber_host_vector_registers() {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        #[target_feature(enable = "avx512f")]
        unsafe fn clobber() {
            macro_rules! clobber {
                ($($zmm:literal)*) => {
                    std::arch::asm!(
                        $(concat!("vpternlogd zmm", $zmm, ", zmm", $zmm, ", zmm", $zmm, ", 0xff"),)*
                        "kxnorw k1, k1, k1",
                        "kxnorw k7, k7, k7",
                        clobber_abi("C"),
                    )
                };
            }
            clobber!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31);
        }
        unsafe { clobber() };
    }
}

declare_builtin_function!(
    /// Clobbers all host vector registers and returns `a ^ 0x5a`
    SyscallClobber,
    fn rust(
        _context_object: &mut TestContextObject,
        a: u64,
        _b: u64,
        _c: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        clobber_host_vector_registers();
        Ok(a ^ 0x5a)
    }
);

fn syscall(name: &str) -> u64 {
    insn(
        ebpf::CALL_IMM,
        0,
        0,
        0,
        ebpf::hash_symbol_name(name.as_bytes()) as i32,
    )
}

/// Internal call (static syscalls): `offset` is relative to the next instruction
fn call_internal(offset: i32) -> u64 {
    insn(ebpf::CALL_IMM, 0, 1, 0, offset)
}

fn config(enable_vector_ops: bool, aligned_memory_mapping: bool) -> Config {
    Config {
        enable_vector_ops,
        aligned_memory_mapping,
        enable_instruction_meter: true,
        ..Config::default()
    }
}

fn executable(
    words: &[u64],
    config: Config,
    version: SBPFVersion,
) -> Executable<TestContextObject> {
    let mut loader = BuiltinProgram::new_loader(config);
    loader
        .register_function(
            SYSCALL_CLOBBER,
            (SyscallClobber::vm, SyscallClobber::codegen),
        )
        .unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Executable::from_text_bytes(
        &bytes,
        Arc::new(loader),
        version,
        FunctionRegistry::default(),
    )
    .unwrap()
}

fn verified(words: &[u64], config: Config) -> Executable<TestContextObject> {
    let executable = executable(words, config, SBPFVersion::V3);
    executable.verify::<RequisiteVerifier>().unwrap();
    executable
}

// Execution

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    result: String,
    instruction_count: u64,
    pc: u64,
    vector_registers: Vec<VectorRegister>,
    memory: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Writable,
    ReadOnly,
    /// Read-only, the access violation handler makes it writable on the first store
    CopyOnWrite,
}

fn run(
    executable: &Executable<TestContextObject>,
    input: &[u8],
    kind: Input,
    budget: u64,
    jit: bool,
) -> Outcome {
    let mut memory = input.to_vec();
    let mut copy = input.to_vec();
    let mut context_object = TestContextObject::new(budget);
    let (region, handler): (MemoryRegion, Option<AccessViolationHandler>) = match kind {
        Input::Writable => (
            MemoryRegion::new(&raw mut memory[..], ebpf::MM_INPUT_START),
            None,
        ),
        Input::ReadOnly => (
            MemoryRegion::new(&raw const memory[..], ebpf::MM_INPUT_START),
            None,
        ),
        Input::CopyOnWrite => {
            let mut region = MemoryRegion::new(&raw const memory[..], ebpf::MM_INPUT_START);
            region.access_violation_handler_payload = Some(1);
            let writable = &raw mut copy[..];
            let handler: AccessViolationHandler =
                Box::new(move |region, _max_len, access_type, _vm_addr, _len| {
                    if access_type == AccessType::Store
                        && region.access_violation_handler_payload == Some(1)
                    {
                        unsafe { region.redirect(writable) };
                    }
                });
            (region, Some(handler))
        }
    };
    let mut call_frames = vec![CallFrame::default(); executable.get_config().max_call_depth];
    let (instruction_count, result, pc, vector_registers) = {
        create_vm!(
            vm,
            executable,
            &mut context_object,
            stack,
            heap,
            vec![region],
            handler
        );
        let (instruction_count, result) = vm.execute_program(
            executable,
            &mut if jit {
                ExecutionMode::Jit
            } else {
                ExecutionMode::Interpreted
            },
            &mut call_frames,
        );
        (
            instruction_count,
            result,
            vm.registers[11],
            vm.vector_registers.to_vec(),
        )
    };
    let written = if kind == Input::CopyOnWrite && copy != input {
        copy
    } else {
        memory
    };
    Outcome {
        result: format!("{result:?}"),
        instruction_count,
        pc,
        vector_registers,
        memory: written,
    }
}

/// Whether this host can run the vector op JIT
fn jit_available() -> bool {
    #[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
    {
        let executable = executable(
            &[vop(op::ZERO, 0, 0, 0, 0), exit()],
            config(true, true),
            SBPFVersion::V3,
        );
        match executable.jit_compile() {
            Ok(()) => return true,
            Err(EbpfError::UnsupportedHostFeatures(feature)) => {
                eprintln!("skipping JIT checks, the host lacks {feature}");
            }
            Err(err) => panic!("{:?}", err),
        }
    }
    false
}

/// Runs the program in the interpreter and (if possible) the JIT and checks they agree
fn run_both(
    executable: &Executable<TestContextObject>,
    input: &[u8],
    kind: Input,
    budget: u64,
) -> Outcome {
    let interpreted = run(executable, input, kind, budget, false);
    if jit_available() {
        executable.jit_compile().unwrap();
        let mut jitted = run(executable, input, kind, budget, true);
        if interpreted.result.contains("ExceededMaxInstructions") {
            // Like the scalar registers, the vector registers can be ahead in the JIT, because it
            // only checks the instruction meter at branches and memory accesses
            jitted.vector_registers = interpreted.vector_registers.clone();
        }
        assert_eq!(interpreted, jitted, "interpreter and JIT diverged");
    }
    interpreted
}

fn random_vector(rng: &mut SmallRng) -> [u8; 64] {
    let mut v = [0u8; 64];
    match rng.gen_range(0..4) {
        // Lanes of edge values
        0 => {
            let width = [1, 2, 4, 8][rng.gen_range(0..4)];
            for lane in v.chunks_mut(width) {
                let bits = width * 8;
                let value: u64 = match rng.gen_range(0..7) {
                    0 => 0,
                    1 => u64::MAX,
                    2 => 1,
                    3 => 1 << (bits - 1),
                    4 => (1 << (bits - 1)) - 1,
                    5 => rng.gen_range(0..64),
                    _ => rng.gen(),
                };
                lane.copy_from_slice(&value.to_le_bytes()[..width]);
            }
        }
        1 => v.fill([0x00, 0xff, 0x80, 0x7f][rng.gen_range(0..4)]),
        _ => rng.fill(&mut v[..]),
    }
    v
}

fn random_imm8(rng: &mut SmallRng, op: u8) -> u8 {
    let info = op_info(op).unwrap();
    loop {
        let imm8 = if rng.gen_bool(0.5) {
            rng.gen_range(0..64)
        } else {
            rng.gen()
        };
        if info.imm8_is_valid(imm8) {
            return imm8;
        }
    }
}

/// Every op, interpreter vs JIT, with random inputs and register assignments
#[test]
fn test_vector_ops_interpreter_vs_jit() {
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    let register_sets = [
        (3, 4, 5),
        (29, 16, 8),
        (0, 0, 1),
        (17, 17, 17),
        (15, 31 % 30, 24),
    ];
    for info in OPS {
        for iteration in 0..24 {
            let (out, in0, in1) = register_sets[iteration % register_sets.len()];
            let kind = info.kind;
            let (out, in0, in1) = (
                if kind.uses_out() { out } else { 0 },
                if kind.uses_in0() { in0 } else { 0 },
                if kind.uses_in1() { in1 } else { 0 },
            );
            let imm8 = random_imm8(&mut rng, info.op);
            // Input: three vectors, then 128 bytes for memory ops / output
            let mut input = vec![0u8; 3 * 64 + 128];
            for k in 0..3 {
                input[64 * k..64 * (k + 1)].copy_from_slice(&random_vector(&mut rng));
            }
            rng.fill(&mut input[192..]);
            let mut words = vec![mov64_reg(6, 1)];
            // Load the operands, in1 last so that aliased registers end up with distinct values
            for (k, reg) in [(0, out), (1, in0), (2, in1)] {
                words.push(mov64_reg(1, 6));
                words.push(add64_imm(1, 64 * k));
                words.push(vop(op::LD, 0, reg, 0, 0));
            }
            // r1 + off: an address inside the second half of the input (for memory ops),
            // otherwise r1 is a random value
            let (r1, off) = if info.accesses_memory() {
                let off: i16 = match iteration % 4 {
                    0 => 0,
                    1 => rng.gen_range(-256..256),
                    _ => rng.gen(),
                };
                let address = ebpf::MM_INPUT_START + 192 + rng.gen_range(0..64);
                (address.wrapping_sub(off as i64 as u64), off)
            } else {
                let r1 = random_vector(&mut rng)[..8]
                    .iter()
                    .fold(0, |acc, b| acc << 8 | *b as u64);
                (r1, 0)
            };
            words.extend(lddw(1, r1));
            words.push(mov64_imm(0, 0x1234));
            words.push(encode_with_offset(info.op, imm8, out, in0, in1, off));
            // Store all registers this op could have written
            words.push(mov64_reg(1, 6));
            words.push(vop(op::ST, 0, 0, out, 0));
            words.push(exit());
            let executable = verified(&words, config(true, iteration % 2 == 0));
            let outcome = run_both(&executable, &input, Input::Writable, 1000);
            assert!(
                outcome.result.starts_with("Ok("),
                "{} {:?}",
                info.name,
                outcome.result
            );
        }
    }
}

/// The vector registers are zeroed at the start of every execution
#[test]
fn test_vector_registers_zeroed_per_execution() {
    let words = [
        vop(op::HSUM_Q, 0, 0, 7, 0),
        vop(op::SPLAT_Q, 0, 7, 0, 0),
        exit(),
    ];
    let executable = verified(&words, config(true, true));
    let splat = VectorRegister(
        ebpf::MM_INPUT_START
            .to_le_bytes()
            .repeat(8)
            .try_into()
            .unwrap(),
    );
    let jit = jit_available();
    if jit {
        executable.jit_compile().unwrap();
    }
    for jit in [false, jit] {
        let mut context_object = TestContextObject::new(100);
        let mut call_frames = vec![CallFrame::default(); executable.get_config().max_call_depth];
        create_vm!(
            vm,
            &executable,
            &mut context_object,
            stack,
            heap,
            vec![],
            None
        );
        for _ in 0..2 {
            vm.context().remaining = 100;
            let (instruction_count, result) = vm.execute_program(
                &executable,
                &mut if jit {
                    ExecutionMode::Jit
                } else {
                    ExecutionMode::Interpreted
                },
                &mut call_frames,
            );
            assert_eq!(format!("{result:?}"), "Ok(0)");
            assert_eq!(instruction_count, 3);
            assert_eq!(vm.vector_registers[7], splat);
        }
    }
}

/// Faulting memory ops raise the same error as a scalar access and leave the registers unchanged
#[test]
fn test_vector_memory_faults() {
    let preload = |reg: u8| {
        let mut words = vec![mov64_reg(6, 1)];
        words.extend(lddw(1, 0x0123_4567_89ab_cdef));
        words.push(vop(op::SPLAT_Q, 0, reg, 0, 0));
        words
    };
    let input = vec![0x11u8; 100];
    let cases: Vec<(u8, u64, Input, &str)> = vec![
        // ld beyond the end of the input
        (
            op::LD,
            ebpf::MM_INPUT_START + 40,
            Input::Writable,
            "AccessViolation(Load, 17179869224, 64, \"input\")",
        ),
        (op::LD, ebpf::MM_INPUT_START + 36, Input::Writable, "Ok("),
        // ldbc crossing the end
        (
            op::LDBC_D,
            ebpf::MM_INPUT_START + 97,
            Input::Writable,
            "AccessViolation(Load, 17179869281, 4, \"input\")",
        ),
        (
            op::LDBC_Q,
            ebpf::MM_INPUT_START + 93,
            Input::Writable,
            "AccessViolation(Load, 17179869277, 8, \"input\")",
        ),
        (
            op::LDBC_Q,
            ebpf::MM_INPUT_START + 92,
            Input::Writable,
            "Ok(",
        ),
        // Unmapped
        (
            op::LD,
            0x1234,
            Input::Writable,
            "AccessViolation(Load, 4660, 64, \"unallocated\")",
        ),
        (
            op::LD,
            ebpf::MM_INPUT_START - 1,
            Input::Writable,
            "AccessViolation(Load, 17179869183, 64, \"unallocated\")",
        ),
        (
            op::LD,
            u64::MAX - 10,
            Input::Writable,
            "AccessViolation(Load, 18446744073709551605, 64",
        ),
        // Stack (in bounds and beyond the end)
        (
            op::LD,
            ebpf::MM_STACK_START + 4096 - 64,
            Input::Writable,
            "Ok(",
        ),
        (
            op::LD,
            ebpf::MM_STACK_START + 64 * 4096 - 63,
            Input::Writable,
            "StackAccessViolation",
        ),
        // Stores need write permission
        (
            op::ST,
            ebpf::MM_INPUT_START,
            Input::ReadOnly,
            "AccessViolation(Store, 17179869184, 64, \"input\")",
        ),
        (op::ST, ebpf::MM_INPUT_START + 8, Input::Writable, "Ok("),
        (
            op::ST,
            ebpf::MM_INPUT_START + 37,
            Input::Writable,
            "AccessViolation(Store, 17179869221, 64, \"input\")",
        ),
        (
            op::ST,
            ebpf::MM_BYTECODE_START,
            Input::Writable,
            "AccessViolation(Store, 4294967296, 64",
        ),
        // Copy on write through the access violation handler (slow path)
        (op::ST, ebpf::MM_INPUT_START + 3, Input::CopyOnWrite, "Ok("),
        (op::LD, ebpf::MM_INPUT_START + 3, Input::CopyOnWrite, "Ok("),
        // Address 0 (read-only data, reached by r1 + off wrapping around for some offsets)
        (op::LDBC_D, 0, Input::Writable, "Ok("),
        (op::ST, 0, Input::Writable, "AccessViolation(Store, 0, 64"),
    ];
    // The same effective addresses as r1 + off with various displacements
    let offsets: [i16; 6] = [0, 1, -1, 0x1230, i16::MIN, i16::MAX];
    for (aligned, (op, address, input_kind, expected), off) in [false, true]
        .iter()
        .copied()
        .flat_map(|a| cases.iter().map(move |c| (a, c)))
        .flat_map(|(a, c)| offsets.iter().map(move |off| (a, c, *off)))
    {
        let mut words = preload(9);
        words.extend(lddw(1, address.wrapping_sub(off as i64 as u64)));
        words.push(encode_with_offset(
            *op,
            0,
            if *op == op::ST { 0 } else { 9 },
            if *op == op::ST { 9 } else { 0 },
            0,
            off,
        ));
        words.push(mov64_imm(0, 7));
        words.push(exit());
        let executable = verified(&words, config(true, aligned));
        let outcome = run_both(&executable, &input, *input_kind, 1000);
        assert!(
            outcome.result.contains(expected),
            "{:#x} {} {} {}",
            address,
            off,
            outcome.result,
            expected
        );
        if expected.starts_with("Ok(") {
            assert_eq!(outcome.result, "Ok(7)");
        } else {
            assert_eq!(
                outcome.vector_registers[9].0,
                0x0123_4567_89ab_cdef_u64.to_le_bytes().repeat(8)[..]
            );
            assert_eq!(outcome.pc, 6, "{}", outcome.result);
            assert_eq!(outcome.instruction_count, 5);
        }
        if *op == op::ST && expected.starts_with("Ok(") {
            let offset = (*address - ebpf::MM_INPUT_START) as usize;
            assert_eq!(
                &outcome.memory[offset..offset + 64],
                &0x0123_4567_89ab_cdef_u64.to_le_bytes().repeat(8)[..]
            );
        }
    }
}

/// Running out of CUs at a vector memory op, inside and outside of the fast path
#[test]
fn test_vector_ops_instruction_meter() {
    let mut words = vec![mov64_reg(6, 1)];
    for _ in 0..10 {
        words.push(vop(op::LD, 0, 1, 0, 0));
        words.push(vop(op::ADD_D, 0, 2, 2, 1));
        words.push(insn(ebpf::LD_DW_REG, 3, 6, 8, 0));
    }
    words.push(vop(op::HSUM_D, 0, 0, 2, 0));
    words.push(exit());
    let input: Vec<u8> = (0..128).collect();
    for aligned in [false, true] {
        let executable = verified(&words, config(true, aligned));
        for budget in 0..34 {
            let outcome = run_both(&executable, &input, Input::Writable, budget);
            if budget < 33 {
                assert!(
                    outcome.result.contains("ExceededMaxInstructions"),
                    "{budget} {}",
                    outcome.result
                );
            } else {
                assert_eq!(
                    outcome.result,
                    format!(
                        "Ok({})",
                        (0..16)
                            .map(|i| (4 * i..4 * i + 4)
                                .fold(0u32, |a, b| a | (b as u32) << (8 * (b % 4))))
                            .fold(0u32, |a, b| a.wrapping_add(b))
                            .wrapping_mul(10)
                    )
                );
            }
        }
    }
}

/// Scalar loads and stores of all sizes in a program with vector ops (inline translation)
#[test]
fn test_scalar_memory_access_in_vector_program() {
    let mut rng = SmallRng::seed_from_u64(7);
    for aligned in [false, true] {
        for _ in 0..20 {
            let input: Vec<u8> = (0..64).map(|_| rng.gen()).collect();
            let mut words = vec![vop(op::SPLAT_D, 0, 4, 0, 0), mov64_imm(0, 0)];
            for (load, store_imm, store_reg) in [
                (ebpf::LD_B_REG, ebpf::ST_B_IMM, ebpf::ST_B_REG),
                (ebpf::LD_H_REG, ebpf::ST_H_IMM, ebpf::ST_H_REG),
                (ebpf::LD_W_REG, ebpf::ST_W_IMM, ebpf::ST_W_REG),
                (ebpf::LD_DW_REG, ebpf::ST_DW_IMM, ebpf::ST_DW_REG),
            ] {
                let offset = rng.gen_range(-8..72);
                words.push(insn(load, 2, 1, offset, 0));
                words.push(insn(ebpf::ADD64_REG, 0, 2, 0, 0));
                words.push(insn(
                    store_imm,
                    1,
                    0,
                    rng.gen_range(0..56),
                    rng.gen::<i32>(),
                ));
                words.push(insn(
                    store_imm,
                    1,
                    0,
                    rng.gen_range(0..56),
                    rng.gen_range(-2..2),
                ));
                words.push(insn(store_reg, 1, 2, rng.gen_range(-2..60), 0));
            }
            words.push(vop(op::ST, 0, 0, 4, 0));
            words.push(exit());
            let executable = verified(&words, config(true, aligned));
            for input_kind in [Input::Writable, Input::ReadOnly, Input::CopyOnWrite] {
                run_both(&executable, &input, input_kind, 1000);
            }
        }
    }
}

/// Vector registers survive scalar memory accesses, internal calls and syscalls in both engines
#[test]
fn test_vector_registers_survive_calls() {
    let mut words = vec![
        // v1 = splat.d(7), v2 = splat.d(input[0..4]), v29 = splat.q(r1)
        mov64_reg(6, 1),
        vop(op::SPLAT_Q, 0, 29, 0, 0),
        vop(op::LDBC_D, 0, 2, 0, 0),
        mov64_imm(1, 7),
        vop(op::SPLAT_D, 0, 1, 0, 0),
        // Internal call to `function` below
        call_internal(0),
    ];
    let call_index = words.len() - 1;
    words.extend([
        // v3 = v1 + v2 (v2 was modified by the callee)
        vop(op::ADD_D, 0, 3, 1, 2),
        // Syscall which clobbers all host vector registers, result in r0
        mov64_imm(1, 0x100),
        syscall(SYSCALL_CLOBBER),
        mov64_reg(7, 0),
        // Scalar store and load of the input
        insn(ebpf::ST_W_IMM, 6, 0, 64, 5),
        insn(ebpf::LD_W_REG, 8, 6, 64, 0),
        mov64_reg(1, 8),
        vop(op::SPLAT_D, 0, 4, 0, 0),
        vop(op::ADD_D, 0, 3, 3, 4),
        // Store v3 and v29, return hsum.d(v3) + r7
        mov64_reg(1, 6),
        vop(op::ST, 0, 0, 3, 0),
        add64_imm(1, 64),
        vop(op::ST, 0, 0, 29, 0),
        vop(op::HSUM_D, 0, 0, 3, 0),
        insn(ebpf::ADD64_REG, 0, 7, 0, 0),
        exit(),
    ]);
    let function = words.len();
    words[call_index] = call_internal((function - call_index - 1) as i32);
    words.extend([
        // function: v2 += splat.d(ldxw [r6 + 4]), with a syscall and a nested call in between
        insn(ebpf::LD_W_REG, 1, 6, 4, 0),
        vop(op::SPLAT_D, 0, 5, 0, 0),
        mov64_imm(1, 1),
        syscall(SYSCALL_CLOBBER),
        call_internal(2),
        vop(op::ADD_D, 0, 2, 2, 5),
        exit(),
        // nested: modifies v5 too
        vop(op::ADD_D, 0, 5, 5, 5),
        exit(),
    ]);
    let input: Vec<u8> = [3u32, 10]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .chain([0u8; 128])
        .collect();
    for aligned in [false, true] {
        let executable = verified(&words, config(true, aligned));
        let outcome = run_both(&executable, &input, Input::Writable, 1000);
        // v3.d[i] = 7 + (3 + 2 * 10) + 5 = 35
        assert_eq!(outcome.result, format!("Ok({})", 16 * 35 + (0x100 ^ 0x5a)));
        assert_eq!(&outcome.memory[..64], &35u32.to_le_bytes().repeat(16)[..]);
        assert_eq!(
            &outcome.memory[64..128],
            &ebpf::MM_INPUT_START.to_le_bytes().repeat(8)[..]
        );
        assert_eq!(
            outcome.vector_registers[5].0,
            20u32.to_le_bytes().repeat(16)[..]
        );
    }
}

/// Programs without vector ops (or with the flag off) are unaffected
#[test]
fn test_vector_ops_disabled() {
    let words = [vop(op::ZERO, 0, 1, 0, 0), mov64_imm(0, 1), exit()];
    // Flag off: src = 2 calls pass the verifier and fault at runtime as before
    let disabled = verified(&words, config(false, true));
    let outcome = run_both(&disabled, &[], Input::Writable, 100);
    assert_eq!(outcome.result, "Err(UnsupportedInstruction)");
    // Without static syscalls there are no vector ops
    executable(&words, config(true, true), SBPFVersion::V2)
        .verify::<RequisiteVerifier>()
        .unwrap();
    // Invalid encodings are only rejected if enabled
    let invalid = [vop(0x0a, 0, 1, 0, 0), exit()];
    verified(&invalid, config(false, true));
    #[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
    {
        // A program without vector ops compiles to the same machine code size as before
        let scalar = [mov64_imm(0, 1), insn(ebpf::LD_DW_REG, 2, 1, 0, 0), exit()];
        let lengths: Vec<usize> = [false, true]
            .iter()
            .map(|enable| {
                // Without diversification and (randomly keyed) sanitization the length is deterministic
                let config = Config {
                    noop_instruction_rate: 0,
                    sanitize_user_provided_values: false,
                    enable_instruction_meter: false,
                    ..config(*enable, true)
                };
                let executable = verified(&scalar, config);
                executable.jit_compile().unwrap();
                let program = executable.get_compiled_program().unwrap();
                assert!(program.entry_prologue().is_none());
                program.machine_code_length()
            })
            .collect();
        assert_eq!(lengths[0], lengths[1]);
    }
}

fn verify_vector_op(word: u64) -> Result<(), EbpfError> {
    executable(&[word, exit()], config(true, true), SBPFVersion::V3).verify::<RequisiteVerifier>()
}

#[test]
fn test_vector_verifier() {
    let rejected = |word: u64, reason: &str| {
        let result = verify_vector_op(word);
        assert!(
            matches!(&result, Err(EbpfError::VerifierError(VerifierError::InvalidVectorOp(0, r))) if *r == reason),
            "{:#x} {:?}",
            word,
            result
        );
    };
    // All defined ops with valid fields pass
    for info in OPS {
        let kind = info.kind;
        let imm8 = (0..=255).find(|i| info.imm8_is_valid(*i)).unwrap();
        let word = vop(
            info.op,
            imm8,
            kind.uses_out() as u8 * 29,
            kind.uses_in0() as u8 * 28,
            kind.uses_in1() as u8 * 27,
        );
        verify_vector_op(word).unwrap();
        // off is the displacement of memory accesses and must be 0 otherwise
        for off in [1i16, -1, 64, i16::MIN, i16::MAX] {
            let word = word | (off as u16 as u64) << 16;
            if info.accesses_memory() {
                verify_vector_op(word).unwrap();
            } else {
                rejected(word, "off must be 0");
            }
        }
        // Unused register fields must be zero
        if !kind.uses_out() {
            rejected(word | 1 << (32 + 10), "unused register field must be 0");
        }
        if !kind.uses_in0() {
            rejected(word | 1 << (32 + 5), "unused register field must be 0");
        }
        if !kind.uses_in1() {
            rejected(word | 1 << 32, "unused register field must be 0");
        }
        // Register fields >= 30
        if kind.uses_out() {
            rejected(
                vop(info.op, imm8, 30, 0, 0) | word & !(0x1f << 42),
                "register field out of range",
            );
        }
        if kind.uses_in0() {
            rejected(word | 31 << (32 + 5), "register field out of range");
        }
        if kind.uses_in1() {
            rejected(word | 30 << 32, "register field out of range");
        }
        // imm8 range
        for imm8 in 0..=255u8 {
            let word = word & !(0xff << 48) | (imm8 as u64) << 48;
            if info.imm8_is_valid(imm8) {
                verify_vector_op(word).unwrap();
            } else {
                rejected(word, "imm8 out of range");
            }
        }
    }
    // Spot checks of the imm8 ranges
    assert!(
        op_info(op::SHLI_W).unwrap().imm8_is_valid(15)
            && !op_info(op::SHLI_W).unwrap().imm8_is_valid(16)
    );
    assert!(
        op_info(op::SARI_Q).unwrap().imm8_is_valid(63)
            && !op_info(op::SARI_Q).unwrap().imm8_is_valid(64)
    );
    assert!(
        op_info(op::EXTRACT_D).unwrap().imm8_is_valid(15)
            && !op_info(op::EXTRACT_D).unwrap().imm8_is_valid(16)
    );
    assert!(
        op_info(op::CLMUL).unwrap().imm8_is_valid(0x11)
            && !op_info(op::CLMUL).unwrap().imm8_is_valid(0x02)
    );
    assert!(!op_info(op::ADD_D).unwrap().imm8_is_valid(1));
    // Unknown ops
    for op in 0..=255u8 {
        if op_info(op).is_none() {
            rejected(vop(op, 0, 0, 0, 0), "unknown op");
        }
    }
    // dst, off and imm bit 15
    rejected(vop(op::ADD_D, 0, 1, 2, 3) | 1 << 8, "dst must be 0");
    rejected(vop(op::ADD_D, 0, 1, 2, 3) | 1 << 16, "off must be 0");
    rejected(vop(op::ADD_D, 0, 1, 2, 3) | 1 << 47, "imm bit 15 must be 0");
    // The error reports the instruction index and skips the second half of lddw
    let words = lddw(1, 5);
    let executable = executable(
        &[words[0], words[1], vop(op::ADD_D, 0, 1, 2, 30), exit()],
        config(true, true),
        SBPFVersion::V3,
    );
    assert!(matches!(
        executable.verify::<RequisiteVerifier>(),
        Err(EbpfError::VerifierError(VerifierError::InvalidVectorOp(
            2,
            "register field out of range"
        )))
    ));
}

/// The used register set covers every register field which is in use
#[test]
fn test_vector_registers_across_whole_file() {
    // Write every vector register, then sum them all
    let mut words = vec![];
    for reg in 0..VECTOR_REGISTERS as u8 {
        words.push(mov64_imm(1, reg as i32 + 1));
        words.push(vop(op::SPLAT_Q, 0, reg, 0, 0));
    }
    words.push(syscall(SYSCALL_CLOBBER));
    for reg in 1..VECTOR_REGISTERS as u8 {
        words.push(vop(op::ADD_Q, 0, 0, 0, reg));
    }
    words.push(vop(op::HSUM_Q, 0, 0, 0, 0));
    words.push(exit());
    let executable = verified(&words, config(true, true));
    let outcome = run_both(&executable, &[], Input::Writable, 1000);
    assert_eq!(outcome.result, format!("Ok({})", 8 * (1..=30).sum::<u64>()));
}

/// Comparison of the reference semantics (interpreter) against native AVX-512 intrinsics
#[cfg(target_arch = "x86_64")]
mod native {
    use super::*;
    use solana_sbpf::vector::{
        execute_register_op, execute_scalar_input_op, execute_scalar_output_op,
    };
    use std::arch::x86_64::*;

    fn host_has_avx512() -> bool {
        let detected = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512ifma")
            && is_x86_feature_detected!("avx512bitalg")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("vpclmulqdq");
        if !detected {
            eprintln!("skipping native AVX-512 comparison, the host lacks features");
        }
        detected
    }

    fn to_native(v: &VectorRegister) -> __m512i {
        unsafe { std::mem::transmute(v.0) }
    }
    fn from_native(v: __m512i) -> VectorRegister {
        VectorRegister(unsafe { std::mem::transmute::<__m512i, [u8; 64]>(v) })
    }

    /// Instantiates `$body` for the runtime `$imm` from a list of literals. The body can use the
    /// constants `I` (i32), `U` (u32) and the 128-bit lane / element split for qwords (`QL`, `QE`)
    /// and dwords (`DL`, `DE`).
    macro_rules! with_imm {
        ($imm:expr, [$($n:literal)*], $body:expr) => {
            match $imm as u32 {
                $($n => {
                    #[allow(dead_code)] const I: i32 = $n;
                    #[allow(dead_code)] const U: u32 = $n;
                    #[allow(dead_code)] const QL: i32 = $n >> 1;
                    #[allow(dead_code)] const QE: i32 = $n & 1;
                    #[allow(dead_code)] const DL: i32 = $n >> 2;
                    #[allow(dead_code)] const DE: i32 = $n & 3;
                    $body
                })*
                _ => unreachable!(),
            }
        };
    }

    enum Native {
        Vector(__m512i),
        Scalar(u64),
    }

    /// The intrinsic column of the spec, for every op without a memory operand
    #[target_feature(
        enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512ifma,avx512bitalg,avx512vpopcntdq,vpclmulqdq"
    )]
    unsafe fn native(op: u8, imm8: u8, c: __m512i, a: __m512i, b: __m512i, r1: u64) -> Native {
        use Native::{Scalar, Vector};
        match op {
            op::SPLAT_B => Vector(_mm512_set1_epi8(r1 as i8)),
            op::SPLAT_W => Vector(_mm512_set1_epi16(r1 as i16)),
            op::SPLAT_D => Vector(_mm512_set1_epi32(r1 as i32)),
            op::SPLAT_Q => Vector(_mm512_set1_epi64(r1 as i64)),
            op::ZERO => Vector(_mm512_setzero_si512()),
            op::MOV => Vector(a),
            op::ADD_B => Vector(_mm512_add_epi8(a, b)),
            op::ADD_W => Vector(_mm512_add_epi16(a, b)),
            op::ADD_D => Vector(_mm512_add_epi32(a, b)),
            op::ADD_Q => Vector(_mm512_add_epi64(a, b)),
            op::SUB_B => Vector(_mm512_sub_epi8(a, b)),
            op::SUB_W => Vector(_mm512_sub_epi16(a, b)),
            op::SUB_D => Vector(_mm512_sub_epi32(a, b)),
            op::SUB_Q => Vector(_mm512_sub_epi64(a, b)),
            op::MULLO_W => Vector(_mm512_mullo_epi16(a, b)),
            op::MULLO_D => Vector(_mm512_mullo_epi32(a, b)),
            op::MULLO_Q => Vector(_mm512_mullo_epi64(a, b)),
            op::MULWU => Vector(_mm512_mul_epu32(a, b)),
            op::MULWS => Vector(_mm512_mul_epi32(a, b)),
            op::MULHI_W => Vector(_mm512_mulhi_epi16(a, b)),
            op::MULHIU_W => Vector(_mm512_mulhi_epu16(a, b)),
            op::MAX_D => Vector(_mm512_max_epi32(a, b)),
            op::MIN_D => Vector(_mm512_min_epi32(a, b)),
            op::MAXU_D => Vector(_mm512_max_epu32(a, b)),
            op::MINU_D => Vector(_mm512_min_epu32(a, b)),
            op::MAX_W => Vector(_mm512_max_epi16(a, b)),
            op::MIN_W => Vector(_mm512_min_epi16(a, b)),
            op::MAXU_Q => Vector(_mm512_max_epu64(a, b)),
            op::MINU_Q => Vector(_mm512_min_epu64(a, b)),
            op::MAXU_W => Vector(_mm512_max_epu16(a, b)),
            op::MINU_W => Vector(_mm512_min_epu16(a, b)),
            op::AND => Vector(_mm512_and_si512(a, b)),
            op::OR => Vector(_mm512_or_si512(a, b)),
            op::XOR => Vector(_mm512_xor_si512(a, b)),
            op::ANDN => Vector(_mm512_andnot_si512(a, b)),
            op::TERNLOG => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_ternarylogic_epi64::<I>(c, a, b)),
            ),
            op::SHLI_W => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], _mm512_slli_epi16::<U>(a)),
            ),
            op::SHLI_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_slli_epi32::<U>(a)),
            ),
            op::SHLI_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_slli_epi64::<U>(a)),
            ),
            op::SHRI_W => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], _mm512_srli_epi16::<U>(a)),
            ),
            op::SHRI_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_srli_epi32::<U>(a)),
            ),
            op::SHRI_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_srli_epi64::<U>(a)),
            ),
            op::SARI_W => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], _mm512_srai_epi16::<U>(a)),
            ),
            op::SARI_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_srai_epi32::<U>(a)),
            ),
            op::SARI_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_srai_epi64::<U>(a)),
            ),
            op::ROLI_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_rol_epi32::<I>(a)),
            ),
            op::ROLI_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_rol_epi64::<I>(a)),
            ),
            op::SHLV_D => Vector(_mm512_sllv_epi32(a, b)),
            op::SHLV_Q => Vector(_mm512_sllv_epi64(a, b)),
            op::SHRV_D => Vector(_mm512_srlv_epi32(a, b)),
            op::SHRV_Q => Vector(_mm512_srlv_epi64(a, b)),
            op::ROLV_D => Vector(_mm512_rolv_epi32(a, b)),
            op::ROLV_Q => Vector(_mm512_rolv_epi64(a, b)),
            op::CMPEQ_B => Vector(_mm512_movm_epi8(_mm512_cmpeq_epi8_mask(a, b))),
            op::CMPEQ_D => Vector(_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(a, b))),
            op::CMPEQ_Q => Vector(_mm512_movm_epi64(_mm512_cmpeq_epi64_mask(a, b))),
            op::CMPGT_D => Vector(_mm512_movm_epi32(_mm512_cmpgt_epi32_mask(a, b))),
            op::CMPGT_Q => Vector(_mm512_movm_epi64(_mm512_cmpgt_epi64_mask(a, b))),
            op::CMPGTU_Q => Vector(_mm512_movm_epi64(_mm512_cmpgt_epu64_mask(a, b))),
            op::DPBUSD => Vector(_mm512_dpbusd_epi32(c, a, b)),
            op::DPWSSD => Vector(_mm512_dpwssd_epi32(c, a, b)),
            op::MADDUBS => Vector(_mm512_maddubs_epi16(a, b)),
            op::MADDWD => Vector(_mm512_madd_epi16(a, b)),
            op::MADD52LO => Vector(_mm512_madd52lo_epu64(c, a, b)),
            op::MADD52HI => Vector(_mm512_madd52hi_epu64(c, a, b)),
            op::PACKS_D => Vector(_mm512_packs_epi32(a, b)),
            op::PACKS_W => Vector(_mm512_packs_epi16(a, b)),
            op::PACKUS_W => Vector(_mm512_packus_epi16(a, b)),
            op::PERMD => Vector(_mm512_permutexvar_epi32(a, b)),
            op::PERMQ => Vector(_mm512_permutexvar_epi64(a, b)),
            op::SHUFB => Vector(_mm512_shuffle_epi8(a, b)),
            op::PERM2_W => Vector(_mm512_permutex2var_epi16(c, a, b)),
            op::NARROWUS_D_B => Vector(_mm512_zextsi128_si512(_mm512_cvtusepi32_epi8(a))),
            op::NARROWS_D_B => Vector(_mm512_zextsi128_si512(_mm512_cvtsepi32_epi8(a))),
            op::NARROWS_D_W => Vector(_mm512_zextsi256_si512(_mm512_cvtsepi32_epi16(a))),
            op::NARROW_D_B => Vector(_mm512_zextsi128_si512(_mm512_cvtepi32_epi8(a))),
            op::ZX_B_D => Vector(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(a))),
            op::SX_B_D => Vector(_mm512_cvtepi8_epi32(_mm512_castsi512_si128(a))),
            op::ZX_W_D => Vector(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(a))),
            op::SX_W_D => Vector(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(a))),
            op::ZX_D_Q => Vector(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(a))),
            op::SX_D_Q => Vector(_mm512_cvtepi32_epi64(_mm512_castsi512_si256(a))),
            op::POPCNT_B => Vector(_mm512_popcnt_epi8(a)),
            op::POPCNT_Q => Vector(_mm512_popcnt_epi64(a)),
            op::HSUM_D => Scalar(_mm512_reduce_add_epi32(a) as u32 as u64),
            op::HSUM_Q => Scalar(_mm512_reduce_add_epi64(a) as u64),
            op::EXTRACT_Q => Scalar(
                with_imm!(imm8, [0 1 2 3 4 5 6 7], _mm_extract_epi64::<QE>(_mm512_extracti32x4_epi32::<QL>(a)))
                    as u64,
            ),
            op::EXTRACT_D => Scalar(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], _mm_extract_epi32::<DE>(_mm512_extracti32x4_epi32::<DL>(a)))
                    as u32 as u64,
            ),
            op::INSERT_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7], _mm512_inserti32x4::<QL>(c, _mm_insert_epi64::<QE>(_mm512_extracti32x4_epi32::<QL>(c), r1 as i64))),
            ),
            op::MOVEMASK_B => Scalar(_mm512_movepi8_mask(a)),
            op::CLMUL => Vector(with_imm!(imm8, [0 1 16 17], _mm512_clmulepi64_epi128::<I>(a, b))),
            _ => unreachable!("{:#x}", op),
        }
    }

    /// The reference semantics through the same entry points the interpreter uses
    fn reference(
        op: u8,
        imm8: u8,
        c: &VectorRegister,
        a: &VectorRegister,
        b: &VectorRegister,
        r1: u64,
    ) -> (Option<VectorRegister>, Option<u64>) {
        let vector = execute_register_op(op, imm8, c, a, b)
            .or_else(|| execute_scalar_input_op(op, imm8, c, r1));
        (vector, execute_scalar_output_op(op, imm8, a))
    }

    #[test]
    fn test_vector_ops_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0xa5a5);
        let mut checked = 0;
        for info in OPS
            .iter()
            .filter(|info| !matches!(info.op, op::LD | op::ST | op::LDBC_D | op::LDBC_Q))
        {
            for iteration in 0..2000 {
                let (c, a, b) = (
                    VectorRegister(random_vector(&mut rng)),
                    VectorRegister(random_vector(&mut rng)),
                    VectorRegister(random_vector(&mut rng)),
                );
                // Cover every valid imm8, then random ones
                let imm8 = if iteration < 256 && info.imm8_is_valid(iteration as u8) {
                    iteration as u8
                } else {
                    random_imm8(&mut rng, info.op)
                };
                let r1: u64 = if rng.gen_bool(0.2) {
                    [0, u64::MAX, 0x80, 0x8000_0000][rng.gen_range(0..4)]
                } else {
                    rng.gen()
                };
                let expected = unsafe {
                    native(
                        info.op,
                        imm8,
                        to_native(&c),
                        to_native(&a),
                        to_native(&b),
                        r1,
                    )
                };
                let actual = reference(info.op, imm8, &c, &a, &b, r1);
                match expected {
                    Native::Vector(v) => assert_eq!(
                        actual,
                        (Some(from_native(v)), None),
                        "{} imm8={:#x}\nc={:?}\na={:?}\nb={:?}",
                        info.name,
                        imm8,
                        c,
                        a,
                        b
                    ),
                    Native::Scalar(x) => assert_eq!(
                        actual,
                        (None, Some(x)),
                        "{} imm8={:#x}\na={:?}",
                        info.name,
                        imm8,
                        a
                    ),
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 91 * 2000);
    }

    /// The memory ops (and splat) through the interpreter, against native loads and stores
    #[test]
    fn test_vector_memory_ops_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0x3c3c);
        for _ in 0..200 {
            let mut input = vec![0u8; 256];
            rng.fill(&mut input[..]);
            let offset = rng.gen_range(0..128);
            let address = ebpf::MM_INPUT_START + offset as u64;
            let mut words = lddw(1, address).to_vec();
            words.extend([
                vop(op::LD, 0, 1, 0, 0),
                vop(op::LDBC_D, 0, 2, 0, 0),
                vop(op::LDBC_Q, 0, 3, 0, 0),
                vop(op::ST, 0, 0, 3, 0),
                exit(),
            ]);
            let outcome = run(
                &verified(&words, config(true, true)),
                &input,
                Input::Writable,
                100,
                false,
            );
            assert_eq!(outcome.result, "Ok(0)");
            unsafe {
                let p = input.as_mut_ptr().add(offset);
                assert_eq!(
                    outcome.vector_registers[1],
                    from_native(_mm512_loadu_si512(p.cast()))
                );
                assert_eq!(
                    outcome.vector_registers[2],
                    from_native(_mm512_set1_epi32(p.cast::<i32>().read_unaligned()))
                );
                let broadcast = _mm512_set1_epi64(p.cast::<i64>().read_unaligned());
                assert_eq!(outcome.vector_registers[3], from_native(broadcast));
                _mm512_storeu_si512(p.cast(), broadcast);
            }
            assert_eq!(outcome.memory, input);
        }
    }
}
