#![allow(clippy::arithmetic_side_effects)]
//! Tests of the 512-bit vector ops: verifier, interpreter vs JIT, faults, host calls.
//!
//! The comparison of the reference semantics against native AVX-512 is in `mod native` below.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use solana_sbpf::{
    declare_builtin_function, ebpf,
    elf::Executable,
    error::EbpfError,
    memory_region::{AccessType, AccessViolationHandler, MemoryRegion},
    program::{BuiltinFunctionDefinition, BuiltinProgram, FunctionRegistry, SBPFVersion},
    vector::{
        encode, encode_masked, encode_with_offset, op, op_info, Kind, Mask, OpInfo, VectorRegister,
        OPS, VECTOR_REGISTERS,
    },
    verifier::{RequisiteVerifier, VerifierError},
    vm::{CallFrame, Config, ExecutionMode},
};
use std::{cell::RefCell, convert::TryInto, rc::Rc, sync::Arc};
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
    /// Calls of the access violation handler: access type, address and length
    handler_calls: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Writable,
    ReadOnly,
    /// Read-only, the access violation handler makes it writable on the first store
    CopyOnWrite,
    /// Writable, 64 byte frames with 64 byte gaps in between
    Gapped,
    /// Writable, mapped at MM_INPUT_START + 64
    Shifted,
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
    let handler_calls = Rc::new(RefCell::new(vec![]));
    let (region, handler): (MemoryRegion, Option<AccessViolationHandler>) = match kind {
        Input::Writable => (
            MemoryRegion::new(&raw mut memory[..], ebpf::MM_INPUT_START),
            None,
        ),
        Input::ReadOnly => (
            MemoryRegion::new(&raw const memory[..], ebpf::MM_INPUT_START),
            None,
        ),
        Input::Gapped => (
            MemoryRegion::new_gapped(&raw mut memory[..], ebpf::MM_INPUT_START, 64),
            None,
        ),
        Input::Shifted => (
            MemoryRegion::new(&raw mut memory[..], ebpf::MM_INPUT_START + 64),
            None,
        ),
        Input::CopyOnWrite => {
            let mut region = MemoryRegion::new(&raw const memory[..], ebpf::MM_INPUT_START);
            region.access_violation_handler_payload = Some(1);
            let writable = &raw mut copy[..];
            let calls = handler_calls.clone();
            let handler: AccessViolationHandler =
                Box::new(move |region, _max_len, access_type, vm_addr, len| {
                    calls
                        .borrow_mut()
                        .push(format!("{access_type:?} {vm_addr:#x} {len}"));
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
        handler_calls: handler_calls.take(),
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

fn random_mask(rng: &mut SmallRng) -> u64 {
    match rng.gen_range(0..7) {
        0 => 0,
        1 => u64::MAX,
        2 => 0x5555_5555_5555_5555,
        3 => 1 << rng.gen_range(0..64),
        4 => !(1 << rng.gen_range(0..64)),
        _ => rng.gen(),
    }
}

/// The implemented ops
fn implemented() -> impl Iterator<Item = &'static OpInfo> {
    OPS.iter().filter(|info| info.implemented)
}

/// The allowed mask modes of an op
fn modes(info: &OpInfo) -> Vec<u8> {
    (0..3)
        .filter(|mode| info.modes() & (1 << mode) != 0)
        .collect()
}

fn stxdw(dst: u8, src: u8, off: i16) -> u64 {
    insn(ebpf::ST_DW_REG, dst, src, off, 0)
}

/// Every implemented op in every allowed mask mode, interpreter vs JIT with random inputs, masks
/// and register assignments. The comparison covers the whole state: all vector registers, the
/// scalar registers (stored to memory at the end), memory, CUs, pc and the result.
#[test]
fn test_vector_ops_interpreter_vs_jit() {
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    let register_sets = [
        (3, 4, 5),
        (29, 16, 8),
        (0, 0, 1),
        (17, 17, 17),
        (15, 31 % 30, 24),
        (1, 2, 0),
        (20, 0, 21),
        (16, 18, 16),
    ];
    // Input: three vectors, the memory the op accesses, then the scalar registers at the end
    const AREA: usize = 192;
    const AREA_LEN: usize = 2048 + 128;
    const DUMP: usize = AREA + AREA_LEN;
    for info in implemented() {
        let modes = modes(info);
        for iteration in 0..48 {
            let (out, in0, in1) = register_sets[iteration % register_sets.len()];
            let mode = modes[iteration % modes.len()];
            let kind = info.kind;
            let imm8 = random_imm8(&mut rng, info.op);
            let (mut out, mut in0, in1) = (
                if kind.uses_out() { out } else { 0 },
                if kind.uses_in0() { in0 } else { 0 },
                if kind.uses_in1() { in1 } else { 0 },
            );
            if info.imm8_is_count() {
                // ldn / stn: the registers base..base + n must exist
                out = out.min(30 - imm8) * kind.uses_out() as u8;
                in0 = in0.min(30 - imm8) * kind.uses_in0() as u8;
            }
            let gather_scatter = matches!(kind, Kind::Ga | Kind::Sc);
            // Gathers and scatters with small indices stay in the area, others likely fault
            let wild = gather_scatter && iteration % 8 == 7;
            let mut input = vec![0u8; DUMP + 10 * 8];
            for k in 0..3 {
                input[64 * k..64 * (k + 1)].copy_from_slice(&random_vector(&mut rng));
            }
            if gather_scatter && !wild {
                for lane in input[64..128].chunks_mut(8) {
                    lane.copy_from_slice(&rng.gen_range(-6i64..6).to_le_bytes());
                }
            }
            rng.fill(&mut input[AREA..DUMP]);
            let mut words = vec![mov64_reg(6, 1)];
            // Load the operands, the last one wins for aliased registers: in1 (or the index)
            let order = if gather_scatter {
                [(0, out), (2, in1), (1, in0)]
            } else {
                [(0, out), (1, in0), (2, in1)]
            };
            for (k, reg) in order {
                words.push(mov64_reg(1, 6));
                words.push(add64_imm(1, 64 * k));
                words.push(vop(op::LD, 0, reg, 0, 0));
            }
            // r1 + off: an address inside the area (for memory ops), otherwise a random value
            let (r1, off) = if info.accesses_memory() {
                let off: i16 = match iteration % 4 {
                    0 => 0,
                    1 => rng.gen_range(-256..256),
                    _ => rng.gen(),
                };
                let address = ebpf::MM_INPUT_START
                    + AREA as u64
                    + if gather_scatter { 512 } else { 0 }
                    + rng.gen_range(0..64);
                (address.wrapping_sub(off as i64 as u64), off)
            } else {
                let r1 = random_vector(&mut rng)[..8]
                    .iter()
                    .fold(0, |acc, b| acc << 8 | *b as u64);
                (r1, 0)
            };
            words.extend(lddw(1, r1));
            words.extend(lddw(2, random_mask(&mut rng)));
            words.push(mov64_imm(0, 0x1234));
            words.push(encode_masked(info.op, imm8, out, in0, in1, off, mode));
            for reg in 0..10 {
                words.push(stxdw(6, reg, (DUMP + 8 * reg as usize) as i16));
            }
            words.push(exit());
            let executable = verified(&words, config(true, iteration % 2 == 0));
            let outcome = run_both(&executable, &input, Input::Writable, 1000);
            assert!(
                wild || outcome.result.starts_with("Ok("),
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
        // 1, 2, 16 and 32 byte accesses at the end of the input
        (
            op::LDBC_B,
            ebpf::MM_INPUT_START + 99,
            Input::Writable,
            "Ok(",
        ),
        (
            op::LDBC_B,
            ebpf::MM_INPUT_START + 100,
            Input::Writable,
            "AccessViolation(Load, 17179869284, 1, \"unallocated\")",
        ),
        (
            op::LDBC_W,
            ebpf::MM_INPUT_START + 98,
            Input::Writable,
            "Ok(",
        ),
        (
            op::LDBC_W,
            ebpf::MM_INPUT_START + 99,
            Input::Writable,
            "AccessViolation(Load, 17179869283, 2, \"input\")",
        ),
        (op::LD_X, ebpf::MM_INPUT_START + 84, Input::Writable, "Ok("),
        (
            op::LD_X,
            ebpf::MM_INPUT_START + 85,
            Input::Writable,
            "AccessViolation(Load, 17179869269, 16, \"input\")",
        ),
        (
            op::LDBC_X,
            ebpf::MM_INPUT_START + 85,
            Input::Writable,
            "AccessViolation(Load, 17179869269, 16, \"input\")",
        ),
        (
            op::LDBC_X,
            ebpf::MM_INPUT_START + 3,
            Input::CopyOnWrite,
            "Ok(",
        ),
        (op::LD_Y, ebpf::MM_INPUT_START + 68, Input::Writable, "Ok("),
        (
            op::LD_Y,
            ebpf::MM_INPUT_START + 69,
            Input::Writable,
            "AccessViolation(Load, 17179869253, 32, \"input\")",
        ),
        (
            op::LDBC_Y,
            ebpf::MM_INPUT_START + 69,
            Input::Writable,
            "AccessViolation(Load, 17179869253, 32, \"input\")",
        ),
        (op::ST_X, ebpf::MM_INPUT_START + 84, Input::Writable, "Ok("),
        (
            op::ST_X,
            ebpf::MM_INPUT_START + 85,
            Input::Writable,
            "AccessViolation(Store, 17179869269, 16, \"input\")",
        ),
        (
            op::ST_X,
            ebpf::MM_INPUT_START,
            Input::ReadOnly,
            "AccessViolation(Store, 17179869184, 16, \"input\")",
        ),
        (op::ST_Y, ebpf::MM_INPUT_START + 68, Input::Writable, "Ok("),
        (
            op::ST_Y,
            ebpf::MM_INPUT_START + 69,
            Input::Writable,
            "AccessViolation(Store, 17179869253, 32, \"input\")",
        ),
        (
            op::ST_Y,
            ebpf::MM_INPUT_START + 5,
            Input::CopyOnWrite,
            "Ok(",
        ),
        (op::ST_Y, 0, Input::Writable, "AccessViolation(Store, 0, 32"),
    ];
    // The same effective addresses as r1 + off with various displacements
    let offsets: [i16; 6] = [0, 1, -1, 0x1230, i16::MIN, i16::MAX];
    for (aligned, (op, address, input_kind, expected), off) in [false, true]
        .iter()
        .copied()
        .flat_map(|a| cases.iter().map(move |c| (a, c)))
        .flat_map(|(a, c)| offsets.iter().map(move |off| (a, c, *off)))
    {
        let store = op_info(*op).unwrap().kind == Kind::St;
        let mut words = preload(9);
        words.extend(lddw(1, address.wrapping_sub(off as i64 as u64)));
        words.push(encode_with_offset(
            *op,
            0,
            if store { 0 } else { 9 },
            if store { 9 } else { 0 },
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
        if store && expected.starts_with("Ok(") {
            let offset = (*address - ebpf::MM_INPUT_START) as usize;
            let len = match *op {
                op::ST_X => 16,
                op::ST_Y => 32,
                _ => 64,
            };
            let mut expected_memory = input.clone();
            expected_memory[offset..offset + len]
                .copy_from_slice(&0x0123_4567_89ab_cdef_u64.to_le_bytes().repeat(8)[..len]);
            assert_eq!(outcome.memory, expected_memory);
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
    let invalid = [vop(0xff, 0, 1, 0, 0), exit()];
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
    // All implemented ops with valid fields pass, the others are rejected
    for info in OPS {
        let kind = info.kind;
        let imm8 = (0..=255).find(|i| info.imm8_is_valid(*i)).unwrap();
        let mode = modes(info)[0];
        // ldn / stn need room for imm8 registers above the base
        let base = if info.imm8_is_count() { 30 - imm8 } else { 29 };
        let (out, in0) = if kind.uses_out() {
            (base, 28)
        } else {
            (0, base)
        };
        let word = encode_masked(
            info.op,
            imm8,
            out,
            kind.uses_in0() as u8 * in0,
            kind.uses_in1() as u8 * 27,
            0,
            mode,
        );
        if !info.implemented {
            rejected(word, "unimplemented op");
            continue;
        }
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
            rejected(word | 0x1f << 42, "register field out of range");
        }
        if kind.uses_in0() {
            rejected(word | 31 << (32 + 5), "register field out of range");
        }
        if kind.uses_in1() {
            rejected(
                word & !(0x1f << 32) | 30 << 32,
                "register field out of range",
            );
        }
        // imm8 range
        for imm8 in 0..=255u8 {
            let word = word & !(0xff << 48) | (imm8 as u64) << 48;
            if !info.imm8_is_valid(imm8) {
                rejected(word, "imm8 out of range");
            } else if info.imm8_is_count() && base as usize + imm8 as usize > 30 {
                rejected(word, "register range out of range");
            } else {
                verify_vector_op(word).unwrap();
            }
        }
        // Mask modes (the dst field)
        for mode in 0..16u8 {
            let word = word & !(0xf << 8) | (mode as u64) << 8;
            if mode >= 10 {
                // The generic register checks of the verifier come first
                assert!(verify_vector_op(word).is_err());
            } else if mode >= 3 {
                rejected(word, "mask mode out of range");
            } else if info.modes() & (1 << mode) == 0 {
                rejected(word, "mask mode not allowed");
            } else {
                verify_vector_op(word).unwrap();
            }
        }
    }
    // Which ops take which modes
    let allowed = |op: u8| modes(op_info(op).unwrap());
    assert_eq!(allowed(op::ADD_B), [0, 1, 2]);
    assert_eq!(allowed(op::LD), [0]);
    assert_eq!(allowed(op::CMP), [0]);
    assert_eq!(allowed(op::BLENDM), [0]);
    assert_eq!(allowed(op::LD_M), [1, 2]);
    assert_eq!(allowed(op::ST_M), [1]);
    assert_eq!(allowed(op::GATHER_QQ), [0, 1, 2]);
    assert_eq!(allowed(op::SCATTER_QQ), [0, 1]);
    for op in [op::GATHER_DD, op::GATHER_DQ, op::GATHER_QD] {
        assert_eq!(allowed(op), [0, 1, 2]);
    }
    for op in [op::SCATTER_DD, op::SCATTER_DQ, op::SCATTER_QD] {
        assert_eq!(allowed(op), [0, 1]);
    }
    for op in [op::LD_X, op::LD_Y, op::ST_X, op::ST_Y] {
        assert_eq!(allowed(op), [0]);
    }
    for op in [op::LDBC_B, op::LDBC_W, op::LDBC_X, op::LDBC_Y] {
        assert_eq!(allowed(op), [0, 1, 2]);
    }
    // Gather and scatter scales are 1, 2, 4 and 8
    for op in [op::GATHER_DD, op::SCATTER_QD] {
        rejected(vop(op, 4, 0, 1, 0), "imm8 out of range");
    }
    assert_eq!(allowed(op::SHA256RNDS2), [0]);
    assert_eq!(allowed(op::COMPRESS), [1, 2]);
    assert_eq!(allowed(op::EXPAND), [1, 2]);
    assert_eq!(allowed(op::PERM2I), [0, 1, 2]);
    assert_eq!(allowed(op::BSHLI), [0]);
    assert_eq!(allowed(op::INSERT_B), [0]);
    assert_eq!(allowed(op::EXTRACT_W), [0]);
    assert_eq!(allowed(op::MOVR_Q), [0]);
    // Data movement imm8 ranges: (op, first invalid imm8, or 0 for any imm8)
    for (op, limit) in [
        (op::BCAST_Q, 8),
        (op::BCAST_D, 16),
        (op::BCAST_W, 32),
        (op::BCAST_B, 64),
        (op::PERMQI, 0),
        (op::EXTRACT256, 2),
        (op::PERM2I, 4),
        (op::ALIGN_Q, 8),
        (op::ALIGN_D, 16),
        (op::EXTRACT128, 4),
        (op::INSERT256, 2),
        (op::INSERT128, 4),
        (op::SHUFD, 0),
        (op::SHUFHW, 0),
        (op::SHUFLW, 0),
        (op::SHUF128, 0),
        (op::SHUFPS, 0),
        (op::ALIGNR, 32),
        (op::BSHLI, 16),
        (op::BSHRI, 16),
        (op::UNPACKLO, 4),
        (op::UNPACKHI, 4),
        (op::COMPRESS, 4),
        (op::EXPAND, 4),
        (op::EXTRACT_B, 64),
        (op::EXTRACT_W, 32),
        (op::INSERT_B, 64),
        (op::INSERT_W, 32),
        (op::INSERT_D, 16),
    ] {
        let info = op_info(op).unwrap();
        for imm8 in 0..=255u8 {
            assert_eq!(
                info.imm8_is_valid(imm8),
                limit == 0 || imm8 < limit,
                "{} {imm8}",
                info.name
            );
        }
    }
    // perm2t: W = 1 is perm2.w
    let perm2t = op_info(op::PERM2T).unwrap();
    assert!((0..=255u8).all(|imm8| perm2t.imm8_is_valid(imm8) == matches!(imm8, 0 | 2 | 3)));
    rejected(vop(op::PERM2T, 1, 1, 2, 3), "imm8 out of range");
    rejected(vop(op::COMPRESS, 2, 1, 2, 0), "mask mode not allowed");
    rejected(vop(op::BSHRI, 1, 1, 2, 0) | 2 << 8, "mask mode not allowed");
    rejected(
        vop(op::EXTRACT_B, 1, 0, 2, 3),
        "unused register field must be 0",
    );
    rejected(
        vop(op::INSERT_W, 1, 1, 2, 0),
        "unused register field must be 0",
    );
    rejected(vop(op::MOVR_Q, 1, 1, 0, 0), "imm8 out of range");
    assert_eq!(allowed(op::TEST), [0]);
    assert_eq!(allowed(op::MOVM_Q), [0]);
    assert_eq!(allowed(op::SHLDV_W), [0, 1, 2]);
    assert_eq!(allowed(op::CONFLICT_D), [0, 1, 2]);
    // ldn / stn register ranges
    for (op, n, base, valid) in [
        (op::LDN, 2, 28, true),
        (op::LDN, 2, 29, false),
        (op::LDN, 30, 0, true),
        (op::LDN, 30, 1, false),
        (op::STN, 3, 27, true),
        (op::STN, 3, 28, false),
    ] {
        let (out, in0) = if op == op::LDN { (base, 0) } else { (0, base) };
        let word = vop(op, n, out, in0, 0);
        if valid {
            verify_vector_op(word).unwrap();
        } else {
            rejected(word, "register range out of range");
        }
    }
    for n in [0, 1, 31, 255] {
        rejected(vop(op::LDN, n, 0, 0, 0), "imm8 out of range");
    }
    // cmp: predicates 3 and 7 and imm8[7:6] are rejected
    let cmp = op_info(op::CMP).unwrap();
    for imm8 in 0..=255u8 {
        let valid = imm8 >> 6 == 0 && imm8 & 7 != 3 && imm8 & 7 != 7;
        assert_eq!(cmp.imm8_is_valid(imm8), valid, "{imm8:#x}");
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
    assert!(
        op_info(op::SHRDI_W).unwrap().imm8_is_valid(15)
            && !op_info(op::SHRDI_W).unwrap().imm8_is_valid(16)
    );
    assert!(
        op_info(op::TESTN).unwrap().imm8_is_valid(3)
            && !op_info(op::TESTN).unwrap().imm8_is_valid(4)
    );
    // Unknown ops
    for op in 0..=255u8 {
        if op_info(op).is_none() {
            rejected(vop(op, 0, 0, 0, 0), "unknown op");
        }
    }
    // dst (mask mode), off and imm bit 15
    rejected(
        vop(op::ADD_D, 0, 1, 2, 3) | 3 << 8,
        "mask mode out of range",
    );
    rejected(vop(op::MOV, 0, 1, 2, 0) | 1 << 8, "mask mode not allowed");
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

// Memory runs: consecutive vector ops with several accesses share one address translation in the
// JIT (aligned memory mapping only)

fn jump(opc: u8, dst: u8, imm: i32, off: i16) -> u64 {
    insn(opc, dst, 0, off, imm)
}
fn ld(out: u8, off: i16) -> u64 {
    encode_with_offset(op::LD, 0, out, 0, 0, off)
}
fn st(in0: u8, off: i16) -> u64 {
    encode_with_offset(op::ST, 0, 0, in0, 0, off)
}
fn ldbc_d(out: u8, off: i16) -> u64 {
    encode_with_offset(op::LDBC_D, 0, out, 0, 0, off)
}
fn ldbc_q(out: u8, off: i16) -> u64 {
    encode_with_offset(op::LDBC_Q, 0, out, 0, 0, off)
}

fn run_config(aligned: bool, checkpoint_distance: usize) -> Config {
    Config {
        instruction_meter_checkpoint_distance: checkpoint_distance,
        ..config(true, aligned)
    }
}

/// Whether the JIT turned the instruction at `pc` into the inside of a memory run: callx and
/// jumps to it then enter through `xor r11d, r11d; jmp`
fn inside_memory_run(executable: &Executable<TestContextObject>, pc: usize) -> bool {
    let program = executable.get_compiled_program().unwrap();
    let code = &program.text_section()[program.pc_section()[pc] as usize..];
    code.starts_with(&[0x45, 0x31, 0xdb, 0xe9]) || code.starts_with(&[0x45, 0x31, 0xdb, 0x90, 0xe9])
}

/// Whether the JIT emits memory runs (the tracer disables them)
fn memory_runs_available() -> bool {
    jit_available() && !cfg!(feature = "tracer")
}

/// Runs the program in the interpreter and the JIT with both memory mappings and checks they all
/// agree. With the unaligned mapping the JIT emits no memory runs and checks the instruction meter
/// at the same accesses, so the JIT vector registers must agree even if it runs out.
fn run_with_and_without_memory_runs(
    words: &[u64],
    input: &[u8],
    kind: Input,
    budget: u64,
    checkpoint_distance: usize,
) -> Outcome {
    let mut outcomes = vec![];
    for aligned in [false, true] {
        let executable = verified(words, run_config(aligned, checkpoint_distance));
        let interpreted = run_both(&executable, input, kind, budget);
        let jitted = jit_available().then(|| run(&executable, input, kind, budget, true));
        outcomes.push((interpreted, jitted));
    }
    let (with_runs, without_runs) = (outcomes.pop().unwrap(), outcomes.pop().unwrap());
    assert_eq!(
        without_runs.0, with_runs.0,
        "the mapping changed the result"
    );
    assert_eq!(
        without_runs.1, with_runs.1,
        "memory runs changed the JIT result"
    );
    with_runs.0
}

fn input_pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + 3) as u8).collect()
}

fn broadcast<const N: usize>(bytes: &[u8]) -> Vec<u8> {
    bytes[..N].repeat(64 / N)
}

/// A run which faults at one of its accesses: the earlier ones take effect, the error, pc and CUs
/// are those of the faulting access
#[test]
fn test_memory_run_faults() {
    let input = input_pattern(200);
    // (r1, displacement of the last access, faulting access, expected error)
    let cases: [(i64, i16, Option<usize>, &str); 7] = [
        (0, 136, None, "Ok(9)"),
        (
            0,
            137,
            Some(3),
            "AccessViolation(Store, 17179869321, 64, \"input\")",
        ),
        (
            0,
            150,
            Some(3),
            "AccessViolation(Store, 17179869334, 64, \"input\")",
        ),
        // Below the start of the input region (a different region index)
        (
            0,
            -1,
            Some(3),
            "AccessViolation(Store, 17179869183, 64, \"unallocated\")",
        ),
        // r1 beyond the end: the first access faults
        (
            200,
            -200,
            Some(0),
            "AccessViolation(Store, 17179869384, 64, \"unallocated\")",
        ),
        // The span covers both sides of the end, the accesses do not cross it
        (64, 8, Some(2), "AccessViolation(Load, 17179869380, 8"),
        (-64, 64, None, "Ok(9)"),
    ];
    for (r1, last, faulting, expected) in cases {
        let mut words = vec![mov64_reg(6, 1), vop(op::SPLAT_Q, 0, 1, 0, 0)];
        words.push(add64_imm(1, r1 as i32));
        let run_start = words.len();
        words.extend([
            st(1, 0),
            ld(2, if r1 == 64 { -64 } else { 64 }),
            ldbc_q(3, if r1 == 64 { 132 } else { 100 }),
            st(2, last),
            mov64_imm(0, 9),
            exit(),
        ]);
        if r1 == -64 {
            // All accesses above the start, the first displacement is not the smallest
            words[run_start] = st(1, 64);
            words[run_start + 1] = ld(2, 128);
            words[run_start + 2] = ldbc_q(3, 164);
        }
        let outcome = run_with_and_without_memory_runs(&words, &input, Input::Writable, 100, 1000);
        assert!(
            outcome.result.contains(expected),
            "{r1} {last}: {}",
            outcome.result
        );
        if let Some(access) = faulting {
            assert_eq!(outcome.pc, (run_start + access) as u64, "{r1} {last}");
            assert_eq!(outcome.instruction_count, (run_start + access + 1) as u64);
        }
        if faulting.is_none_or(|access| access > 0) && r1 == 0 {
            let r1_bytes = ebpf::MM_INPUT_START.to_le_bytes().repeat(8);
            assert_eq!(&outcome.memory[..64], &r1_bytes[..]);
            assert_eq!(outcome.vector_registers[2].0, input[64..128]);
            assert_eq!(
                outcome.vector_registers[3].0[..],
                broadcast::<8>(&input[100..])[..]
            );
        }
        if memory_runs_available() {
            let executable = verified(&words, run_config(true, 1000));
            executable.jit_compile().unwrap();
            assert!(inside_memory_run(&executable, run_start + 1));
        }
    }
}

/// Runs in every kind of region: stores to read-only memory fault, stores to copy-on-write memory
/// redirect the region in the middle of a run, gaps and a region which does not start at its
/// region index fault like single accesses
#[test]
fn test_memory_run_region_kinds() {
    let input = input_pattern(256);
    let words = [
        mov64_reg(6, 1),
        vop(op::SPLAT_Q, 0, 7, 0, 0),
        // Loads only: one translation even for read-only and copy-on-write
        ld(1, 0),
        ldbc_d(2, 132),
        ld(3, 128),
        // A store, then loads in the same run see the copy
        st(7, 64),
        ld(4, 64),
        ld(5, 0),
        st(5, 192),
        ld(6, 192),
        // Scalar store, then a load only run sees the redirected region
        insn(ebpf::ST_B_IMM, 1, 0, 1, 0x5a),
        ld(8, 0),
        ldbc_q(9, 1),
        vop(op::HSUM_D, 0, 0, 6, 0),
        exit(),
    ];
    for kind in [
        Input::Writable,
        Input::ReadOnly,
        Input::CopyOnWrite,
        Input::Gapped,
        Input::Shifted,
    ] {
        for budget in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 100] {
            let outcome = run_with_and_without_memory_runs(&words, &input, kind, budget, 1000);
            match (kind, budget) {
                (_, 100) if kind == Input::Writable || kind == Input::CopyOnWrite => {
                    assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
                    let mut expected = input.clone();
                    expected[64..128]
                        .copy_from_slice(&ebpf::MM_INPUT_START.to_le_bytes().repeat(8));
                    let mut v5 = input[..64].to_vec();
                    expected[192..256].copy_from_slice(&v5);
                    v5[1] = 0x5a;
                    expected[1] = 0x5a;
                    assert_eq!(outcome.memory, expected);
                    assert_eq!(outcome.vector_registers[4].0, expected[64..128]);
                    assert_eq!(outcome.vector_registers[6].0, expected[192..256]);
                    assert_eq!(outcome.vector_registers[8].0, v5[..]);
                }
                (Input::ReadOnly, 100) => {
                    assert!(outcome.result.contains("AccessViolation(Store"));
                    assert_eq!(outcome.pc, 5);
                    assert_eq!(outcome.vector_registers[3].0, input[128..192]);
                }
                (Input::Gapped, 100) => {
                    // Only even frames are mapped: the load at 128 maps the second frame
                    assert!(
                        outcome.result.contains("AccessViolation(Store"),
                        "{}",
                        outcome.result
                    );
                    assert_eq!(outcome.pc, 5);
                    assert_eq!(outcome.vector_registers[3].0, input[64..128]);
                }
                (Input::Shifted, 100) => {
                    // The first access is below the region, in the same region index
                    assert!(outcome
                        .result
                        .contains("AccessViolation(Load, 17179869184, 64"));
                    assert_eq!(outcome.pc, 2);
                }
                _ => {}
            }
        }
    }
}

/// A region longer than the region size of the aligned mapping: accesses which start beyond its
/// region index fault even though they are inside the region (the unaligned mapping has no such
/// limit), so a run must not translate across the index boundary
#[test]
fn test_memory_run_region_index_boundary() {
    let region_size = 1 << 20;
    let input = input_pattern(region_size + 256);
    let mut words = lddw(1, ebpf::MM_INPUT_START + region_size as u64 - 128).to_vec();
    words.extend([ld(1, 0), ld(2, 64), ld(3, 128), mov64_imm(0, 9), exit()]);
    for last in [127, 128] {
        words[4] = ld(3, last);
        for aligned in [false, true] {
            let executable = verified(
                &words,
                Config {
                    aligned_region_size: region_size,
                    ..config(true, aligned)
                },
            );
            let outcome = run_both(&executable, &input, Input::Writable, 100);
            if aligned && last == 128 {
                assert!(outcome
                    .result
                    .contains("AccessViolation(Load, 17180917760, 64"));
                assert_eq!(
                    outcome.vector_registers[2].0,
                    input[region_size - 64..region_size]
                );
            } else {
                assert_eq!(outcome.result, "Ok(9)");
            }
        }
    }
}

/// The instruction meter running out at any instruction of a run, with and without instruction
/// meter checkpoints inside the run
#[test]
fn test_memory_run_instruction_meter() {
    let input = input_pattern(320);
    let mut words = vec![mov64_reg(6, 1), vop(op::SPLAT_D, 0, 9, 0, 0)];
    for i in 0..12i16 {
        words.push(ld(i as u8 % 4, 16 * i));
        words.push(vop(op::ADD_D, 0, 8, 8, i as u8 % 4));
        words.push(st(8, 128 + 8 * i));
        if i % 5 == 4 {
            words.push(ldbc_q(9, 3 * i));
        }
    }
    words.push(vop(op::HSUM_D, 0, 0, 8, 0));
    words.push(exit());
    // Straight-line code, one CU per instruction
    let needed = words.len() as u64;
    for checkpoint_distance in [1000, 0, 1, 2, 3, 7] {
        for budget in 0..=needed {
            let outcome = run_with_and_without_memory_runs(
                &words,
                &input,
                Input::Writable,
                budget,
                checkpoint_distance,
            );
            if budget < needed {
                assert_eq!(outcome.result, "Err(ExceededMaxInstructions)", "{budget}");
            } else {
                assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
            }
        }
    }
    if memory_runs_available() {
        let executable = verified(&words, run_config(true, 1000));
        executable.jit_compile().unwrap();
        assert!((3..words.len() - 3).all(|pc| inside_memory_run(&executable, pc)));
    }
}

/// Anything between two vector ops ends a run: r1 may have changed and host code may have run
#[test]
fn test_memory_run_boundaries() {
    let input = input_pattern(512);
    let mut words = vec![
        mov64_reg(6, 1),
        ld(1, 0),
        ld(2, 64),
        add64_imm(1, 64),
        ld(3, 0),
        ld(4, 64),
        mov64_imm(1, 1),
        syscall(SYSCALL_CLOBBER),
        mov64_reg(1, 6),
        st(3, 256),
        st(4, 320),
        // The callee moves r1 and stores through it
        call_internal(0),
    ];
    let call = words.len() - 1;
    words.extend([
        ld(5, 0),
        ld(6, 64),
        vop(op::ADD_D, 0, 0, 5, 6),
        vop(op::HSUM_D, 0, 0, 0, 0),
        exit(),
    ]);
    let function = words.len();
    words[call] = call_internal((function - call - 1) as i32);
    words.extend([ld(7, 384), add64_imm(1, 128), st(7, 0), ld(8, 0), exit()]);
    for kind in [Input::Writable, Input::CopyOnWrite] {
        let outcome = run_with_and_without_memory_runs(&words, &input, kind, 1000, 1000);
        assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
        assert_eq!(outcome.vector_registers[3].0, input[64..128]);
        assert_eq!(outcome.vector_registers[4].0, input[128..192]);
        assert_eq!(outcome.vector_registers[5].0, input[384..448]);
        assert_eq!(&outcome.memory[128..192], &input[384..448]);
        assert_eq!(&outcome.memory[256..320], &input[64..128]);
    }
}

/// Jumps and calls into a run: a jump target starts a new run, callx may enter anywhere
#[test]
fn test_memory_run_entered_in_the_middle() {
    let input = input_pattern(256);
    let text_vaddr = executable(&[exit()], config(true, true), SBPFVersion::V3)
        .get_text_bytes()
        .0;
    let (head, target, last_exit) = (4, 8, 20);
    let mut words = vec![
        mov64_reg(6, 1),
        mov64_imm(7, 3),
        mov64_imm(8, 0),
        vop(op::SPLAT_D, 0, 9, 0, 0),
        // Three iterations, the loop head starts a run
        ld(1, 0),
        ld(2, 64),
        vop(op::ADD_D, 0, 1, 1, 9),
        st(1, 0),
        ld(3, 128),
        vop(op::ADD_D, 0, 3, 3, 2),
        st(3, 128),
        add64_imm(7, -1),
        jump(ebpf::JNE64_IMM, 7, 0, head as i16 - 13),
        // The callx below returns from here
        jump(ebpf::JNE64_IMM, 8, 0, last_exit as i16 - 14),
        mov64_imm(8, 1),
        mov64_imm(7, 1),
    ];
    // callx into the middle of the run
    words.extend(lddw(2, text_vaddr + 8 * target as u64));
    words.extend([
        insn(ebpf::CALL_REG, 2, 0, 0, 0),
        vop(op::HSUM_D, 0, 0, 3, 0),
        exit(),
    ]);
    assert_eq!(words.len(), last_exit + 1);
    for budget in [1000, 30, 40, 41, 42, 43, 44, 45] {
        let outcome =
            run_with_and_without_memory_runs(&words, &input, Input::Writable, budget, 1000);
        if budget == 1000 {
            assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
            // The callx adds v2 to v3 a fourth time
            let lane = |bytes: &[u8], i: usize| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            };
            for i in 0..16 {
                assert_eq!(
                    lane(&outcome.memory[128..192], i),
                    lane(&input[128..192], i)
                        .wrapping_add(lane(&input[64..128], i).wrapping_mul(4))
                );
            }
        }
    }
    if memory_runs_available() {
        let executable = verified(&words, run_config(true, 1000));
        executable.jit_compile().unwrap();
        assert!(!inside_memory_run(&executable, head));
        assert!(inside_memory_run(&executable, head + 1) && inside_memory_run(&executable, target));
    }
}

/// Random programs of runs: accesses near region boundaries, every kind of region, random budgets
#[test]
fn test_memory_runs_random() {
    let mut rng = SmallRng::seed_from_u64(0x7275_6e73);
    let input = input_pattern(320);
    for _ in 0..400 {
        let mut words = vec![mov64_reg(6, 1), vop(op::SPLAT_Q, 0, 29, 0, 0)];
        words.extend(lddw(2, random_mask(&mut rng)));
        for _ in 0..rng.gen_range(1..4) {
            words.push(mov64_reg(1, 6));
            let base = [0, 64, 256, 300, 320, -64, -8, 5000][rng.gen_range(0..8)];
            words.push(add64_imm(1, base));
            for _ in 0..rng.gen_range(1..10) {
                let off = if rng.gen_bool(0.8) {
                    rng.gen_range(-80..330)
                } else {
                    rng.gen()
                };
                let (out, in0) = (rng.gen_range(0..30), rng.gen_range(0..30));
                let (n, width, mode) = (
                    rng.gen_range(2..5),
                    rng.gen_range(0..4),
                    rng.gen_range(1..3),
                );
                words.push(match rng.gen_range(0..10) {
                    0 | 1 => ld(out, off),
                    2 => st(in0, off),
                    3 => ldbc_d(out, off),
                    4 => ldbc_q(out, off),
                    5 => encode_masked(op::LD_M, width, out, 0, 0, off, mode),
                    6 => encode_masked(op::ST_M, width, 0, in0, 0, off, 1),
                    7 => encode_with_offset(op::LDN, n, out.min(30 - n), 0, 0, off),
                    8 => encode_with_offset(op::STN, n, 0, in0.min(30 - n), 0, off),
                    _ => vop(op::ADD_Q, 0, out, in0, 29),
                });
            }
        }
        words.push(vop(op::HSUM_Q, 0, 0, 29, 0));
        words.push(exit());
        let kind = [
            Input::Writable,
            Input::ReadOnly,
            Input::CopyOnWrite,
            Input::Gapped,
            Input::Shifted,
        ][rng.gen_range(0..5)];
        let budget = if rng.gen_bool(0.5) {
            1000
        } else {
            rng.gen_range(0..words.len() as u64)
        };
        let checkpoint_distance = [1000, 1000, 2, 5][rng.gen_range(0..4)];
        run_with_and_without_memory_runs(&words, &input, kind, budget, checkpoint_distance);
    }
}

// Masked, multi register and indexed memory ops

/// Number of instructions in `words` (lddw is one instruction in two slots)
fn instruction_count(words: &[u64]) -> u64 {
    let (mut count, mut i) = (0, 0);
    while i < words.len() {
        i += if words[i] as u8 == ebpf::LD_DW_IMM {
            2
        } else {
            1
        };
        count += 1;
    }
    count
}

/// Puts 0x0123_4567_89ab_cdef into every qword of `regs`, `r6` = input
fn preload(regs: std::ops::Range<u8>) -> Vec<u64> {
    let mut words = vec![mov64_reg(6, 1)];
    words.extend(lddw(1, 0x0123_4567_89ab_cdef));
    for reg in regs {
        words.push(vop(op::SPLAT_Q, 0, reg, 0, 0));
    }
    words
}

const PRELOADED: u64 = 0x0123_4567_89ab_cdef;

/// ld.m and st.m check the whole 64 byte span, whatever the mask
#[test]
fn test_masked_memory_faults() {
    let input: Vec<u8> = input_pattern(100);
    for aligned in [false, true] {
        for (op, offset, mask, kind, expected) in [
            (op::LD_M, 36, u64::MAX, Input::Writable, "Ok("),
            (
                op::LD_M,
                37,
                u64::MAX,
                Input::Writable,
                "AccessViolation(Load, 17179869221, 64",
            ),
            (
                op::LD_M,
                37,
                1,
                Input::Writable,
                "AccessViolation(Load, 17179869221, 64",
            ),
            (
                op::LD_M,
                37,
                0,
                Input::Writable,
                "AccessViolation(Load, 17179869221, 64",
            ),
            (op::LD_M, 0, 0, Input::ReadOnly, "Ok("),
            (op::ST_M, 36, 0x8000_0001, Input::Writable, "Ok("),
            (
                op::ST_M,
                37,
                1,
                Input::Writable,
                "AccessViolation(Store, 17179869221, 64",
            ),
            (
                op::ST_M,
                0,
                0,
                Input::ReadOnly,
                "AccessViolation(Store, 17179869184, 64",
            ),
            (op::ST_M, 0, 0, Input::CopyOnWrite, "Ok("),
            (op::ST_M, 8, 0xf0f0, Input::CopyOnWrite, "Ok("),
        ] {
            for (width, mode) in [(0, 1), (1, 2), (2, 1), (3, 2)] {
                let mode = if op == op::ST_M { 1 } else { mode };
                let mut words = preload(1..3);
                words.extend(lddw(1, ebpf::MM_INPUT_START + offset));
                words.extend(lddw(2, mask));
                words.push(if op == op::LD_M {
                    encode_masked(op, width, 1, 0, 0, 0, mode)
                } else {
                    encode_masked(op, width, 0, 2, 0, 0, mode)
                });
                words.push(mov64_imm(0, 7));
                words.push(exit());
                let executable = verified(&words, config(true, aligned));
                let outcome = run_both(&executable, &input, kind, 1000);
                assert!(outcome.result.contains(expected), "{:?}", outcome);
                let pc = words.len() - 3;
                let before = instruction_count(&words[..pc]);
                if expected.starts_with("Ok(") {
                    assert_eq!(outcome.instruction_count, before + 2 + 2);
                } else {
                    assert_eq!(
                        (outcome.pc, outcome.instruction_count),
                        (pc as u64, before + 2)
                    );
                    assert_eq!(
                        outcome.vector_registers[1].0,
                        PRELOADED.to_le_bytes().repeat(8)[..]
                    );
                }
                if op == op::ST_M && expected.starts_with("Ok(") {
                    // Only the active lanes are written
                    let bytes = 1 << width;
                    let mut expected_memory = input.clone();
                    for i in (0..64 / bytes).filter(|i| (mask >> i) & 1 == 1) {
                        for k in i * bytes..(i + 1) * bytes {
                            expected_memory[offset as usize + k] = PRELOADED.to_le_bytes()[k % 8];
                        }
                    }
                    assert_eq!(outcome.memory, expected_memory);
                    assert_eq!(
                        outcome.handler_calls.len(),
                        (kind == Input::CopyOnWrite) as usize
                    );
                }
            }
        }
    }
}

/// ldn and stn translate the whole span once: it must lie in one region
#[test]
fn test_multi_register_memory_faults() {
    let input: Vec<u8> = input_pattern(200);
    for aligned in [false, true] {
        for (op, n, offset, kind, expected) in [
            (op::LDN, 3, 8, Input::Writable, "Ok("),
            (
                op::LDN,
                3,
                9,
                Input::Writable,
                "AccessViolation(Load, 17179869193, 192",
            ),
            (
                op::LDN,
                2,
                64,
                Input::Gapped,
                "AccessViolation(Load, 17179869248, 128",
            ),
            (
                op::LDN,
                2,
                -64i64 as u64,
                Input::Writable,
                "AccessViolation(Load, 17179869120, 128",
            ),
            (op::STN, 2, 72, Input::Writable, "Ok("),
            (
                op::STN,
                2,
                73,
                Input::Writable,
                "AccessViolation(Store, 17179869257, 128",
            ),
            (
                op::STN,
                3,
                0,
                Input::ReadOnly,
                "AccessViolation(Store, 17179869184, 192",
            ),
            (op::STN, 3, 4, Input::CopyOnWrite, "Ok("),
            (
                op::STN,
                2,
                0,
                Input::Shifted,
                "AccessViolation(Store, 17179869184, 128",
            ),
        ] {
            for off in [0i16, 1, -1, 0x400] {
                let mut words = preload(1..5);
                words.extend(lddw(
                    1,
                    ebpf::MM_INPUT_START
                        .wrapping_add(offset)
                        .wrapping_sub(off as i64 as u64),
                ));
                // stn stores v1..v1+n with distinct lanes
                words.push(vop(op::INSERT_Q, 3, 2, 0, 0));
                words.push(encode_with_offset(
                    op,
                    n,
                    if op == op::LDN { 1 } else { 0 },
                    if op == op::STN { 1 } else { 0 },
                    0,
                    off,
                ));
                words.push(mov64_imm(0, 7));
                words.push(exit());
                let executable = verified(&words, config(true, aligned));
                let outcome = run_both(&executable, &input, kind, 1000);
                assert!(
                    outcome.result.contains(expected),
                    "{} {} {:?}",
                    offset,
                    off,
                    outcome
                );
                let pc = words.len() - 3;
                let before = instruction_count(&words[..pc]);
                if expected.starts_with("Ok(") {
                    assert_eq!(outcome.instruction_count, before + n as u64 + 2);
                    if op == op::LDN {
                        for k in 0..n as usize {
                            let start = offset as usize + 64 * k;
                            assert_eq!(outcome.vector_registers[1 + k].0, input[start..start + 64]);
                        }
                    } else {
                        let start = offset as usize;
                        for k in 0..n as usize {
                            assert_eq!(
                                &outcome.memory[start + 64 * k..start + 64 * (k + 1)],
                                &outcome.vector_registers[1 + k].0[..]
                            );
                        }
                    }
                } else {
                    assert_eq!(
                        (outcome.pc, outcome.instruction_count),
                        (pc as u64, before + n as u64)
                    );
                    for k in 1..5 {
                        if k != 2 {
                            assert_eq!(
                                outcome.vector_registers[k].0,
                                PRELOADED.to_le_bytes().repeat(8)[..]
                            );
                        }
                    }
                    assert_eq!(outcome.memory, input);
                }
            }
        }
    }
}

/// Builds a gather or scatter program: v1 = indices, v2 = data, v3 = old out (or data), then the
/// op at `base` with the mask in r2 and `mov r0, 7; exit`
fn gather_scatter_program(
    op: u8,
    scale: u8,
    mode: u8,
    indices: &[u64; 8],
    base: u64,
    mask: u64,
) -> Vec<u64> {
    let mut words = preload(2..4);
    words.push(vop(op::INSERT_Q, 5, 2, 0, 0));
    for (i, index) in indices.iter().enumerate() {
        words.extend(lddw(1, *index));
        words.push(vop(op::INSERT_Q, i as u8, 1, 0, 0));
    }
    words.extend(lddw(1, base));
    words.extend(lddw(2, mask));
    words.push(if op == op::GATHER_QQ {
        encode_masked(op, scale, 3, 1, 0, 0, mode)
    } else {
        encode_masked(op, scale, 0, 1, 2, 0, mode)
    });
    words.push(mov64_imm(0, 7));
    words.push(exit());
    words
}

/// The per lane model: faults, regions, overlaps, empty masks, wrapping and the handler
#[test]
fn test_gather_scatter_model() {
    let input: Vec<u8> = input_pattern(512);
    let start = ebpf::MM_INPUT_START;
    let stack = ebpf::MM_STACK_START;
    let in_input = [0, 8, 16, 504, 3, 200, 8, 0];
    for aligned in [false, true] {
        for op in [op::GATHER_QQ, op::SCATTER_QQ] {
            let modes: &[u8] = if op == op::GATHER_QQ {
                &[0, 1, 2]
            } else {
                &[0, 1]
            };
            for &mode in modes {
                let run_case = |indices: [u64; 8], scale: u8, base: u64, mask: u64, kind: Input| {
                    let words = gather_scatter_program(op, scale, mode, &indices, base, mask);
                    let executable = verified(&words, config(true, aligned));
                    let pc = words.len() - 3;
                    (
                        run_both(&executable, &input, kind, 1000),
                        pc as u64,
                        instruction_count(&words[..pc]),
                    )
                };
                let active = |mask: u64, i: usize| mode == 0 || (mask >> i) & 1 == 1;
                // All lanes in the input, scale 1 (fast path if aligned)
                for mask in [u64::MAX, 0x55, 0x80, 0x1_0000_0000] {
                    let (outcome, _, _) = run_case(in_input, 0, start, mask, Input::Writable);
                    assert_eq!(outcome.result, "Ok(7)");
                    if op == op::GATHER_QQ {
                        for (i, index) in in_input.iter().enumerate() {
                            let expected = if active(mask, i) {
                                u64::from_le_bytes(
                                    input[*index as usize..*index as usize + 8]
                                        .try_into()
                                        .unwrap(),
                                )
                            } else if mode == 1 {
                                PRELOADED
                            } else {
                                0
                            };
                            assert_eq!(
                                outcome.vector_registers[3].lane::<8>(i),
                                expected,
                                "{mask:#x}"
                            );
                        }
                    }
                }
                // Lanes in different regions (input and stack): absolute qword indices, base 0
                let indices = in_input.map(|i| {
                    if i % 2 == 0 {
                        start + i
                    } else {
                        stack + 64 * i
                    }
                });
                let (outcome, _, _) = run_case(indices, 0, 0, 0xff, Input::Writable);
                assert_eq!(outcome.result, "Ok(7)");
                // Overlapping and duplicate lanes, the later lane wins
                let overlapping = [0, 4, 4, 0, 2, 9, 9, 3];
                let (outcome, _, _) = run_case(overlapping, 0, start + 100, 0xff, Input::Writable);
                assert_eq!(outcome.result, "Ok(7)");
                if op == op::SCATTER_QQ {
                    let mut expected = input.clone();
                    for (i, index) in overlapping.iter().enumerate() {
                        let at = 100 + *index as usize;
                        expected[at..at + 8]
                            .copy_from_slice(&outcome.vector_registers[2].0[8 * i..8 * i + 8]);
                    }
                    assert_eq!(outcome.memory, expected);
                }
                // A faulting lane: its error, nothing changes, CUs of the op are charged
                for (lane, bad, error) in [
                    (5, 505, "AccessViolation(Load, 17179869689, 8, \"input\")"),
                    (
                        0,
                        u64::MAX,
                        "AccessViolation(Load, 17179869183, 8, \"unallocated\")",
                    ),
                    (7, 1 << 40, "AccessViolation(Load, 1099511627776, 8"),
                ] {
                    let mut indices = in_input;
                    indices[lane] = bad;
                    let base = if bad == 1 << 40 { 0 } else { start };
                    if base == 0 {
                        indices = indices.map(|i| if i == bad { bad } else { start + i });
                    }
                    let error = if op == op::SCATTER_QQ {
                        error.replace("Load", "Store")
                    } else {
                        error.to_string()
                    };
                    for mask in [u64::MAX, 1 << lane, !(1 << lane)] {
                        let (outcome, pc, before) =
                            run_case(indices, 0, base, mask, Input::Writable);
                        if active(mask, lane) {
                            assert!(
                                outcome.result.contains(&error),
                                "{} {mask:#x}",
                                outcome.result
                            );
                            assert_eq!((outcome.pc, outcome.instruction_count), (pc, before + 8));
                            assert_eq!(outcome.memory, input);
                            assert_eq!(
                                outcome.vector_registers[3].0,
                                PRELOADED.to_le_bytes().repeat(8)[..]
                            );
                        } else {
                            assert_eq!(outcome.result, "Ok(7)");
                        }
                    }
                }
                // Empty mask: no access at all
                if mode != 0 {
                    let (outcome, _, _) =
                        run_case([u64::MAX; 8], 3, 0, 0xffff_ff00, Input::ReadOnly);
                    assert_eq!(outcome.result, "Ok(7)");
                    assert_eq!(outcome.memory, input);
                    let expected = if op == op::GATHER_QQ && mode == 2 {
                        0
                    } else {
                        PRELOADED
                    };
                    assert_eq!(
                        outcome.vector_registers[3].0,
                        expected.to_le_bytes().repeat(8)[..]
                    );
                }
                // index << scale wraps around: 2^61 << 3 = 0, -1 << 3 = -8
                let wrapping = [1 << 61, u64::MAX, 2, 1 << 61, 5, 6, 7, 1];
                let (outcome, _, _) = run_case(wrapping, 3, start + 8, 0xff, Input::Writable);
                assert_eq!(outcome.result, "Ok(7)");
                if op == op::GATHER_QQ {
                    assert_eq!(outcome.vector_registers[3].0[..8], input[8..16]);
                    assert_eq!(outcome.vector_registers[3].0[8..16], input[0..8]);
                }
                // Read-only and copy-on-write: the handler sees the lanes in ascending order
                for kind in [Input::ReadOnly, Input::CopyOnWrite] {
                    let (outcome, _, _) = run_case(in_input, 0, start, 0xff, kind);
                    if op == op::GATHER_QQ {
                        assert_eq!(outcome.result, "Ok(7)");
                        assert!(outcome.handler_calls.is_empty());
                    } else if kind == Input::ReadOnly {
                        assert!(outcome
                            .result
                            .contains("AccessViolation(Store, 17179869184, 8"));
                    } else {
                        assert_eq!(outcome.result, "Ok(7)");
                        assert_eq!(outcome.handler_calls, ["Store 0x400000000 8"]);
                    }
                }
            }
        }
    }
}

/// Gathers and scatters with dword indices or dword elements: `index` holds the signed indices
/// of the lanes, v2 distinct data, v3 the preloaded old value
fn narrow_gather_scatter_program(
    op: u8,
    scale: u8,
    mode: u8,
    index: &[i64],
    base: u64,
    mask: u64,
) -> Vec<u64> {
    let dword_index = matches!(
        op,
        op::GATHER_DD | op::GATHER_DQ | op::SCATTER_DD | op::SCATTER_DQ
    );
    let qwords: Vec<u64> = if dword_index {
        index
            .chunks(2)
            .map(|pair| pair[0] as u32 as u64 | (pair.get(1).copied().unwrap_or(0) as u64) << 32)
            .collect()
    } else {
        index.iter().map(|i| *i as u64).collect()
    };
    let mut words = preload(3..4);
    for (i, qword) in qwords.iter().enumerate() {
        words.extend(lddw(1, *qword));
        words.push(vop(op::INSERT_Q, i as u8, 1, 0, 0));
        words.extend(lddw(1, 0xd000_0001_d000_0000 + 0x2_0000_0002 * i as u64));
        words.push(vop(op::INSERT_Q, i as u8, 2, 0, 0));
    }
    words.extend(lddw(1, base));
    words.extend(lddw(2, mask));
    words.push(if op_info(op).unwrap().kind == Kind::Ga {
        encode_masked(op, scale, 3, 1, 0, 0, mode)
    } else {
        encode_masked(op, scale, 0, 1, 2, 0, mode)
    });
    words.push(mov64_imm(0, 7));
    words.push(exit());
    words
}

/// The per lane model for dword indices and dword elements: sign extension, faults, regions,
/// overlaps, empty masks and the handler. gather.qd clears the upper half in every mode.
#[test]
fn test_narrow_gather_scatter_model() {
    let input: Vec<u8> = input_pattern(512);
    let start = ebpf::MM_INPUT_START;
    let ops = [
        op::GATHER_DD,
        op::GATHER_DQ,
        op::GATHER_QD,
        op::SCATTER_DD,
        op::SCATTER_DQ,
        op::SCATTER_QD,
    ];
    let preloaded = PRELOADED.to_le_bytes().repeat(8);
    for aligned in [false, true] {
        for op in ops {
            let info = op_info(op).unwrap();
            let gather = info.kind == Kind::Ga;
            let (lanes, element) = match op {
                op::GATHER_DD | op::SCATTER_DD => (16, 4),
                op::GATHER_DQ | op::SCATTER_DQ => (8, 8),
                _ => (8, 4),
            };
            let cu = info.cu(0);
            for mode in modes(info) {
                let run_case = |index: &[i64], scale: u8, base: u64, mask: u64, kind: Input| {
                    let words = narrow_gather_scatter_program(op, scale, mode, index, base, mask);
                    let executable = verified(&words, config(true, aligned));
                    let pc = words.len() - 3;
                    (
                        run_both(&executable, &input, kind, 1000),
                        pc as u64,
                        instruction_count(&words[..pc]),
                    )
                };
                let active = |mask: u64, i: usize| mode == 0 || (mask >> i) & 1 == 1;
                let small = [0, 8, 16, 40, 3, -8, 8, 0, 1, -1, 5, 9, -4, 12, 12, 6];
                let index = &small[..lanes];
                for scale in 0..4 {
                    for mask in [u64::MAX, 0x5555, 0x80, 0x1_0000_0000] {
                        let base = start + 64;
                        let (outcome, _, _) = run_case(index, scale, base, mask, Input::Writable);
                        assert_eq!(outcome.result, "Ok(7)");
                        let address = |i: usize| (64 + (index[i] << scale)) as usize;
                        if gather {
                            let mut expected = if mode == 1 {
                                preloaded.clone()
                            } else {
                                vec![0; 64]
                            };
                            for i in (0..lanes).filter(|i| active(mask, *i)) {
                                expected[i * element..(i + 1) * element]
                                    .copy_from_slice(&input[address(i)..address(i) + element]);
                            }
                            expected[lanes * element..].fill(0);
                            assert_eq!(outcome.vector_registers[3].0[..], expected[..]);
                        } else {
                            let mut expected = input.clone();
                            let data = outcome.vector_registers[2].0;
                            for i in (0..lanes).filter(|i| active(mask, *i)) {
                                expected[address(i)..address(i) + element]
                                    .copy_from_slice(&data[i * element..(i + 1) * element]);
                            }
                            assert_eq!(outcome.memory, expected, "{mask:#x}");
                        }
                    }
                }
                // Lanes in the input and on the stack: -2^30 << 3 = -(start - stack start)
                let mut regions = index.to_vec();
                regions[1] = if matches!(op, op::GATHER_QD | op::SCATTER_QD) {
                    (ebpf::MM_STACK_START as i64 - start as i64) >> 3
                } else {
                    -(1 << 30)
                };
                let (outcome, _, _) = run_case(&regions, 3, start + 64, 0xffff, Input::Writable);
                assert_eq!(outcome.result, "Ok(7)");
                // A faulting lane: its error, nothing changes, all CUs charged
                let access = if gather { "Load" } else { "Store" };
                for (lane, bad, error) in [
                    (
                        lanes - 1,
                        446,
                        format!("AccessViolation({access}, 17179869694, {element}, \"input\")"),
                    ),
                    (
                        2,
                        -65,
                        format!(
                            "AccessViolation({access}, 17179869183, {element}, \"unallocated\")"
                        ),
                    ),
                ] {
                    let mut faulting = index.to_vec();
                    faulting[lane] = bad;
                    for mask in [u64::MAX, 1 << lane, !(1 << lane)] {
                        let (outcome, pc, before) =
                            run_case(&faulting, 0, start + 64, mask, Input::Writable);
                        if active(mask, lane) {
                            assert!(outcome.result.contains(&error), "{}", outcome.result);
                            assert_eq!((outcome.pc, outcome.instruction_count), (pc, before + cu));
                            assert_eq!(outcome.memory, input);
                            assert_eq!(outcome.vector_registers[3].0[..], preloaded[..]);
                        } else {
                            assert_eq!(outcome.result, "Ok(7)");
                        }
                    }
                }
                // Empty mask: no access at all
                if mode != 0 {
                    let wild = vec![i32::MIN as i64; lanes];
                    let (outcome, _, _) = run_case(&wild, 3, 0, u64::MAX << lanes, Input::ReadOnly);
                    assert_eq!(outcome.result, "Ok(7)");
                    assert_eq!(outcome.memory, input);
                    let mut expected = preloaded.clone();
                    if gather {
                        if mode == 2 {
                            expected.fill(0);
                        }
                        expected[lanes * element..].fill(0);
                    }
                    assert_eq!(outcome.vector_registers[3].0[..], expected[..]);
                }
                // Read-only and copy-on-write: one handler call at the first active lane
                for kind in [Input::ReadOnly, Input::CopyOnWrite] {
                    let (outcome, _, _) = run_case(index, 0, start + 64, u64::MAX, kind);
                    if gather {
                        assert_eq!(outcome.result, "Ok(7)");
                        assert!(outcome.handler_calls.is_empty());
                    } else if kind == Input::ReadOnly {
                        assert!(outcome
                            .result
                            .contains(&format!("AccessViolation(Store, 17179869248, {element}")));
                    } else {
                        assert_eq!(outcome.result, "Ok(7)");
                        assert_eq!(
                            outcome.handler_calls,
                            [format!("Store 0x400000040 {element}")]
                        );
                    }
                }
            }
        }
    }
}

/// Ops with more than one CU fail at their pc if the budget does not cover all of them, in both
/// engines with the same state (the vector registers too: every op checks)
#[test]
fn test_multi_cu_instruction_meter() {
    let input = input_pattern(1024);
    let mut words = vec![mov64_reg(6, 1), vop(op::ZERO, 0, 29, 0, 0)];
    words.extend(lddw(2, 0x5a5a_5a5a_5a5a_5a5b));
    let setup = instruction_count(&words);
    let ops = [
        (encode_masked(op::LD_M, 2, 1, 0, 0, 0, 2), 2),
        (encode_with_offset(op::LDN, 3, 4, 0, 0, 64), 3),
        (vop(op::CMP, 0x36, 0, 1, 4), 2),
        (encode_masked(op::ST_M, 0, 0, 5, 0, 256, 1), 2),
        (vop(op::SHA256RNDS2, 0, 20, 4, 5), 2),
        (vop(op::GATHER_QQ, 3, 9, 29, 0), 8),
        (encode_masked(op::SCATTER_QQ, 0, 0, 29, 4, 512, 1), 8),
        (encode_with_offset(op::STN, 2, 0, 20, 0, 600), 2),
        (vop(op::BITSHUF, 0, 0, 4, 5), 2),
        (vop(op::MUL52LO, 0, 6, 4, 5), 2),
        (encode_masked(op::MUL52HI, 0, 4, 4, 5, 0, 1), 2),
        (vop(op::CONFLICT_D, 0, 6, 4, 0), 4),
        (encode_masked(op::CONFLICT_Q, 0, 7, 5, 0, 0, 1), 3),
        (vop(op::TESTN, 1, 0, 4, 5), 2),
        (vop(op::MOVEMASK_Q, 0, 0, 7, 0), 2),
        (encode_masked(op::GATHER_DD, 2, 10, 29, 0, 8, 2), 16),
        (vop(op::GATHER_DQ, 3, 11, 29, 0), 8),
        (encode_masked(op::GATHER_QD, 1, 12, 29, 0, 0, 1), 8),
        (vop(op::SCATTER_DD, 0, 0, 29, 4), 16),
        (encode_masked(op::SCATTER_DQ, 3, 0, 29, 5, 128, 1), 8),
        (vop(op::SCATTER_QD, 2, 0, 29, 6), 8),
    ];
    for (word, _) in ops {
        words.push(word);
    }
    words.push(exit());
    let total = setup + ops.iter().map(|(_, cu)| cu).sum::<u64>() + 1;
    for aligned in [false, true] {
        for checkpoint_distance in [1000, 1, 3] {
            let executable = verified(&words, run_config(aligned, checkpoint_distance));
            if jit_available() {
                executable.jit_compile().unwrap();
            }
            for budget in 0..=total + 1 {
                let interpreted = run(&executable, &input, Input::Writable, budget, false);
                if budget < total {
                    assert_eq!(
                        interpreted.result, "Err(ExceededMaxInstructions)",
                        "{budget}"
                    );
                    assert_eq!(interpreted.instruction_count, budget);
                } else {
                    assert!(interpreted.result.starts_with("Ok("), "{}", budget);
                    assert_eq!(interpreted.instruction_count, total);
                }
                if jit_available() {
                    let jitted = run(&executable, &input, Input::Writable, budget, true);
                    assert_eq!(
                        interpreted, jitted,
                        "{aligned} {checkpoint_distance} {budget}"
                    );
                }
            }
        }
    }
}

/// Every register op with more than one CU (for some imm8) in every mode: the budget boundary
/// lands on the same pc with the same state in both engines
#[test]
fn test_multi_cu_register_ops_instruction_meter() {
    let input = input_pattern(64);
    let mut checked = 0;
    for info in implemented().filter(|info| !info.accesses_memory()) {
        let imm8s: Vec<u8> = (0..=255)
            .filter(|imm8| info.imm8_is_valid(*imm8) && info.cu(*imm8) > 1)
            .collect();
        for imm8 in imm8s.first().into_iter().chain(imm8s.last()).copied() {
            for mode in modes(info) {
                let kind = info.kind;
                let mut words = lddw(1, 0x0123_4567_89ab_cdef).to_vec();
                words.push(vop(op::SPLAT_Q, 0, 7, 0, 0));
                words.extend(lddw(2, 0x5a5a_5a5a_5a5a_5a5b));
                words.push(encode_masked(
                    info.op,
                    imm8,
                    7 * kind.uses_out() as u8,
                    7 * kind.uses_in0() as u8,
                    7 * kind.uses_in1() as u8,
                    0,
                    mode,
                ));
                words.push(exit());
                let total = instruction_count(&words) + info.cu(imm8) - 1;
                let executable = verified(&words, run_config(true, 1000));
                for budget in total - 2..=total + 1 {
                    let outcome = run_both(&executable, &input, Input::Writable, budget);
                    if budget < total {
                        assert_eq!(outcome.result, "Err(ExceededMaxInstructions)", "{budget}");
                        assert_eq!(outcome.instruction_count, budget, "{}", info.name);
                    } else {
                        assert!(outcome.result.starts_with("Ok("), "{}", info.name);
                        assert_eq!(outcome.instruction_count, total, "{}", info.name);
                    }
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

/// The JIT sizes the text section with a bound per vector op: long programs of the longest
/// lowerings (checked against the bound after every op) compile
#[test]
fn test_vector_code_length_budget() {
    if !jit_available() {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(0xb0d9e7);
    let infos: Vec<&OpInfo> = implemented().collect();
    for aligned in [false, true] {
        let mut words = vec![];
        for _ in 0..3000 {
            let info = infos[rng.gen_range(0..infos.len())];
            let modes = modes(info);
            let imm8 = random_imm8(&mut rng, info.op);
            let reg = |rng: &mut SmallRng| [0, 15, 16, 29][rng.gen_range(0..4)];
            let (mut out, mut in0, in1) = (reg(&mut rng), reg(&mut rng), reg(&mut rng));
            if info.imm8_is_count() {
                out = out.min(30 - imm8);
                in0 = in0.min(30 - imm8);
            }
            let kind = info.kind;
            words.push(encode_masked(
                info.op,
                imm8,
                out * kind.uses_out() as u8,
                in0 * kind.uses_in0() as u8,
                in1 * kind.uses_in1() as u8,
                if info.accesses_memory() { rng.gen() } else { 0 },
                modes[rng.gen_range(0..modes.len())],
            ));
        }
        words.push(exit());
        verified(&words, config(true, aligned))
            .jit_compile()
            .unwrap();
    }
}

// Crypto ops: known answers of AES-128 (FIPS-197), SHA-1 and SHA-256 (FIPS 180-4) kernels built
// from the ops, in the reference semantics and op by op in both engines

type Exec<'a> =
    &'a mut dyn FnMut(u8, u8, &VectorRegister, &VectorRegister, &VectorRegister) -> VectorRegister;

fn reference_op(
    op: u8,
    imm8: u8,
    c: &VectorRegister,
    a: &VectorRegister,
    b: &VectorRegister,
) -> VectorRegister {
    solana_sbpf::vector::execute_register_op(op, imm8, c, a, b).unwrap()
}

/// Runs `op` with out = c, in0 = a, in1 = b in a program of its own in both engines, on varying
/// registers
fn engine_op(
    calls: &mut usize,
) -> impl FnMut(u8, u8, &VectorRegister, &VectorRegister, &VectorRegister) -> VectorRegister + '_ {
    move |op, imm8, c, a, b| {
        let (out, in0, in1) = [
            (20, 17, 25),
            (3, 4, 0),
            (0, 15, 29),
            (29, 0, 1),
            (8, 24, 12),
        ][*calls % 5];
        *calls += 1;
        let kind = op_info(op).unwrap().kind;
        let words = [
            ld(out, 0),
            ld(in0, 64),
            ld(in1, 128),
            encode(op, imm8, out, in0, in1 * kind.uses_in1() as u8),
            st(out, 192),
            exit(),
        ];
        let input = [c.0, a.0, b.0, [0; 64]].concat();
        let outcome = run_both(
            &verified(&words, config(true, true)),
            &input,
            Input::Writable,
            100,
        );
        assert_eq!(outcome.result, "Ok(0)");
        let result = VectorRegister(outcome.memory[192..].try_into().unwrap());
        assert_eq!(result, reference_op(op, imm8, c, a, b));
        result
    }
}

fn dwords(d: [u32; 4]) -> VectorRegister {
    let mut v = VectorRegister::ZERO;
    for (k, x) in d.iter().enumerate() {
        v.set_lane::<4>(k, *x as u64);
    }
    v
}

fn low_dwords(v: &VectorRegister) -> [u32; 4] {
    std::array::from_fn(|k| v.lane::<4>(k) as u32)
}

fn padded_abc() -> [u32; 16] {
    let mut block = [0u8; 64];
    block[..4].copy_from_slice(b"abc\x80");
    block[63] = 24;
    std::array::from_fn(|t| u32::from_be_bytes(block[4 * t..4 * t + 4].try_into().unwrap()))
}

fn sha256_abc(exec: Exec) -> [u32; 8] {
    let k: [u32; 64] = {
        let primes = (2u128..).filter(|n| (2..*n).all(|d| n % d != 0));
        let cube_root = |n: u128| {
            let (mut low, mut high) = (0u128, 1 << 36);
            while high - low > 1 {
                let middle = (low + high) / 2;
                if middle * middle * middle <= n {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            low
        };
        let k: Vec<u32> = primes.take(64).map(|p| cube_root(p << 96) as u32).collect();
        k.try_into().unwrap()
    };
    let h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let w = padded_abc();
    let mut schedule: Vec<VectorRegister> = (0..4)
        .map(|i| dwords(std::array::from_fn(|j| w[4 * i + j])))
        .collect();
    for i in 4..16 {
        let m = low_dwords(&exec(
            op::SHA256MSG1,
            0,
            &schedule[i - 4],
            &schedule[i - 3],
            &VectorRegister::ZERO,
        ));
        let (w2, w1) = (low_dwords(&schedule[i - 2]), low_dwords(&schedule[i - 1]));
        let w7 = [w2[1], w2[2], w2[3], w1[0]];
        let m = dwords(std::array::from_fn(|j| m[j].wrapping_add(w7[j])));
        let next = exec(
            op::SHA256MSG2,
            0,
            &m,
            &schedule[i - 1],
            &VectorRegister::ZERO,
        );
        schedule.push(next);
    }
    let mut abef = dwords([h[5], h[4], h[1], h[0]]);
    let mut cdgh = dwords([h[7], h[6], h[3], h[2]]);
    for (i, block) in schedule.iter().enumerate() {
        let x = low_dwords(block);
        let wk: [u32; 4] = std::array::from_fn(|j| x[j].wrapping_add(k[4 * i + j]));
        let new_abef = exec(op::SHA256RNDS2, 0, &cdgh, &abef, &dwords(wk));
        cdgh = abef;
        abef = exec(
            op::SHA256RNDS2,
            0,
            &cdgh,
            &new_abef,
            &dwords([wk[2], wk[3], 0, 0]),
        );
        cdgh = new_abef;
    }
    let ([f, e, b, a], [hh, g, d, c]) = (low_dwords(&abef), low_dwords(&cdgh));
    let state = [a, b, c, d, e, f, g, hh];
    std::array::from_fn(|j| h[j].wrapping_add(state[j]))
}

fn sha1_abc(exec: Exec) -> [u32; 5] {
    let h = [
        0x67452301u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];
    let w = padded_abc();
    let mut schedule: Vec<VectorRegister> = (0..4)
        .map(|i| dwords(std::array::from_fn(|j| w[4 * i + 3 - j])))
        .collect();
    let zero = VectorRegister::ZERO;
    for i in 4..20 {
        let m = exec(op::SHA1MSG1, 0, &schedule[i - 4], &schedule[i - 3], &zero);
        let m = reference_op(op::XOR, 0, &zero, &m, &schedule[i - 2]);
        let next = exec(op::SHA1MSG2, 0, &m, &schedule[i - 1], &zero);
        schedule.push(next);
    }
    let mut abcd = dwords([h[3], h[2], h[1], h[0]]);
    let first = low_dwords(&schedule[0]);
    let mut e = dwords([first[0], first[1], first[2], first[3].wrapping_add(h[4])]);
    let mut previous = abcd;
    for (g, block) in schedule.iter().enumerate() {
        if g > 0 {
            e = exec(op::SHA1NEXTE, 0, &previous, block, &zero);
        }
        previous = abcd;
        abcd = exec(op::SHA1RNDS4, g as u8 / 5, &abcd, &e, &zero);
    }
    let e = exec(op::SHA1NEXTE, 0, &previous, &dwords([0, 0, 0, h[4]]), &zero);
    let ([d, c, b, a], e) = (low_dwords(&abcd), low_dwords(&e)[3]);
    [
        h[0].wrapping_add(a),
        h[1].wrapping_add(b),
        h[2].wrapping_add(c),
        h[3].wrapping_add(d),
        e,
    ]
}

/// AES-128 of FIPS-197 C.1 on all four 128-bit lanes: (ciphertext, decrypted ciphertext)
fn aes128(exec: Exec) -> (VectorRegister, VectorRegister) {
    let lanes = |bytes: &[u8]| VectorRegister(bytes.repeat(4).try_into().unwrap());
    let zero = VectorRegister::ZERO;
    let key: Vec<u8> = (0..16).collect();
    let plaintext: Vec<u8> = (0..16).map(|i| i * 0x11).collect();
    // The key schedule, SubWord through aesenclast of a register of equal columns
    let mut w: Vec<u32> = (0..4)
        .map(|i| u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()))
        .collect();
    let mut rcon = 1u8;
    for i in 4..44 {
        let mut t = w[i - 1];
        if i % 4 == 0 {
            let column = lanes(&t.rotate_right(8).to_le_bytes().repeat(4));
            t = exec(op::AESENCLAST, 0, &zero, &column, &zero).lane::<4>(0) as u32 ^ rcon as u32;
            rcon = (rcon << 1) ^ if rcon & 0x80 != 0 { 0x1b } else { 0 };
        }
        w.push(w[i - 4] ^ t);
    }
    let round_key = |r: usize| {
        lanes(
            &w[4 * r..4 * r + 4]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    };
    let mut state = reference_op(op::XOR, 0, &zero, &lanes(&plaintext), &round_key(0));
    for r in 1..10 {
        state = exec(op::AESENC, 0, &zero, &state, &round_key(r));
    }
    let ciphertext = exec(op::AESENCLAST, 0, &zero, &state, &round_key(10));
    // The equivalent inverse cipher, InvMixColumns of a key k is aesdec(aesenclast(k, 0), 0)
    let mut state = reference_op(op::XOR, 0, &zero, &ciphertext, &round_key(10));
    for r in (1..10).rev() {
        let shifted = exec(op::AESENCLAST, 0, &zero, &round_key(r), &zero);
        let key = exec(op::AESDEC, 0, &zero, &shifted, &zero);
        state = exec(op::AESDEC, 0, &zero, &state, &key);
    }
    let decrypted = exec(op::AESDECLAST, 0, &zero, &state, &round_key(0));
    (ciphertext, decrypted)
}

#[test]
fn test_crypto_known_answers() {
    let ciphertext = [
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
        0x5a,
    ];
    let plaintext: Vec<u8> = (0..16).map(|i| i * 0x11).collect();
    let sha1 = [0xa9993e36, 0x4706816a, 0xba3e2571, 0x7850c26c, 0x9cd0d89d];
    let sha256 = [
        0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
        0xf20015ad,
    ];
    let mut calls = 0;
    let mut engines = engine_op(&mut calls);
    for exec in [&mut reference_op as Exec, &mut engines as Exec] {
        let (encrypted, decrypted) = aes128(exec);
        assert_eq!(encrypted.0, *ciphertext.repeat(4));
        assert_eq!(decrypted.0, *plaintext.repeat(4));
        assert_eq!(sha1_abc(exec), sha1);
        assert_eq!(sha256_abc(exec), sha256);
    }
    drop(engines);
    assert_eq!(calls, 176);
}

/// GF(2^8): x · x^-1 = 1 for every byte, the identity matrix, and the AES S-box as an affine map
/// of the inverse (FIPS-197 5.1.1) agrees with aesenclast
#[test]
fn test_gf2p8_identities() {
    let zero = VectorRegister::ZERO;
    let identity = VectorRegister(
        0x0102_0408_1020_4080u64
            .to_le_bytes()
            .repeat(8)
            .try_into()
            .unwrap(),
    );
    let sbox_matrix = VectorRegister(
        0xf1e3_c78f_1f3e_7cf8u64
            .to_le_bytes()
            .repeat(8)
            .try_into()
            .unwrap(),
    );
    for base in (0..256).step_by(64) {
        let x = VectorRegister(std::array::from_fn(|k| (base + k) as u8));
        for imm8 in [0, 1, 0x80, 0xff] {
            let affine = reference_op(op::GF2P8AFFINE, imm8, &zero, &x, &identity);
            assert!(affine.0.iter().zip(x.0).all(|(y, x)| *y == x ^ imm8));
        }
        let inverse = reference_op(op::GF2P8AFFINEINV, 0, &zero, &x, &identity);
        let product = reference_op(op::GF2P8MUL, 0, &zero, &x, &inverse);
        for (k, p) in product.0.iter().enumerate() {
            assert_eq!(*p, (base + k != 0) as u8, "{}", base + k);
        }
        let sbox = reference_op(op::GF2P8AFFINEINV, 0x63, &zero, &x, &sbox_matrix);
        for (k, s) in sbox.0.iter().enumerate() {
            let column = VectorRegister([(base + k) as u8; 64]);
            let aes = reference_op(op::AESENCLAST, 0, &zero, &column, &zero);
            assert_eq!(aes.0, [*s; 64]);
        }
    }
    // FIPS-197 5.1.1: S-box(0x53) = 0xed
    let x = VectorRegister([0x53; 64]);
    assert_eq!(
        reference_op(op::AESENCLAST, 0, &zero, &x, &zero).0,
        [0xed; 64]
    );
}

/// Comparison of the reference semantics (interpreter) against native AVX-512 intrinsics
#[cfg(target_arch = "x86_64")]
mod native {
    use super::*;
    use solana_sbpf::vector::{
        apply_mask, execute_mask_op, execute_register_op, execute_scalar_input_op,
        execute_scalar_output_op,
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
            && is_x86_feature_detected!("vpclmulqdq")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vbmi2")
            && is_x86_feature_detected!("sha")
            && is_x86_feature_detected!("gfni")
            && is_x86_feature_detected!("vaes")
            && is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("lzcnt");
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

    /// [with_imm] for every imm8 below the given bound
    macro_rules! with_imm_below {
        ($imm:expr, 2, $body:expr) => { with_imm!($imm, [0 1], $body) };
        ($imm:expr, 4, $body:expr) => { with_imm!($imm, [0 1 2 3], $body) };
        ($imm:expr, 8, $body:expr) => { with_imm!($imm, [0 1 2 3 4 5 6 7], $body) };
        ($imm:expr, 16, $body:expr) => {
            with_imm!($imm, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], $body)
        };
        ($imm:expr, 32, $body:expr) => {
            with_imm!($imm, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], $body)
        };
        ($imm:expr, 64, $body:expr) => {
            with_imm!($imm, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], $body)
        };
        ($imm:expr, 256, $body:expr) => {
            with_imm!($imm, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], $body)
        };
    }

    enum Native {
        Vector(__m512i),
        Scalar(u64),
    }

    /// The intrinsic column of the spec, for every op without a memory operand, unmasked (except
    /// for the ops whose purpose is the mask in r2)
    #[target_feature(
        enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512ifma,avx512bitalg,avx512vpopcntdq,vpclmulqdq,avx512cd,avx512vbmi,avx512vbmi2,sha,popcnt,bmi1,lzcnt"
    )]
    unsafe fn native(
        op: u8,
        imm8: u8,
        c: __m512i,
        a: __m512i,
        b: __m512i,
        r1: u64,
        r2: u64,
    ) -> Native {
        use Native::{Scalar, Vector};
        let low = _mm512_castsi512_si128;
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
            op::MPOPCNT => Scalar(_popcnt64(r2 as i64) as u64),
            op::MTZCNT => Scalar(_tzcnt_u64(r2)),
            op::MLZCNT => Scalar(_lzcnt_u64(r2)),
            op::BLENDM => Vector(match imm8 {
                0 => _mm512_mask_blend_epi8(r2, a, b),
                1 => _mm512_mask_blend_epi16(r2 as u32, a, b),
                2 => _mm512_mask_blend_epi32(r2 as u16, a, b),
                _ => _mm512_mask_blend_epi64(r2 as u8, a, b),
            }),
            op::BCAST_Q => Vector(_mm512_permutexvar_epi64(_mm512_set1_epi64(imm8 as i64), a)),
            op::BCAST_D => Vector(_mm512_permutexvar_epi32(_mm512_set1_epi32(imm8 as i32), a)),
            op::BCAST_W => Vector(_mm512_permutexvar_epi16(_mm512_set1_epi16(imm8 as i16), a)),
            op::BCAST_B => Vector(_mm512_permutexvar_epi8(_mm512_set1_epi8(imm8 as i8), a)),
            op::PERMQI => Vector(with_imm_below!(imm8, 256, _mm512_permutex_epi64::<I>(a))),
            op::EXTRACT256 => Vector(_mm512_zextsi256_si512(with_imm_below!(
                imm8,
                2,
                _mm512_extracti64x4_epi64::<I>(a)
            ))),
            op::PERM2I => Vector(match imm8 {
                0 => _mm512_permutex2var_epi8(a, c, b),
                1 => _mm512_permutex2var_epi16(a, c, b),
                2 => _mm512_permutex2var_epi32(a, c, b),
                _ => _mm512_permutex2var_epi64(a, c, b),
            }),
            op::ALIGN_Q => Vector(with_imm_below!(imm8, 8, _mm512_alignr_epi64::<I>(a, b))),
            op::ALIGN_D => Vector(with_imm_below!(imm8, 16, _mm512_alignr_epi32::<I>(a, b))),
            op::EXTRACT128 => Vector(_mm512_zextsi128_si512(with_imm_below!(
                imm8,
                4,
                _mm512_extracti32x4_epi32::<I>(a)
            ))),
            op::INSERT256 => Vector(with_imm_below!(
                imm8,
                2,
                _mm512_inserti64x4::<I>(a, _mm512_castsi512_si256(b))
            )),
            op::INSERT128 => Vector(with_imm_below!(imm8, 4, _mm512_inserti32x4::<I>(a, low(b)))),
            op::PERM2T => Vector(match imm8 {
                0 => _mm512_permutex2var_epi8(c, a, b),
                2 => _mm512_permutex2var_epi32(c, a, b),
                _ => _mm512_permutex2var_epi64(c, a, b),
            }),
            op::PERMB => Vector(_mm512_permutexvar_epi8(a, b)),
            op::PERMW => Vector(_mm512_permutexvar_epi16(a, b)),
            op::SHUFD => Vector(with_imm_below!(imm8, 256, _mm512_shuffle_epi32::<I>(a))),
            op::SHUFHW => Vector(with_imm_below!(imm8, 256, _mm512_shufflehi_epi16::<I>(a))),
            op::SHUFLW => Vector(with_imm_below!(imm8, 256, _mm512_shufflelo_epi16::<I>(a))),
            op::SHUF128 => Vector(with_imm_below!(imm8, 256, _mm512_shuffle_i64x2::<I>(a, b))),
            op::SHUFPS => Vector(_mm512_castps_si512(with_imm_below!(
                imm8,
                256,
                _mm512_shuffle_ps::<I>(_mm512_castsi512_ps(a), _mm512_castsi512_ps(b))
            ))),
            op::ALIGNR => Vector(with_imm_below!(imm8, 32, _mm512_alignr_epi8::<I>(a, b))),
            op::BSHLI => Vector(with_imm_below!(imm8, 16, _mm512_bslli_epi128::<I>(a))),
            op::BSHRI => Vector(with_imm_below!(imm8, 16, _mm512_bsrli_epi128::<I>(a))),
            op::UNPACKLO => Vector(match imm8 {
                0 => _mm512_unpacklo_epi8(a, b),
                1 => _mm512_unpacklo_epi16(a, b),
                2 => _mm512_unpacklo_epi32(a, b),
                _ => _mm512_unpacklo_epi64(a, b),
            }),
            op::UNPACKHI => Vector(match imm8 {
                0 => _mm512_unpackhi_epi8(a, b),
                1 => _mm512_unpackhi_epi16(a, b),
                2 => _mm512_unpackhi_epi32(a, b),
                _ => _mm512_unpackhi_epi64(a, b),
            }),
            // The masked forms are the op, see native_mask_intrinsic
            op::COMPRESS | op::EXPAND => Vector(_mm512_setzero_si512()),
            op::PACKUS_D => Vector(_mm512_packus_epi32(a, b)),
            op::NARROW_D_W => Vector(_mm512_zextsi256_si512(_mm512_cvtepi32_epi16(a))),
            op::NARROWUS_D_W => Vector(_mm512_zextsi256_si512(_mm512_cvtusepi32_epi16(a))),
            op::NARROW_Q_B => Vector(_mm512_zextsi128_si512(_mm512_cvtepi64_epi8(a))),
            op::NARROWS_Q_B => Vector(_mm512_zextsi128_si512(_mm512_cvtsepi64_epi8(a))),
            op::NARROWUS_Q_B => Vector(_mm512_zextsi128_si512(_mm512_cvtusepi64_epi8(a))),
            op::NARROW_Q_W => Vector(_mm512_zextsi128_si512(_mm512_cvtepi64_epi16(a))),
            op::NARROWS_Q_W => Vector(_mm512_zextsi128_si512(_mm512_cvtsepi64_epi16(a))),
            op::NARROWUS_Q_W => Vector(_mm512_zextsi128_si512(_mm512_cvtusepi64_epi16(a))),
            op::NARROW_Q_D => Vector(_mm512_zextsi256_si512(_mm512_cvtepi64_epi32(a))),
            op::NARROWS_Q_D => Vector(_mm512_zextsi256_si512(_mm512_cvtsepi64_epi32(a))),
            op::NARROWUS_Q_D => Vector(_mm512_zextsi256_si512(_mm512_cvtusepi64_epi32(a))),
            op::NARROW_W_B => Vector(_mm512_zextsi256_si512(_mm512_cvtepi16_epi8(a))),
            op::NARROWS_W_B => Vector(_mm512_zextsi256_si512(_mm512_cvtsepi16_epi8(a))),
            op::NARROWUS_W_B => Vector(_mm512_zextsi256_si512(_mm512_cvtusepi16_epi8(a))),
            op::ZX_B_W => Vector(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(a))),
            op::SX_B_W => Vector(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a))),
            op::ZX_B_Q => Vector(_mm512_cvtepu8_epi64(low(a))),
            op::SX_B_Q => Vector(_mm512_cvtepi8_epi64(low(a))),
            op::ZX_W_Q => Vector(_mm512_cvtepu16_epi64(low(a))),
            op::SX_W_Q => Vector(_mm512_cvtepi16_epi64(low(a))),
            op::EXTRACT_B => Scalar(with_imm_below!(
                imm8,
                64,
                _mm_extract_epi8::<{ I & 15 }>(_mm512_extracti32x4_epi32::<{ I >> 4 }>(a))
            ) as u32 as u64),
            op::EXTRACT_W => Scalar(with_imm_below!(
                imm8,
                32,
                _mm_extract_epi16::<{ I & 7 }>(_mm512_extracti32x4_epi32::<{ I >> 3 }>(a))
            ) as u32 as u64),
            op::INSERT_B => Vector(with_imm_below!(
                imm8,
                64,
                _mm512_inserti32x4::<{ I >> 4 }>(
                    c,
                    _mm_insert_epi8::<{ I & 15 }>(
                        _mm512_extracti32x4_epi32::<{ I >> 4 }>(c),
                        r1 as i32
                    )
                )
            )),
            op::INSERT_W => Vector(with_imm_below!(
                imm8,
                32,
                _mm512_inserti32x4::<{ I >> 3 }>(
                    c,
                    _mm_insert_epi16::<{ I & 7 }>(
                        _mm512_extracti32x4_epi32::<{ I >> 3 }>(c),
                        r1 as i32
                    )
                )
            )),
            op::INSERT_D => Vector(with_imm_below!(
                imm8,
                16,
                _mm512_inserti32x4::<DL>(
                    c,
                    _mm_insert_epi32::<DE>(_mm512_extracti32x4_epi32::<DL>(c), r1 as i32)
                )
            )),
            op::MOVR_Q => Vector(_mm512_zextsi128_si512(_mm_cvtsi64_si128(r1 as i64))),
            op::CMP => Scalar(with_imm!(imm8 & 7, [0 1 2 4 5 6], match imm8 >> 3 {
                0 => _mm512_cmp_epi8_mask::<I>(a, b),
                1 => _mm512_cmp_epu8_mask::<I>(a, b),
                2 => _mm512_cmp_epi16_mask::<I>(a, b) as u64,
                3 => _mm512_cmp_epu16_mask::<I>(a, b) as u64,
                4 => _mm512_cmp_epi32_mask::<I>(a, b) as u64,
                5 => _mm512_cmp_epu32_mask::<I>(a, b) as u64,
                6 => _mm512_cmp_epi64_mask::<I>(a, b) as u64,
                _ => _mm512_cmp_epu64_mask::<I>(a, b) as u64,
            })),
            op::BITSHUF => Scalar(_mm512_bitshuffle_epi64_mask(a, b)),
            op::MUL52LO => Vector(_mm512_madd52lo_epu64(_mm512_setzero_si512(), a, b)),
            op::MUL52HI => Vector(_mm512_madd52hi_epu64(_mm512_setzero_si512(), a, b)),
            op::ABS_Q => Vector(_mm512_abs_epi64(a)),
            op::MAX_Q => Vector(_mm512_max_epi64(a, b)),
            op::MIN_Q => Vector(_mm512_min_epi64(a, b)),
            op::ABS_B => Vector(_mm512_abs_epi8(a)),
            op::ABS_W => Vector(_mm512_abs_epi16(a)),
            op::ABS_D => Vector(_mm512_abs_epi32(a)),
            op::MAX_B => Vector(_mm512_max_epi8(a, b)),
            op::MIN_B => Vector(_mm512_min_epi8(a, b)),
            op::MAXU_B => Vector(_mm512_max_epu8(a, b)),
            op::MINU_B => Vector(_mm512_min_epu8(a, b)),
            op::ADDS_B => Vector(_mm512_adds_epi8(a, b)),
            op::ADDS_W => Vector(_mm512_adds_epi16(a, b)),
            op::ADDUS_B => Vector(_mm512_adds_epu8(a, b)),
            op::ADDUS_W => Vector(_mm512_adds_epu16(a, b)),
            op::SUBS_B => Vector(_mm512_subs_epi8(a, b)),
            op::SUBS_W => Vector(_mm512_subs_epi16(a, b)),
            op::SUBUS_B => Vector(_mm512_subs_epu8(a, b)),
            op::SUBUS_W => Vector(_mm512_subs_epu16(a, b)),
            op::AVG_B => Vector(_mm512_avg_epu8(a, b)),
            op::AVG_W => Vector(_mm512_avg_epu16(a, b)),
            op::SAD => Vector(_mm512_sad_epu8(a, b)),
            op::MULHRS_W => Vector(_mm512_mulhrs_epi16(a, b)),
            op::DPBUSDS => Vector(_mm512_dpbusds_epi32(c, a, b)),
            op::DPWSSDS => Vector(_mm512_dpwssds_epi32(c, a, b)),
            op::TEST => Scalar(match imm8 {
                0 => _mm512_test_epi8_mask(a, b),
                1 => _mm512_test_epi16_mask(a, b) as u64,
                2 => _mm512_test_epi32_mask(a, b) as u64,
                _ => _mm512_test_epi64_mask(a, b) as u64,
            }),
            op::TESTN => Scalar(match imm8 {
                0 => _mm512_testn_epi8_mask(a, b),
                1 => _mm512_testn_epi16_mask(a, b) as u64,
                2 => _mm512_testn_epi32_mask(a, b) as u64,
                _ => _mm512_testn_epi64_mask(a, b) as u64,
            }),
            op::MOVEMASK_W => Scalar(_mm512_movepi16_mask(a) as u64),
            op::MOVEMASK_D => Scalar(_mm512_movepi32_mask(a) as u64),
            op::MOVEMASK_Q => Scalar(_mm512_movepi64_mask(a) as u64),
            op::MOVM_B => Vector(_mm512_movm_epi8(r2)),
            op::MOVM_W => Vector(_mm512_movm_epi16(r2 as u32)),
            op::MOVM_D => Vector(_mm512_movm_epi32(r2 as u16)),
            op::MOVM_Q => Vector(_mm512_movm_epi64(r2 as u8)),
            op::TERNLOG_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_ternarylogic_epi32::<I>(c, a, b)),
            ),
            op::SHLV_W => Vector(_mm512_sllv_epi16(a, b)),
            op::SHRV_W => Vector(_mm512_srlv_epi16(a, b)),
            op::SARV_W => Vector(_mm512_srav_epi16(a, b)),
            op::SARV_D => Vector(_mm512_srav_epi32(a, b)),
            op::SARV_Q => Vector(_mm512_srav_epi64(a, b)),
            op::SHLDI_W => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], _mm512_shldi_epi16::<I>(a, b)),
            ),
            op::SHLDI_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_shldi_epi32::<I>(a, b)),
            ),
            op::SHLDI_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_shldi_epi64::<I>(a, b)),
            ),
            op::SHRDI_W => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], _mm512_shrdi_epi16::<I>(a, b)),
            ),
            op::SHRDI_D => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_shrdi_epi32::<I>(a, b)),
            ),
            op::SHRDI_Q => Vector(
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_shrdi_epi64::<I>(a, b)),
            ),
            op::SHLDV_W => Vector(_mm512_shldv_epi16(c, a, b)),
            op::SHLDV_D => Vector(_mm512_shldv_epi32(c, a, b)),
            op::SHLDV_Q => Vector(_mm512_shldv_epi64(c, a, b)),
            op::SHRDV_W => Vector(_mm512_shrdv_epi16(c, a, b)),
            op::SHRDV_D => Vector(_mm512_shrdv_epi32(c, a, b)),
            op::SHRDV_Q => Vector(_mm512_shrdv_epi64(c, a, b)),
            op::CONFLICT_D => Vector(_mm512_conflict_epi32(a)),
            op::CONFLICT_Q => Vector(_mm512_conflict_epi64(a)),
            op::LZCNT_D => Vector(_mm512_lzcnt_epi32(a)),
            op::LZCNT_Q => Vector(_mm512_lzcnt_epi64(a)),
            op::POPCNT_W => Vector(_mm512_popcnt_epi16(a)),
            op::POPCNT_D => Vector(_mm512_popcnt_epi32(a)),
            op::MULTISHIFT => Vector(_mm512_multishift_epi64_epi8(a, b)),
            op::SHA256RNDS2 => Vector(_mm512_inserti32x4::<0>(
                c,
                _mm_sha256rnds2_epu32(low(c), low(a), low(b)),
            )),
            op::SHA256MSG1 => Vector(_mm512_inserti32x4::<0>(
                c,
                _mm_sha256msg1_epu32(low(c), low(a)),
            )),
            op::SHA1RNDS4 => Vector(_mm512_inserti32x4::<0>(
                c,
                with_imm!(imm8, [0 1 2 3], _mm_sha1rnds4_epu32::<I>(low(c), low(a))),
            )),
            op::SHA256MSG2 => Vector(_mm512_inserti32x4::<0>(
                c,
                _mm_sha256msg2_epu32(low(c), low(a)),
            )),
            op::SHA1NEXTE => Vector(_mm512_inserti32x4::<0>(
                c,
                _mm_sha1nexte_epu32(low(c), low(a)),
            )),
            op::SHA1MSG1 => Vector(_mm512_inserti32x4::<0>(
                c,
                _mm_sha1msg1_epu32(low(c), low(a)),
            )),
            op::SHA1MSG2 => Vector(_mm512_inserti32x4::<0>(
                c,
                _mm_sha1msg2_epu32(low(c), low(a)),
            )),
            op::GF2P8MUL
            | op::GF2P8AFFINE
            | op::GF2P8AFFINEINV
            | op::AESENC
            | op::AESENCLAST
            | op::AESDEC
            | op::AESDECLAST => Vector(native_gfni_vaes(op, imm8, 0, c, a, b, 0)),
            _ => unreachable!("{:#x}", op),
        }
    }

    /// The GFNI and VAES intrinsics, the GFNI ones in mask mode `mode`
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512bw,gfni,vaes")]
    unsafe fn native_gfni_vaes(
        op: u8,
        imm8: u8,
        mode: u8,
        c: __m512i,
        a: __m512i,
        b: __m512i,
        r2: u64,
    ) -> __m512i {
        match (op, mode) {
            (op::GF2P8MUL, 0) => _mm512_gf2p8mul_epi8(a, b),
            (op::GF2P8MUL, 1) => _mm512_mask_gf2p8mul_epi8(c, r2, a, b),
            (op::GF2P8MUL, _) => _mm512_maskz_gf2p8mul_epi8(r2, a, b),
            (op::GF2P8AFFINE, 0) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_gf2p8affine_epi64_epi8::<I>(a, b))
            }
            (op::GF2P8AFFINE, 1) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_mask_gf2p8affine_epi64_epi8::<I>(c, r2, a, b))
            }
            (op::GF2P8AFFINE, _) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_maskz_gf2p8affine_epi64_epi8::<I>(r2, a, b))
            }
            (op::GF2P8AFFINEINV, 0) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_gf2p8affineinv_epi64_epi8::<I>(a, b))
            }
            (op::GF2P8AFFINEINV, 1) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_mask_gf2p8affineinv_epi64_epi8::<I>(c, r2, a, b))
            }
            (op::GF2P8AFFINEINV, _) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_maskz_gf2p8affineinv_epi64_epi8::<I>(r2, a, b))
            }
            (op::AESENC, _) => _mm512_aesenc_epi128(a, b),
            (op::AESENCLAST, _) => _mm512_aesenclast_epi128(a, b),
            (op::AESDEC, _) => _mm512_aesdec_epi128(a, b),
            (op::AESDECLAST, _) => _mm512_aesdeclast_epi128(a, b),
            _ => unreachable!("{:#x}", op),
        }
    }

    /// Masked forms the hardware computes as the unmasked result with lanes of the old value (mode
    /// 1) or zero (mode 2): the lane width is that of the instruction
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn native_masked(
        bytes: usize,
        mode: u8,
        c: __m512i,
        result: __m512i,
        r2: u64,
    ) -> __m512i {
        match (bytes, mode) {
            (1, 1) => _mm512_mask_mov_epi8(c, r2, result),
            (1, _) => _mm512_maskz_mov_epi8(r2, result),
            (2, 1) => _mm512_mask_mov_epi16(c, r2 as u32, result),
            (2, _) => _mm512_maskz_mov_epi16(r2 as u32, result),
            (4, 1) => _mm512_mask_mov_epi32(c, r2 as u16, result),
            (4, _) => _mm512_maskz_mov_epi32(r2 as u16, result),
            (_, 1) => _mm512_mask_mov_epi64(c, r2 as u8, result),
            (_, _) => _mm512_maskz_mov_epi64(r2 as u8, result),
        }
    }

    /// The masked intrinsics themselves for the ops whose masked result is not a plain blend of
    /// the unmasked one (narrowing writes a part of the register) and a sample of the others
    #[allow(clippy::too_many_arguments)]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512ifma,avx512bitalg,avx512vpopcntdq,avx512cd,avx512vbmi,avx512vbmi2"
    )]
    unsafe fn native_mask_intrinsic(
        op: u8,
        imm8: u8,
        mode: u8,
        c: __m512i,
        a: __m512i,
        b: __m512i,
        r1: u64,
        r2: u64,
    ) -> Option<__m512i> {
        let (k8, k16, k32) = (r2 as u8, r2 as u16, r2 as u32);
        let low = _mm512_castsi512_si128;
        let zext128 = _mm512_zextsi128_si512;
        let zext256 = _mm512_zextsi256_si512;
        let low256 = _mm512_castsi512_si256;
        Some(match (op, mode) {
            (op::NARROWUS_D_B, 1) => zext128(_mm512_mask_cvtusepi32_epi8(low(c), k16, a)),
            (op::NARROWUS_D_B, _) => zext128(_mm512_maskz_cvtusepi32_epi8(k16, a)),
            (op::NARROWS_D_B, 1) => zext128(_mm512_mask_cvtsepi32_epi8(low(c), k16, a)),
            (op::NARROWS_D_B, _) => zext128(_mm512_maskz_cvtsepi32_epi8(k16, a)),
            (op::NARROW_D_B, 1) => zext128(_mm512_mask_cvtepi32_epi8(low(c), k16, a)),
            (op::NARROW_D_B, _) => zext128(_mm512_maskz_cvtepi32_epi8(k16, a)),
            (op::NARROWS_D_W, 1) => _mm512_zextsi256_si512(_mm512_mask_cvtsepi32_epi16(
                _mm512_castsi512_si256(c),
                k16,
                a,
            )),
            (op::NARROWS_D_W, _) => _mm512_zextsi256_si512(_mm512_maskz_cvtsepi32_epi16(k16, a)),
            (op::EXTRACT256, 1) => _mm512_zextsi256_si512(with_imm_below!(
                imm8,
                2,
                _mm512_mask_extracti64x4_epi64::<I>(_mm512_castsi512_si256(c), k8, a)
            )),
            (op::EXTRACT256, _) => _mm512_zextsi256_si512(with_imm_below!(
                imm8,
                2,
                _mm512_maskz_extracti64x4_epi64::<I>(k8, a)
            )),
            (op::EXTRACT128, 1) => zext128(with_imm_below!(
                imm8,
                4,
                _mm512_mask_extracti32x4_epi32::<I>(low(c), k8, a)
            )),
            (op::EXTRACT128, _) => zext128(with_imm_below!(
                imm8,
                4,
                _mm512_maskz_extracti32x4_epi32::<I>(k8, a)
            )),
            (op::COMPRESS, 1) => match imm8 {
                0 => _mm512_mask_compress_epi8(c, r2, a),
                1 => _mm512_mask_compress_epi16(c, k32, a),
                2 => _mm512_mask_compress_epi32(c, k16, a),
                _ => _mm512_mask_compress_epi64(c, k8, a),
            },
            (op::COMPRESS, _) => match imm8 {
                0 => _mm512_maskz_compress_epi8(r2, a),
                1 => _mm512_maskz_compress_epi16(k32, a),
                2 => _mm512_maskz_compress_epi32(k16, a),
                _ => _mm512_maskz_compress_epi64(k8, a),
            },
            (op::EXPAND, 1) => match imm8 {
                0 => _mm512_mask_expand_epi8(c, r2, a),
                1 => _mm512_mask_expand_epi16(c, k32, a),
                2 => _mm512_mask_expand_epi32(c, k16, a),
                _ => _mm512_mask_expand_epi64(c, k8, a),
            },
            (op::EXPAND, _) => match imm8 {
                0 => _mm512_maskz_expand_epi8(r2, a),
                1 => _mm512_maskz_expand_epi16(k32, a),
                2 => _mm512_maskz_expand_epi32(k16, a),
                _ => _mm512_maskz_expand_epi64(k8, a),
            },
            (op::NARROW_D_W, 1) => zext256(_mm512_mask_cvtepi32_epi16(low256(c), k16, a)),
            (op::NARROW_D_W, _) => zext256(_mm512_maskz_cvtepi32_epi16(k16, a)),
            (op::NARROWUS_D_W, 1) => zext256(_mm512_mask_cvtusepi32_epi16(low256(c), k16, a)),
            (op::NARROWUS_D_W, _) => zext256(_mm512_maskz_cvtusepi32_epi16(k16, a)),
            (op::NARROW_Q_B, 1) => zext128(_mm512_mask_cvtepi64_epi8(low(c), k8, a)),
            (op::NARROW_Q_B, _) => zext128(_mm512_maskz_cvtepi64_epi8(k8, a)),
            (op::NARROWS_Q_B, 1) => zext128(_mm512_mask_cvtsepi64_epi8(low(c), k8, a)),
            (op::NARROWS_Q_B, _) => zext128(_mm512_maskz_cvtsepi64_epi8(k8, a)),
            (op::NARROWUS_Q_B, 1) => zext128(_mm512_mask_cvtusepi64_epi8(low(c), k8, a)),
            (op::NARROWUS_Q_B, _) => zext128(_mm512_maskz_cvtusepi64_epi8(k8, a)),
            (op::NARROW_Q_W, 1) => zext128(_mm512_mask_cvtepi64_epi16(low(c), k8, a)),
            (op::NARROW_Q_W, _) => zext128(_mm512_maskz_cvtepi64_epi16(k8, a)),
            (op::NARROWS_Q_W, 1) => zext128(_mm512_mask_cvtsepi64_epi16(low(c), k8, a)),
            (op::NARROWS_Q_W, _) => zext128(_mm512_maskz_cvtsepi64_epi16(k8, a)),
            (op::NARROWUS_Q_W, 1) => zext128(_mm512_mask_cvtusepi64_epi16(low(c), k8, a)),
            (op::NARROWUS_Q_W, _) => zext128(_mm512_maskz_cvtusepi64_epi16(k8, a)),
            (op::NARROW_Q_D, 1) => zext256(_mm512_mask_cvtepi64_epi32(low256(c), k8, a)),
            (op::NARROW_Q_D, _) => zext256(_mm512_maskz_cvtepi64_epi32(k8, a)),
            (op::NARROWS_Q_D, 1) => zext256(_mm512_mask_cvtsepi64_epi32(low256(c), k8, a)),
            (op::NARROWS_Q_D, _) => zext256(_mm512_maskz_cvtsepi64_epi32(k8, a)),
            (op::NARROWUS_Q_D, 1) => zext256(_mm512_mask_cvtusepi64_epi32(low256(c), k8, a)),
            (op::NARROWUS_Q_D, _) => zext256(_mm512_maskz_cvtusepi64_epi32(k8, a)),
            (op::NARROW_W_B, 1) => zext256(_mm512_mask_cvtepi16_epi8(low256(c), k32, a)),
            (op::NARROW_W_B, _) => zext256(_mm512_maskz_cvtepi16_epi8(k32, a)),
            (op::NARROWS_W_B, 1) => zext256(_mm512_mask_cvtsepi16_epi8(low256(c), k32, a)),
            (op::NARROWS_W_B, _) => zext256(_mm512_maskz_cvtsepi16_epi8(k32, a)),
            (op::NARROWUS_W_B, 1) => zext256(_mm512_mask_cvtusepi16_epi8(low256(c), k32, a)),
            (op::NARROWUS_W_B, _) => zext256(_mm512_maskz_cvtusepi16_epi8(k32, a)),
            (op::PERM2I, 1) => match imm8 {
                0 => _mm512_mask2_permutex2var_epi8(a, c, r2, b),
                1 => _mm512_mask2_permutex2var_epi16(a, c, k32, b),
                2 => _mm512_mask2_permutex2var_epi32(a, c, k16, b),
                _ => _mm512_mask2_permutex2var_epi64(a, c, k8, b),
            },
            (op::PERM2T, 1) => match imm8 {
                0 => _mm512_mask_permutex2var_epi8(c, r2, a, b),
                2 => _mm512_mask_permutex2var_epi32(c, k16, a, b),
                _ => _mm512_mask_permutex2var_epi64(c, k8, a, b),
            },
            (op::BCAST_W, 2) => {
                _mm512_maskz_permutexvar_epi16(k32, _mm512_set1_epi16(imm8 as i16), a)
            }
            (op::UNPACKLO, 1) if imm8 == 0 => _mm512_mask_unpacklo_epi8(c, r2, a, b),
            (op::SHUFPS, 1) => _mm512_castps_si512(with_imm_below!(
                imm8,
                256,
                _mm512_mask_shuffle_ps::<I>(
                    _mm512_castsi512_ps(c),
                    k16,
                    _mm512_castsi512_ps(a),
                    _mm512_castsi512_ps(b)
                )
            )),
            (op::ALIGNR, 2) => with_imm_below!(imm8, 32, _mm512_maskz_alignr_epi8::<I>(r2, a, b)),
            (op::ADD_B, 1) => _mm512_mask_add_epi8(c, r2, a, b),
            (op::ADD_B, _) => _mm512_maskz_add_epi8(r2, a, b),
            (op::SUB_Q, 1) => _mm512_mask_sub_epi64(c, k8, a, b),
            (op::SUB_Q, _) => _mm512_maskz_sub_epi64(k8, a, b),
            (op::MULWU, 1) => _mm512_mask_mul_epu32(c, k8, a, b),
            (op::MULWU, _) => _mm512_maskz_mul_epu32(k8, a, b),
            (op::DPBUSD, 1) => _mm512_mask_dpbusd_epi32(c, k16, a, b),
            (op::DPBUSD, _) => _mm512_maskz_dpbusd_epi32(k16, c, a, b),
            (op::MADD52LO, 1) => _mm512_mask_madd52lo_epu64(c, k8, a, b),
            (op::MADD52LO, _) => _mm512_maskz_madd52lo_epu64(k8, c, a, b),
            (op::PERM2_W, 1) => _mm512_mask_permutex2var_epi16(c, k32, a, b),
            (op::PERM2_W, _) => _mm512_maskz_permutex2var_epi16(k32, c, a, b),
            (op::MADDUBS, 1) => _mm512_mask_maddubs_epi16(c, k32, a, b),
            (op::MADDUBS, _) => _mm512_maskz_maddubs_epi16(k32, a, b),
            (op::SPLAT_Q, 1) => _mm512_mask_set1_epi64(c, k8, r1 as i64),
            (op::SPLAT_Q, _) => _mm512_maskz_set1_epi64(k8, r1 as i64),
            (op::ZX_B_D, 1) => _mm512_mask_cvtepu8_epi32(c, k16, low(a)),
            (op::ZX_B_D, _) => _mm512_maskz_cvtepu8_epi32(k16, low(a)),
            (op::POPCNT_B, 1) => _mm512_mask_popcnt_epi8(c, r2, a),
            (op::POPCNT_B, _) => _mm512_maskz_popcnt_epi8(r2, a),
            (op::TERNLOG, 1) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_mask_ternarylogic_epi64::<I>(c, k8, a, b))
            }
            (op::GF2P8MUL | op::GF2P8AFFINE | op::GF2P8AFFINEINV, _) => {
                native_gfni_vaes(op, imm8, mode, c, a, b, r2)
            }
            (op::SHLI_D, 1) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_mask_slli_epi32::<U>(c, k16, a))
            }
            (op::TERNLOG_D, 1) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_mask_ternarylogic_epi32::<I>(c, k16, a, b))
            }
            (op::TERNLOG_D, _) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255], _mm512_maskz_ternarylogic_epi32::<I>(k16, c, a, b))
            }
            (op::SARV_W, 1) => _mm512_mask_srav_epi16(c, k32, a, b),
            (op::SARV_W, _) => _mm512_maskz_srav_epi16(k32, a, b),
            (op::SHRDI_Q, 1) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_mask_shrdi_epi64::<I>(c, k8, a, b))
            }
            (op::SHRDI_Q, _) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63], _mm512_maskz_shrdi_epi64::<I>(k8, a, b))
            }
            (op::SHLDV_W, 1) => _mm512_mask_shldv_epi16(c, k32, a, b),
            (op::SHLDV_W, _) => _mm512_maskz_shldv_epi16(k32, c, a, b),
            (op::SHRDV_D, 1) => _mm512_mask_shrdv_epi32(c, k16, a, b),
            (op::SHRDV_D, _) => _mm512_maskz_shrdv_epi32(k16, c, a, b),
            (op::CONFLICT_D, 1) => _mm512_mask_conflict_epi32(c, k16, a),
            (op::CONFLICT_D, _) => _mm512_maskz_conflict_epi32(k16, a),
            (op::LZCNT_Q, 1) => _mm512_mask_lzcnt_epi64(c, k8, a),
            (op::LZCNT_Q, _) => _mm512_maskz_lzcnt_epi64(k8, a),
            (op::POPCNT_W, 1) => _mm512_mask_popcnt_epi16(c, k32, a),
            (op::POPCNT_W, _) => _mm512_maskz_popcnt_epi16(k32, a),
            (op::MULTISHIFT, 1) => _mm512_mask_multishift_epi64_epi8(c, r2, a, b),
            (op::MULTISHIFT, _) => _mm512_maskz_multishift_epi64_epi8(r2, a, b),
            (op::SHLI_D, _) => {
                with_imm!(imm8, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], _mm512_maskz_slli_epi32::<U>(k16, a))
            }
            (op::ADD_Q, 1) => _mm512_mask_add_epi64(c, k8, a, b),
            (op::ADD_Q, _) => _mm512_maskz_add_epi64(k8, a, b),
            (op::MULLO_Q, 1) => _mm512_mask_mullo_epi64(c, k8, a, b),
            (op::MULLO_Q, _) => _mm512_maskz_mullo_epi64(k8, a, b),
            (op::MINU_Q, 1) => _mm512_mask_min_epu64(c, k8, a, b),
            (op::MINU_Q, _) => _mm512_maskz_min_epu64(k8, a, b),
            (op::MADD52HI, 1) => _mm512_mask_madd52hi_epu64(c, k8, a, b),
            (op::MADD52HI, _) => _mm512_maskz_madd52hi_epu64(k8, c, a, b),
            (op::MUL52LO, 2) => _mm512_maskz_madd52lo_epu64(k8, _mm512_setzero_si512(), a, b),
            (op::MUL52HI, 2) => _mm512_maskz_madd52hi_epu64(k8, _mm512_setzero_si512(), a, b),
            (op::ABS_Q, 1) => _mm512_mask_abs_epi64(c, k8, a),
            (op::ABS_Q, _) => _mm512_maskz_abs_epi64(k8, a),
            (op::MAX_Q, 1) => _mm512_mask_max_epi64(c, k8, a, b),
            (op::MAX_Q, _) => _mm512_maskz_max_epi64(k8, a, b),
            (op::MIN_Q, 1) => _mm512_mask_min_epi64(c, k8, a, b),
            (op::MIN_Q, _) => _mm512_maskz_min_epi64(k8, a, b),
            (op::ABS_B, 1) => _mm512_mask_abs_epi8(c, r2, a),
            (op::ABS_B, _) => _mm512_maskz_abs_epi8(r2, a),
            (op::ABS_W, 1) => _mm512_mask_abs_epi16(c, k32, a),
            (op::ABS_D, 2) => _mm512_maskz_abs_epi32(k16, a),
            (op::MAXU_B, 1) => _mm512_mask_max_epu8(c, r2, a, b),
            (op::MIN_B, 2) => _mm512_maskz_min_epi8(r2, a, b),
            (op::ADDS_B, 1) => _mm512_mask_adds_epi8(c, r2, a, b),
            (op::ADDS_W, 2) => _mm512_maskz_adds_epi16(k32, a, b),
            (op::ADDUS_B, 2) => _mm512_maskz_adds_epu8(r2, a, b),
            (op::SUBS_W, 1) => _mm512_mask_subs_epi16(c, k32, a, b),
            (op::SUBUS_B, 1) => _mm512_mask_subs_epu8(c, r2, a, b),
            (op::AVG_W, 1) => _mm512_mask_avg_epu16(c, k32, a, b),
            (op::AVG_B, 2) => _mm512_maskz_avg_epu8(r2, a, b),
            (op::MULHRS_W, 1) => _mm512_mask_mulhrs_epi16(c, k32, a, b),
            (op::MULHRS_W, _) => _mm512_maskz_mulhrs_epi16(k32, a, b),
            (op::DPBUSDS, 1) => _mm512_mask_dpbusds_epi32(c, k16, a, b),
            (op::DPBUSDS, _) => _mm512_maskz_dpbusds_epi32(k16, c, a, b),
            (op::DPWSSDS, 1) => _mm512_mask_dpwssds_epi32(c, k16, a, b),
            (op::DPWSSDS, _) => _mm512_maskz_dpwssds_epi32(k16, c, a, b),
            _ => return None,
        })
    }

    /// The reference semantics through the same entry points the interpreter uses
    #[allow(clippy::too_many_arguments)]
    fn reference(
        info: &OpInfo,
        imm8: u8,
        mode: u8,
        c: &VectorRegister,
        a: &VectorRegister,
        b: &VectorRegister,
        r1: u64,
        r2: u64,
    ) -> (Option<VectorRegister>, Option<u64>) {
        let op = info.op;
        let vector = execute_mask_op(op, imm8, mode, c, a, b, r2).or_else(|| {
            execute_register_op(op, imm8, c, a, b)
                .or_else(|| execute_scalar_input_op(op, imm8, c, r1))
                .map(|result| apply_mask(info, imm8, mode, c, result, r2))
        });
        (vector, execute_scalar_output_op(op, imm8, a, b, r2))
    }

    #[test]
    fn test_vector_ops_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0xa5a5);
        let mut checked = 0;
        for info in implemented().filter(|info| !info.accesses_memory()) {
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
                let modes = modes(info);
                let mode = modes[iteration % modes.len()];
                let r1: u64 = if rng.gen_bool(0.2) {
                    [0, u64::MAX, 0x80, 0x8000_0000][rng.gen_range(0..4)]
                } else {
                    rng.gen()
                };
                let r2 = random_mask(&mut rng);
                let (nc, na, nb) = (to_native(&c), to_native(&a), to_native(&b));
                let mut expected = unsafe { native(info.op, imm8, nc, na, nb, r1, r2) };
                if mode != 0 && matches!(info.mask, Mask::Modes(..)) {
                    let intrinsic =
                        unsafe { native_mask_intrinsic(info.op, imm8, mode, nc, na, nb, r1, r2) };
                    let Native::Vector(result) = expected else {
                        unreachable!()
                    };
                    let blended =
                        unsafe { native_masked(info.mask_lane_bytes(imm8), mode, nc, result, r2) };
                    expected = Native::Vector(match intrinsic {
                        Some(intrinsic) => {
                            if result_is_full(info.op) {
                                assert_eq!(from_native(intrinsic), from_native(blended));
                            }
                            intrinsic
                        }
                        None => {
                            assert!(result_is_full(info.op), "{}", info.name);
                            blended
                        }
                    });
                }
                let actual = reference(info, imm8, mode, &c, &a, &b, r1, r2);
                match expected {
                    Native::Vector(v) => assert_eq!(
                        actual,
                        (Some(from_native(v)), None),
                        "{} imm8={:#x} mode={mode} r2={r2:#x}\nc={:?}\na={:?}\nb={:?}",
                        info.name,
                        imm8,
                        c,
                        a,
                        b
                    ),
                    Native::Scalar(x) => assert_eq!(
                        actual,
                        (None, Some(x)),
                        "{} imm8={:#x} r2={r2:#x}\na={:?}\nb={:?}",
                        info.name,
                        imm8,
                        a,
                        b
                    ),
                }
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            implemented().filter(|info| !info.accesses_memory()).count() * 2000
        );
    }

    /// gf2p8mul on every pair of bytes, the affine maps on every byte with random and edge matrices
    #[test]
    fn test_gf2p8_exhaustive_against_native() {
        if !host_has_avx512() {
            return;
        }
        let zero = VectorRegister::ZERO;
        let mut rng = SmallRng::seed_from_u64(0x9f2);
        for chunk in 0..1024 {
            let a = VectorRegister(std::array::from_fn(|k| ((64 * chunk + k) >> 8) as u8));
            let b = VectorRegister(std::array::from_fn(|k| (64 * chunk + k) as u8));
            let expected = unsafe {
                native_gfni_vaes(
                    op::GF2P8MUL,
                    0,
                    0,
                    to_native(&zero),
                    to_native(&a),
                    to_native(&b),
                    0,
                )
            };
            let info = op_info(op::GF2P8MUL).unwrap();
            assert_eq!(
                reference(info, 0, 0, &zero, &a, &b, 0, 0),
                (Some(from_native(expected)), None)
            );
        }
        for op in [op::GF2P8AFFINE, op::GF2P8AFFINEINV] {
            let info = op_info(op).unwrap();
            for iteration in 0..64 {
                let matrix = VectorRegister(match iteration {
                    0 => [0; 64],
                    1 => [0xff; 64],
                    _ => random_vector(&mut rng),
                });
                let imm8 = rng.gen();
                for base in (0..256).step_by(64) {
                    let x = VectorRegister(std::array::from_fn(|k| (base + k) as u8));
                    let expected = unsafe {
                        native_gfni_vaes(
                            op,
                            imm8,
                            0,
                            to_native(&zero),
                            to_native(&x),
                            to_native(&matrix),
                            0,
                        )
                    };
                    assert_eq!(
                        reference(info, imm8, 0, &zero, &x, &matrix, 0, 0),
                        (Some(from_native(expected)), None)
                    );
                }
            }
        }
    }

    /// Whether the result of a (maskable) op fills the register
    fn result_is_full(op: u8) -> bool {
        !matches!(
            op,
            op::NARROWUS_D_B | op::NARROWS_D_B | op::NARROWS_D_W | op::NARROW_D_B
        ) && !matches!(
            op,
            op::EXTRACT256 | op::EXTRACT128 | op::COMPRESS | op::EXPAND
        ) && !(op::NARROW_D_W..=op::NARROWUS_W_B).contains(&op)
    }

    /// Edge masks and inputs of the mask helpers and compares
    #[test]
    fn test_mask_ops_edges_against_native() {
        if !host_has_avx512() {
            return;
        }
        let zero = VectorRegister::ZERO;
        for info in [op::MPOPCNT, op::MTZCNT, op::MLZCNT].map(|op| op_info(op).unwrap()) {
            for shift in 0..64 {
                for r2 in [0, u64::MAX, 1 << shift, !(1 << shift), u64::MAX >> shift] {
                    let Native::Scalar(expected) = (unsafe {
                        native(
                            info.op,
                            0,
                            to_native(&zero),
                            to_native(&zero),
                            to_native(&zero),
                            0,
                            r2,
                        )
                    }) else {
                        unreachable!()
                    };
                    assert_eq!(
                        reference(info, 0, 0, &zero, &zero, &zero, 0, r2),
                        (None, Some(expected))
                    );
                }
            }
        }
        // Every predicate, signedness and width on lanes of edge values
        let cmp = op_info(op::CMP).unwrap();
        let mut rng = SmallRng::seed_from_u64(9);
        for imm8 in (0..64).filter(|imm8| cmp.imm8_is_valid(*imm8)) {
            let width = 1 << ((imm8 >> 4) & 3);
            for _ in 0..50 {
                let mut a = VectorRegister::ZERO;
                let mut b = VectorRegister::ZERO;
                let bits = 8 * width as u32;
                let edges = [0, 1, u64::MAX, 1 << (bits - 1), (1 << (bits - 1)) - 1];
                for i in 0..64 / width {
                    for v in [&mut a, &mut b] {
                        let value = edges[rng.gen_range(0..edges.len())];
                        v.0[i * width..(i + 1) * width]
                            .copy_from_slice(&value.to_le_bytes()[..width]);
                    }
                    if rng.gen_bool(0.3) {
                        let lane = a.0[i * width..(i + 1) * width].to_vec();
                        b.0[i * width..(i + 1) * width].copy_from_slice(&lane);
                    }
                }
                let Native::Scalar(expected) = (unsafe {
                    native(
                        op::CMP,
                        imm8,
                        to_native(&zero),
                        to_native(&a),
                        to_native(&b),
                        0,
                        0,
                    )
                }) else {
                    unreachable!()
                };
                assert_eq!(
                    reference(cmp, imm8, 0, &zero, &a, &b, 0, 0),
                    (None, Some(expected)),
                    "{imm8:#x}"
                );
            }
        }
    }

    /// Saturating, absolute and min/max arithmetic on edge inputs: every byte pair of the 8-bit ops, every u16 of the 16-bit
    /// ops against edge values, and lanes of edge values for the wider and saturating ops
    #[test]
    fn test_arithmetic_edges_against_native() {
        if !host_has_avx512() {
            return;
        }
        let check = |op: u8, c: &VectorRegister, a: &VectorRegister, b: &VectorRegister| {
            let info = op_info(op).unwrap();
            let Native::Vector(expected) =
                (unsafe { native(op, 0, to_native(c), to_native(a), to_native(b), 0, 0) })
            else {
                unreachable!()
            };
            assert_eq!(
                reference(info, 0, 0, c, a, b, 0, 0),
                (Some(from_native(expected)), None),
                "{}\nc={c:?}\na={a:?}\nb={b:?}",
                info.name
            );
        };
        let zero = VectorRegister::ZERO;
        for chunk in 0..1024 {
            let (mut a, mut b) = (zero, zero);
            for i in 0..64 {
                let pair = chunk * 64 + i;
                (a.0[i], b.0[i]) = ((pair >> 8) as u8, pair as u8);
            }
            for op in [
                op::ABS_B,
                op::MAX_B,
                op::MIN_B,
                op::MAXU_B,
                op::MINU_B,
                op::ADDS_B,
                op::ADDUS_B,
                op::SUBS_B,
                op::SUBUS_B,
                op::AVG_B,
                op::SAD,
            ] {
                check(op, &zero, &a, &b);
                check(op, &zero, &b, &a);
            }
        }
        for chunk in 0..2048 {
            let mut a = zero;
            for i in 0..32 {
                a.set_lane::<2>(i, (chunk * 32 + i) as u64);
            }
            for edge in [0, 1, 2, 0x4000, 0x7fff, 0x8000, 0x8001, 0xc000, 0xffff] {
                let mut b = zero;
                for i in 0..32 {
                    b.set_lane::<2>(i, edge);
                }
                for op in [
                    op::ABS_W,
                    op::ADDS_W,
                    op::ADDUS_W,
                    op::SUBS_W,
                    op::SUBUS_W,
                    op::AVG_W,
                    op::MULHRS_W,
                ] {
                    check(op, &zero, &a, &b);
                    check(op, &zero, &b, &a);
                }
            }
        }
        let mut rng = SmallRng::seed_from_u64(0xb2);
        let edges: [&[u64]; 4] = [
            &[0, 1, 0x7f, 0x80, 0x81, 0xff],
            &[0, 1, 0x7fff, 0x8000, 0x8001, 0xffff],
            &[
                0,
                1,
                0x7fff_fff0,
                0x7fff_ffff,
                0x8000_0000,
                0x8000_000f,
                0xffff_ffff,
            ],
            &[
                0,
                1,
                u64::MAX,
                1 << 63,
                (1 << 63) - 1,
                (1 << 52) - 1,
                1 << 52,
                (1 << 52) + 1,
                0xfff0_0000_0000_0001,
            ],
        ];
        let edge_vector = |rng: &mut SmallRng, width: usize| {
            let mut v = zero;
            for i in 0..64 >> width {
                let lanes = edges[width];
                let value = if rng.gen_bool(0.8) {
                    lanes[rng.gen_range(0..lanes.len())]
                } else {
                    rng.gen()
                };
                v.0[i << width..(i + 1) << width]
                    .copy_from_slice(&value.to_le_bytes()[..1 << width]);
            }
            v
        };
        for _ in 0..20000 {
            for (op, width, c_width) in [
                (op::ABS_D, 2, 2),
                (op::ABS_Q, 3, 3),
                (op::MAX_Q, 3, 3),
                (op::MIN_Q, 3, 3),
                (op::MUL52LO, 3, 3),
                (op::MUL52HI, 3, 3),
                (op::MADD52LO, 3, 3),
                (op::MADD52HI, 3, 3),
                (op::MULLO_Q, 3, 3),
                (op::MINU_Q, 3, 3),
                (op::DPBUSDS, 0, 2),
                (op::DPWSSDS, 1, 2),
            ] {
                let c = edge_vector(&mut rng, c_width);
                let a = edge_vector(&mut rng, width);
                let b = edge_vector(&mut rng, width);
                check(op, &c, &a, &b);
            }
        }
    }

    /// `reference` against `native`, blended for mask modes 1 and 2
    #[allow(clippy::too_many_arguments)]
    fn check_against_native(
        info: &OpInfo,
        imm8: u8,
        mode: u8,
        c: &VectorRegister,
        a: &VectorRegister,
        b: &VectorRegister,
        r2: u64,
    ) {
        let (nc, na, nb) = (to_native(c), to_native(a), to_native(b));
        let expected = match unsafe { native(info.op, imm8, nc, na, nb, 0, r2) } {
            Native::Vector(result) if mode != 0 => (
                Some(from_native(unsafe {
                    native_masked(info.mask_lane_bytes(imm8), mode, nc, result, r2)
                })),
                None,
            ),
            Native::Vector(result) => (Some(from_native(result)), None),
            Native::Scalar(x) => (None, Some(x)),
        };
        assert_eq!(
            reference(info, imm8, mode, c, a, b, 0, r2),
            expected,
            "{} imm8={imm8:#x} mode={mode} r2={r2:#x}\nc={c:?}\na={a:?}\nb={b:?}",
            info.name
        );
    }

    /// Shift counts at and beyond the lane width, conflicts of repeated lanes, every u16 through
    /// popcnt.w and every leading zero count
    #[test]
    fn test_bit_ops_edges_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0xb175);
        let counts: [u64; 17] = [
            0,
            1,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            128,
            255,
            0xffff,
            1 << 32,
            1 << 63,
            u64::MAX,
        ];
        let variable = [
            op::SHLV_W,
            op::SHRV_W,
            op::SARV_W,
            op::SARV_D,
            op::SARV_Q,
            op::SHLDV_W,
            op::SHLDV_D,
            op::SHLDV_Q,
            op::SHRDV_W,
            op::SHRDV_D,
            op::SHRDV_Q,
        ];
        for info in variable.map(|op| op_info(op).unwrap()) {
            let bytes = info.mask_lane_bytes(0);
            for iteration in 0..300 {
                let (c, a) = (
                    VectorRegister(random_vector(&mut rng)),
                    VectorRegister(random_vector(&mut rng)),
                );
                let mut b = VectorRegister::ZERO;
                for i in 0..64 / bytes {
                    let count = counts[(iteration + i) % counts.len()];
                    b.0[i * bytes..(i + 1) * bytes].copy_from_slice(&count.to_le_bytes()[..bytes]);
                }
                let mode = iteration as u8 % 3;
                check_against_native(info, 0, mode, &c, &a, &b, random_mask(&mut rng));
            }
        }
        for info in [op::CONFLICT_D, op::CONFLICT_Q].map(|op| op_info(op).unwrap()) {
            let bytes = info.mask_lane_bytes(0);
            for iteration in 0..300 {
                let c = VectorRegister(random_vector(&mut rng));
                let mut a = VectorRegister::ZERO;
                let values = [0, 1, u64::MAX, rng.gen()];
                for i in 0..64 / bytes {
                    let value = values[rng.gen_range(0..1 + iteration % values.len())];
                    a.0[i * bytes..(i + 1) * bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
                }
                let mode = iteration as u8 % 3;
                check_against_native(info, 0, mode, &c, &a, &c, random_mask(&mut rng));
            }
        }
        let zero = VectorRegister::ZERO;
        let popcnt_w = op_info(op::POPCNT_W).unwrap();
        for chunk in 0..(1 << 16) / 32 {
            let mut a = VectorRegister::ZERO;
            for i in 0..32 {
                a.set_lane::<2>(i, (chunk * 32 + i) as u64);
            }
            check_against_native(popcnt_w, 0, 0, &zero, &a, &zero, 0);
        }
        for info in [op::LZCNT_D, op::LZCNT_Q].map(|op| op_info(op).unwrap()) {
            let bytes = info.mask_lane_bytes(0);
            let bits = 8 * bytes as u32;
            let mut a = VectorRegister::ZERO;
            for shift in 0..bits {
                for (i, value) in [1u64 << shift, (1u64 << shift) - 1, u64::MAX << shift]
                    .iter()
                    .enumerate()
                {
                    a.0[i * bytes..(i + 1) * bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
                }
                check_against_native(info, 0, 0, &zero, &a, &zero, 0);
            }
        }
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

    /// ld.m, st.m, ldn and stn through the interpreter, against native masked loads and stores
    #[test]
    fn test_masked_and_multi_register_memory_ops_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0x6d61);
        for _ in 0..400 {
            let mut input = vec![0u8; 1280];
            rng.fill(&mut input[..]);
            let offset = rng.gen_range(0..128);
            let (width, mode, mask) = (
                rng.gen_range(0..4u8),
                rng.gen_range(1..3u8),
                random_mask(&mut rng),
            );
            let n = rng.gen_range(2..6u8);
            let mut words = lddw(1, ebpf::MM_INPUT_START + offset as u64).to_vec();
            words.extend(lddw(2, mask));
            words.extend([
                encode_with_offset(op::LD, 0, 1, 0, 0, 256),
                encode_masked(op::LD_M, width, 1, 0, 0, 0, mode),
                encode_with_offset(op::LD, 0, 2, 0, 0, 128),
                encode_masked(op::ST_M, width, 0, 2, 0, 64, 1),
                encode_with_offset(op::LDN, n, 3, 0, 0, 320),
                encode_with_offset(op::STN, n, 0, 3, 0, 640),
                exit(),
            ]);
            let outcome = run(
                &verified(&words, config(true, true)),
                &input,
                Input::Writable,
                100,
                false,
            );
            assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
            unsafe {
                let p = input.as_mut_ptr().add(offset);
                let old = _mm512_loadu_si512(p.add(256).cast());
                let loaded = match (width, mode) {
                    (0, 1) => _mm512_mask_loadu_epi8(old, mask, p.cast()),
                    (0, _) => _mm512_maskz_loadu_epi8(mask, p.cast()),
                    (1, 1) => _mm512_mask_loadu_epi16(old, mask as u32, p.cast()),
                    (1, _) => _mm512_maskz_loadu_epi16(mask as u32, p.cast()),
                    (2, 1) => _mm512_mask_loadu_epi32(old, mask as u16, p.cast()),
                    (2, _) => _mm512_maskz_loadu_epi32(mask as u16, p.cast()),
                    (_, 1) => _mm512_mask_loadu_epi64(old, mask as u8, p.cast()),
                    (_, _) => _mm512_maskz_loadu_epi64(mask as u8, p.cast()),
                };
                assert_eq!(outcome.vector_registers[1], from_native(loaded));
                let data = _mm512_loadu_si512(p.add(128).cast());
                match width {
                    0 => _mm512_mask_storeu_epi8(p.add(64).cast(), mask, data),
                    1 => _mm512_mask_storeu_epi16(p.add(64).cast(), mask as u32, data),
                    2 => _mm512_mask_storeu_epi32(p.add(64).cast(), mask as u16, data),
                    _ => _mm512_mask_storeu_epi64(p.add(64).cast(), mask as u8, data),
                }
                for k in 0..n as usize {
                    let value = _mm512_loadu_si512(p.add(320 + 64 * k).cast());
                    assert_eq!(outcome.vector_registers[3 + k], from_native(value));
                    _mm512_storeu_si512(p.add(640 + 64 * k).cast(), value);
                }
            }
            assert_eq!(outcome.memory, input);
        }
    }

    /// ld.x, ld.y, st.x, st.y and ldbc.b/w/x/y (in every mask mode) in both engines, against
    /// native loads, stores and broadcasts
    #[test]
    fn test_narrow_and_broadcast_memory_ops_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0x7862);
        for iteration in 0..400 {
            let mut input = vec![0u8; 512];
            rng.fill(&mut input[..]);
            let offset = rng.gen_range(0..128);
            let mode = (iteration % 3) as u8;
            let mask = random_mask(&mut rng);
            let mut words = lddw(1, ebpf::MM_INPUT_START + offset as u64).to_vec();
            words.extend(lddw(2, mask));
            for reg in 1..9 {
                words.push(encode_with_offset(op::LD, 0, reg, 0, 0, 256));
            }
            words.extend([
                vop(op::LD_X, 0, 1, 0, 0),
                encode_with_offset(op::LD_Y, 0, 2, 0, 0, 7),
                encode_masked(op::LDBC_B, 0, 3, 0, 0, 1, mode),
                encode_masked(op::LDBC_W, 0, 4, 0, 0, 2, mode),
                encode_masked(op::LDBC_X, 0, 5, 0, 0, 3, mode),
                encode_masked(op::LDBC_Y, 0, 6, 0, 0, 4, mode),
                encode_with_offset(op::ST_X, 0, 0, 7, 0, 320),
                encode_with_offset(op::ST_Y, 0, 0, 8, 0, 345),
                exit(),
            ]);
            let outcome = run_both(
                &verified(&words, config(true, iteration % 2 == 0)),
                &input,
                Input::Writable,
                100,
            );
            assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
            unsafe {
                let p = input.as_mut_ptr().add(offset);
                let old = _mm512_loadu_si512(p.add(256).cast());
                let x = _mm_loadu_si128(p.cast());
                assert_eq!(
                    outcome.vector_registers[1],
                    from_native(_mm512_zextsi128_si512(x))
                );
                let y = _mm256_loadu_si256(p.add(7).cast());
                assert_eq!(
                    outcome.vector_registers[2],
                    from_native(_mm512_zextsi256_si512(y))
                );
                let src = if mode == 1 {
                    old
                } else {
                    _mm512_setzero_si512()
                };
                let full = u64::MAX;
                let k = if mode == 0 { full } else { mask };
                let b = _mm_loadu_si128(p.add(1).cast());
                let w = _mm_loadu_si128(p.add(2).cast());
                let x4 = _mm_loadu_si128(p.add(3).cast());
                let y4 = _mm256_loadu_si256(p.add(4).cast());
                for (reg, expected) in [
                    (3, _mm512_mask_broadcastb_epi8(src, k, b)),
                    (4, _mm512_mask_broadcastw_epi16(src, k as u32, w)),
                    (5, _mm512_mask_broadcast_i32x4(src, k as u16, x4)),
                    (6, _mm512_mask_broadcast_i64x4(src, k as u8, y4)),
                ] {
                    assert_eq!(
                        outcome.vector_registers[reg],
                        from_native(expected),
                        "v{reg} mode {mode}"
                    );
                }
                let v7 = _mm512_loadu_si512(p.add(256).cast());
                _mm_storeu_si128(p.add(320).cast(), _mm512_castsi512_si128(v7));
                _mm256_storeu_si256(p.add(345).cast(), _mm512_castsi512_si256(v7));
            }
            assert_eq!(outcome.memory, input);
        }
    }

    /// Gathers and scatters of every index and element width in both engines, against
    /// vpgather* and vpscatter*, with duplicate and (partially) overlapping indices
    #[test]
    fn test_gather_scatter_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0x6761);
        let shapes = [
            (op::GATHER_QQ, op::SCATTER_QQ),
            (op::GATHER_DD, op::SCATTER_DD),
            (op::GATHER_DQ, op::SCATTER_DQ),
            (op::GATHER_QD, op::SCATTER_QD),
        ];
        for iteration in 0..1000 {
            let (gather, scatter) = shapes[iteration % 4];
            let dword_index = matches!(gather, op::GATHER_DD | op::GATHER_DQ);
            let mut input = vec![0u8; 1024];
            rng.fill(&mut input[..]);
            if dword_index {
                for lane in input[..64].chunks_mut(4) {
                    lane.copy_from_slice(&rng.gen_range(-16i32..16).to_le_bytes());
                }
            } else {
                for lane in input[..64].chunks_mut(8) {
                    lane.copy_from_slice(&rng.gen_range(-16i64..16).to_le_bytes());
                }
            }
            let scale = (iteration / 4 % 4) as u8;
            let (gather_mode, scatter_mode) = (rng.gen_range(0..3u8), rng.gen_range(0..2u8));
            let mask = random_mask(&mut rng);
            let mut words = lddw(1, ebpf::MM_INPUT_START).to_vec();
            words.extend([vop(op::LD, 0, 1, 0, 0)]);
            words.extend([encode_with_offset(op::LD, 0, 2, 0, 0, 64)]);
            words.extend([encode_with_offset(op::LD, 0, 3, 0, 0, 128)]);
            words.extend(lddw(1, ebpf::MM_INPUT_START + 512));
            words.extend(lddw(2, mask));
            words.extend([
                encode_masked(gather, scale, 3, 1, 0, 0, gather_mode),
                encode_masked(scatter, scale, 0, 1, 2, 0, scatter_mode),
                exit(),
            ]);
            let outcome = run_both(
                &verified(&words, config(true, iteration % 8 < 4)),
                &input,
                Input::Writable,
                100,
            );
            assert!(outcome.result.starts_with("Ok("), "{}", outcome.result);
            unsafe {
                let p = input.as_mut_ptr();
                let index = _mm512_loadu_si512(p.cast());
                let data = _mm512_loadu_si512(p.add(64).cast());
                let old = _mm512_loadu_si512(p.add(128).cast());
                let base = p.add(512);
                let k = if gather_mode == 0 { u64::MAX } else { mask };
                let src = if gather_mode == 2 {
                    _mm512_setzero_si512()
                } else {
                    old
                };
                let (index_y, src_y) = (_mm512_castsi512_si256(index), _mm512_castsi512_si256(src));
                let gathered = match gather {
                    op::GATHER_QQ => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i64gather_epi64::<I>(src, k as u8, index, base.cast()))
                    }
                    op::GATHER_DD => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i32gather_epi32::<I>(src, k as u16, index, base.cast()))
                    }
                    op::GATHER_DQ => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i32gather_epi64::<I>(src, k as u8, index_y, base.cast()))
                    }
                    _ => _mm512_zextsi256_si512(
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i64gather_epi32::<I>(src_y, k as u8, index, base.cast())),
                    ),
                };
                assert_eq!(outcome.vector_registers[3], from_native(gathered));
                let k = if scatter_mode == 0 { u64::MAX } else { mask };
                let data_y = _mm512_castsi512_si256(data);
                match scatter {
                    op::SCATTER_QQ => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i64scatter_epi64::<I>(base.cast(), k as u8, index, data))
                    }
                    op::SCATTER_DD => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i32scatter_epi32::<I>(base.cast(), k as u16, index, data))
                    }
                    op::SCATTER_DQ => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i32scatter_epi64::<I>(base.cast(), k as u8, index_y, data))
                    }
                    _ => {
                        with_imm!(1 << scale, [1 2 4 8], _mm512_mask_i64scatter_epi32::<I>(base.cast(), k as u8, index, data_y))
                    }
                }
            }
            assert_eq!(outcome.memory, input);
        }
    }

    /// compress, expand and blendm for every d and q mask, the narrowing and widening ops on lanes
    /// of edge values
    #[test]
    fn test_data_movement_edges_against_native() {
        if !host_has_avx512() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0xd1);
        let compare = |info: &OpInfo,
                       imm8: u8,
                       mode: u8,
                       c: &VectorRegister,
                       a: &VectorRegister,
                       b: &VectorRegister,
                       r2: u64| {
            let (nc, na, nb) = (to_native(c), to_native(a), to_native(b));
            let Native::Vector(result) = (unsafe { native(info.op, imm8, nc, na, nb, 0, r2) })
            else {
                unreachable!()
            };
            let expected = match mode {
                0 => result,
                _ => unsafe { native_mask_intrinsic(info.op, imm8, mode, nc, na, nb, 0, r2) }
                    .unwrap_or_else(|| unsafe {
                        native_masked(info.mask_lane_bytes(imm8), mode, nc, result, r2)
                    }),
            };
            assert_eq!(
                reference(info, imm8, mode, c, a, b, 0, r2),
                (Some(from_native(expected)), None),
                "{} imm8={imm8} mode={mode} r2={r2:#x}",
                info.name
            );
        };
        let (c, a, b) = (
            VectorRegister(random_vector(&mut rng)),
            VectorRegister(random_vector(&mut rng)),
            VectorRegister(random_vector(&mut rng)),
        );
        for (width, masks) in [(2u8, 1u64 << 16), (3, 1 << 8)] {
            for r2 in 0..masks {
                // Bits above the lane count are ignored
                let r2 = r2 | rng.gen::<u64>() << (64 >> width);
                for mode in [1, 2] {
                    for op in [op::COMPRESS, op::EXPAND] {
                        compare(op_info(op).unwrap(), width, mode, &c, &a, &b, r2);
                    }
                }
                compare(op_info(op::BLENDM).unwrap(), width, 0, &c, &a, &b, r2);
            }
        }
        let convert = [
            op::NARROW_D_W,
            op::NARROWUS_D_W,
            op::NARROW_Q_B,
            op::NARROWS_Q_B,
            op::NARROWUS_Q_B,
            op::NARROW_Q_W,
            op::NARROWS_Q_W,
            op::NARROWUS_Q_W,
            op::NARROW_Q_D,
            op::NARROWS_Q_D,
            op::NARROWUS_Q_D,
            op::NARROW_W_B,
            op::NARROWS_W_B,
            op::NARROWUS_W_B,
            op::ZX_B_W,
            op::SX_B_W,
            op::ZX_B_Q,
            op::SX_B_Q,
            op::ZX_W_Q,
            op::SX_W_Q,
            op::PACKUS_D,
        ];
        for _ in 0..2000 {
            let mut a = VectorRegister::ZERO;
            let mut b = VectorRegister::ZERO;
            let width = [1, 2, 4, 8][rng.gen_range(0..4)];
            let bits = 8 * width as u32;
            for v in [&mut a, &mut b] {
                for i in 0..64 / width {
                    // Around 0, the signed and unsigned limits of the narrower widths
                    let edge: u64 = [
                        0,
                        1 << 7,
                        1 << 8,
                        1 << 15,
                        1 << 16,
                        1 << 31,
                        1 << 32,
                        1 << 63,
                    ][rng.gen_range(0..8)];
                    let value = edge.wrapping_add(rng.gen_range(-2i64..2) as u64);
                    let value = if rng.gen_bool(0.5) {
                        value
                    } else {
                        value.wrapping_neg()
                    };
                    let value = value & (u64::MAX >> (64 - bits));
                    v.0[i * width..(i + 1) * width].copy_from_slice(&value.to_le_bytes()[..width]);
                }
            }
            let r2 = random_mask(&mut rng);
            for op in convert {
                let info = op_info(op).unwrap();
                for mode in 0..3 {
                    compare(info, 0, mode, &c, &a, &b, r2);
                }
            }
        }
    }
}
