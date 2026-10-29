#![allow(clippy::arithmetic_side_effects)]
//! Tests of the descriptor encoding of the vector ops: `call sol_avx512` with `r1` pointing to a
//! 16 byte descriptor. Compared against the `src = 2` encoding, interpreter vs JIT, the inline
//! fast path vs the slow path, and the cases where the `r1` the JIT expects is not the one at
//! runtime.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use solana_sbpf::{
    ebpf,
    elf::Executable,
    error::EbpfError,
    memory_region::MemoryRegion,
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    vector::{
        descriptor_syscall_hash, encode_descriptor, encode_masked, op, op_info, Kind, OpInfo,
        VectorRegister, DESCRIPTOR_SIZE, OPS,
    },
    verifier::{LocalVerifier, RequisiteVerifier, VerifierError},
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
fn stxdw(dst: u8, src: u8, off: i16) -> u64 {
    insn(ebpf::ST_DW_REG, dst, src, off, 0)
}
fn exit() -> u64 {
    insn(ebpf::EXIT, 0, 0, 0, 0)
}
/// `call sol_avx512`
fn call() -> u64 {
    insn(ebpf::CALL_IMM, 0, 0, 0, descriptor_syscall_hash() as i32)
}

/// A program with a read-only section of descriptors
#[derive(Default, Clone)]
struct Program {
    words: Vec<u64>,
    rodata: Vec<u8>,
}

impl Program {
    /// Appends a descriptor to the read-only section, returns its address
    fn descriptor(&mut self, bytes: [u8; DESCRIPTOR_SIZE]) -> u64 {
        let address = ebpf::MM_RODATA_START + self.rodata.len() as u64;
        self.rodata.extend_from_slice(&bytes);
        address
    }

    /// `lddw r1, descriptor; call sol_avx512`
    fn op(&mut self, bytes: [u8; DESCRIPTOR_SIZE]) {
        let address = self.descriptor(bytes);
        self.words.extend(lddw(1, address));
        self.words.push(call());
    }

    fn push(&mut self, word: u64) {
        self.words.push(word);
    }

    fn extend(&mut self, words: impl IntoIterator<Item = u64>) {
        self.words.extend(words);
    }
}

fn desc(op: u8, imm8: u8, out: u8, in0: u8, in1: u8) -> [u8; DESCRIPTOR_SIZE] {
    encode_descriptor(op, imm8, out, in0, in1, 0, 0)
}

fn config(enable_vector_ops: bool) -> Config {
    Config {
        enable_vector_ops,
        aligned_memory_mapping: true,
        enable_instruction_meter: true,
        ..Config::default()
    }
}

fn executable(program: &Program, config: Config) -> Executable<TestContextObject> {
    let loader = BuiltinProgram::new_loader(config);
    let bytes: Vec<u8> = program.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Executable::from_text_and_rodata_bytes(
        &bytes,
        &program.rodata,
        Arc::new(loader),
        SBPFVersion::V3,
        FunctionRegistry::default(),
    )
    .unwrap()
}

fn verified(program: &Program, config: Config) -> Executable<TestContextObject> {
    let executable = executable(program, config);
    executable.verify::<RequisiteVerifier>().unwrap();
    executable
}

// Execution

#[derive(Debug, PartialEq, Eq, Clone)]
struct Outcome {
    result: String,
    instruction_count: u64,
    pc: u64,
    vector_registers: Vec<VectorRegister>,
    memory: Vec<u8>,
}

fn run(
    executable: &Executable<TestContextObject>,
    input: &[u8],
    budget: u64,
    jit: bool,
) -> Outcome {
    let mut memory = input.to_vec();
    let mut context_object = TestContextObject::new(budget);
    let regions = vec![MemoryRegion::new(&raw mut memory[..], ebpf::MM_INPUT_START)];
    let mut call_frames = vec![CallFrame::default(); executable.get_config().max_call_depth];
    let (instruction_count, result, pc, vector_registers) = {
        create_vm!(
            vm,
            executable,
            &mut context_object,
            stack,
            heap,
            regions,
            None
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
    Outcome {
        result: format!("{result:?}"),
        instruction_count,
        pc,
        vector_registers,
        memory,
    }
}

fn jit_available() -> bool {
    cfg!(all(
        feature = "jit",
        not(target_os = "windows"),
        target_arch = "x86_64"
    ))
}

/// Runs the program in the interpreter and the JIT and checks they agree
fn run_both(executable: &Executable<TestContextObject>, input: &[u8], budget: u64) -> Outcome {
    let interpreted = run(executable, input, budget, false);
    if jit_available() {
        executable.jit_compile().unwrap();
        let mut jitted = run(executable, input, budget, true);
        if interpreted.result.contains("ExceededMaxInstructions") {
            // The JIT only checks the instruction meter at branches and memory accesses
            jitted.vector_registers = interpreted.vector_registers.clone();
        }
        let summary = |o: &Outcome| (o.result.clone(), o.instruction_count, o.pc);
        assert_eq!(
            summary(&interpreted),
            summary(&jitted),
            "interpreter and JIT diverged"
        );
        assert!(
            interpreted == jitted,
            "interpreter and JIT diverged in vector registers or memory"
        );
    }
    interpreted
}

fn random_vector(rng: &mut SmallRng) -> [u8; 64] {
    let mut v = [0u8; 64];
    match rng.gen_range(0..4) {
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

fn implemented() -> impl Iterator<Item = &'static OpInfo> {
    OPS.iter().filter(|info| info.implemented)
}

fn modes(info: &OpInfo) -> Vec<u8> {
    (0..3)
        .filter(|mode| info.modes() & (1 << mode) != 0)
        .collect()
}

/// How `r1` gets the address of the descriptor
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    /// `lddw r1, descriptor` (the JIT emits the op inline)
    Constant,
    /// Computed at runtime, so the JIT does not know it (slow path)
    Computed,
    /// The descriptor is in the writable input region (slow path)
    Input,
    /// The JIT expects a different descriptor than the one `r1` points to (slow path)
    Mismatch,
}

const TRANSPORTS: [Transport; 4] = [
    Transport::Constant,
    Transport::Computed,
    Transport::Input,
    Transport::Mismatch,
];

/// Every implemented op in every allowed mask mode in the descriptor encoding, with each
/// [Transport], against the same op in the `src = 2` encoding. Within an encoding the interpreter
/// and the JIT agree on the whole state, across encodings the result, vector registers, memory
/// and `r0` agree, and `call sol_avx512` leaves `r1` to `r5` unchanged.
#[test]
fn test_descriptor_ops_match_src2() {
    let mut rng = SmallRng::seed_from_u64(0xd35c);
    let register_sets = [
        (3, 4, 5),
        (29, 16, 8),
        (0, 0, 1),
        (17, 17, 17),
        (15, 1, 24),
        (1, 2, 0),
        (20, 0, 21),
        (7, 15, 23),
    ];
    // Input: three vectors, the memory the op accesses, the scalar registers, a descriptor
    const AREA: usize = 192;
    const AREA_LEN: usize = 2048 + 128;
    const DUMP: usize = AREA + AREA_LEN;
    const DESCRIPTOR: usize = DUMP + 6 * 8;
    for info in implemented() {
        let modes = modes(info);
        for iteration in 0..16 {
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
                out = out.min(30 - imm8) * kind.uses_out() as u8;
                in0 = in0.min(30 - imm8) * kind.uses_in0() as u8;
            }
            let gather_scatter = matches!(kind, Kind::Ga | Kind::Sc);
            let wild = gather_scatter && iteration % 8 == 7;
            let mut input = vec![0u8; DESCRIPTOR + DESCRIPTOR_SIZE];
            for k in 0..3 {
                input[64 * k..64 * (k + 1)].copy_from_slice(&random_vector(&mut rng));
            }
            if gather_scatter && !wild {
                for lane in input[64..128].chunks_mut(8) {
                    lane.copy_from_slice(&rng.gen_range(-6i64..6).to_le_bytes());
                }
            }
            rng.fill(&mut input[AREA..DUMP]);
            let (scalar, off) = if info.accesses_memory() {
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
                (rng.gen::<u64>(), 0)
            };
            let mask = random_mask(&mut rng);
            let order = if gather_scatter {
                [(0, out), (2, in1), (1, in0)]
            } else {
                [(0, out), (1, in0), (2, in1)]
            };

            // src = 2: r1 = scalar, r2 = mask
            let mut reference = Program::default();
            reference.push(mov64_reg(6, 1));
            for (k, reg) in order {
                reference.push(mov64_reg(1, 6));
                reference.push(add64_imm(1, 64 * k));
                reference.push(encode_masked(op::LD, 0, reg, 0, 0, 0, 0));
            }
            reference.extend(lddw(1, scalar));
            reference.extend(lddw(2, mask));
            reference.push(mov64_imm(0, 0x1234));
            reference.push(encode_masked(info.op, imm8, out, in0, in1, off, mode));
            reference.push(stxdw(6, 0, DUMP as i16));
            reference.push(exit());
            let reference_input = input.clone();
            let expected = run_both(&verified(&reference, config(true)), &reference_input, 1000);
            assert!(
                wild || expected.result.starts_with("Ok("),
                "{} {:?}",
                info.name,
                expected.result
            );

            let bytes = encode_descriptor(info.op, imm8, out, in0, in1, off, mode);
            for transport in TRANSPORTS {
                // Descriptor encoding: r2 = scalar, r3 = mask
                let mut program = Program::default();
                program.push(mov64_reg(6, 1));
                for (k, reg) in order {
                    program.push(mov64_reg(2, 6));
                    program.push(add64_imm(2, 64 * k));
                    program.op(encode_descriptor(op::LD, 0, reg, 0, 0, 0, 0));
                }
                let address = program.descriptor(bytes);
                program.extend(lddw(2, scalar));
                program.extend(lddw(3, mask));
                program.extend(lddw(4, 0x4444_4444_4444_4444));
                program.extend(lddw(5, 0x5555_5555_5555_5555));
                program.push(mov64_imm(0, 0x1234));
                match transport {
                    Transport::Constant => program.extend(lddw(1, address)),
                    Transport::Computed => {
                        program.extend(lddw(7, address + 8));
                        program.push(add64_imm(7, -8));
                        program.push(mov64_reg(1, 7));
                    }
                    Transport::Input => {
                        program.push(mov64_reg(1, 6));
                        program.push(add64_imm(1, DESCRIPTOR as i32));
                    }
                    Transport::Mismatch => {
                        // The JIT resolves the last lddw, the jump skips it
                        let other = program.descriptor(desc(op::ZERO, 0, out, 0, 0));
                        program.extend(lddw(1, address));
                        program.push(insn(ebpf::JA, 0, 0, 2, 0));
                        program.extend(lddw(1, other));
                    }
                }
                program.push(call());
                for reg in 0..6 {
                    program.push(stxdw(6, reg, (DUMP + 8 * reg as usize) as i16));
                }
                program.push(exit());
                let mut input = input.clone();
                if transport == Transport::Input {
                    input[DESCRIPTOR..].copy_from_slice(&bytes);
                }
                // Also the JIT without sanitization (r1 checked against an imm32) and meter
                let config = match iteration % 3 {
                    0 => config(true),
                    1 => Config {
                        sanitize_user_provided_values: false,
                        ..config(true)
                    },
                    _ => Config {
                        enable_instruction_meter: false,
                        ..config(true)
                    },
                };
                let outcome = run_both(&verified(&program, config), &input, 1000);
                let context = format!("{} mode {mode} {transport:?}", info.name);
                assert_eq!(outcome.result, expected.result, "{context}");
                assert_eq!(
                    outcome.vector_registers, expected.vector_registers,
                    "{context}"
                );
                assert_eq!(
                    outcome.memory[..DUMP + 8],
                    expected.memory[..DUMP + 8],
                    "{context}"
                );
                if outcome.result.starts_with("Ok(") {
                    let dumped = |reg: usize| {
                        u64::from_le_bytes(
                            outcome.memory[DUMP + 8 * reg..DUMP + 8 * reg + 8]
                                .try_into()
                                .unwrap(),
                        )
                    };
                    let r1 = match transport {
                        Transport::Input => ebpf::MM_INPUT_START + DESCRIPTOR as u64,
                        _ => address,
                    };
                    assert_eq!(
                        (1..6).map(dumped).collect::<Vec<_>>(),
                        [
                            r1,
                            scalar,
                            mask,
                            0x4444_4444_4444_4444,
                            0x5555_5555_5555_5555
                        ],
                        "{context}"
                    );
                }
            }
        }
    }
}

/// Philip's example: the `r1` at the call depends on a branch. The JIT expects the last load and
/// checks it at runtime, every path runs the op of the descriptor `r1` points to.
#[test]
fn test_descriptor_depends_on_branch() {
    let mut program = Program::default();
    let add = program.descriptor(desc(op::ADD_D, 0, 2, 0, 1));
    let xor = program.descriptor(desc(op::XOR, 0, 2, 0, 1));
    program.extend([mov64_reg(2, 1), insn(ebpf::LD_DW_REG, 4, 1, 128, 0)]);
    program.op(desc(op::LD, 0, 0, 0, 0));
    program.push(add64_imm(2, 64));
    program.op(desc(op::LD, 0, 1, 0, 0));
    program.extend(lddw(1, add));
    program.push(insn(ebpf::JEQ64_IMM, 4, 0, 2, 6)); // jeq r4, 6, +2
    program.extend(lddw(1, xor));
    program.push(call());
    program.op(desc(op::EXTRACT_Q, 0, 0, 2, 0));
    program.push(exit());
    let executable = verified(&program, config(true));
    for (r4, expected) in [(6u64, 3u64 + 5), (7, 3 ^ 5)] {
        let mut input = vec![0u8; 136];
        input[..8].copy_from_slice(&3u64.to_le_bytes());
        input[64..72].copy_from_slice(&5u64.to_le_bytes());
        input[128..].copy_from_slice(&r4.to_le_bytes());
        let outcome = run_both(&executable, &input, 100);
        assert_eq!(outcome.result, format!("Ok({expected})"), "r4 = {r4}");
    }
}

/// Entering at the call by callx, with an `r1` the JIT does not expect
#[test]
fn test_descriptor_call_entered_by_callx() {
    let mut program = Program::default();
    let add = program.descriptor(desc(op::ADD_Q, 0, 0, 0, 0));
    let sub = program.descriptor(desc(op::SUB_Q, 0, 0, 0, 0));
    // v0 = 1 in every lane, then callx to the call below with r1 = sub, which returns v0 - v0
    program.extend(lddw(2, 1));
    program.op(desc(op::SPLAT_Q, 0, 0, 0, 0));
    let target = program.words.len() + 2 + 2 + 1 + 1 + 2;
    program.extend(lddw(7, ebpf::MM_BYTECODE_START + 8 * target as u64));
    program.extend(lddw(1, sub));
    program.push(insn(ebpf::CALL_REG, 7, 0, 0, 0));
    program.push(exit());
    program.extend(lddw(1, add));
    assert_eq!(program.words.len(), target);
    program.push(call());
    program.op(desc(op::EXTRACT_Q, 0, 0, 0, 0));
    program.push(exit());
    let outcome = run_both(&verified(&program, config(true)), &[], 100);
    assert_eq!(outcome.result, "Ok(0)");
}

/// `r1` outside of the read-only section: the slow path reads it like a scalar load
#[test]
fn test_descriptor_faults() {
    let mut program = Program::default();
    let valid = program.descriptor(desc(op::ADD_Q, 0, 0, 0, 0));
    let mut reserved = desc(op::ADD_Q, 0, 0, 0, 0);
    reserved[15] = 1;
    let reserved = program.descriptor(reserved);
    let register = program.descriptor(desc(op::ADD_Q, 0, 30, 0, 0));
    let wide = program.descriptor(desc(op::ADD_Q, 0, 33, 0, 0));
    let unknown = program.descriptor(desc(0xff, 0, 0, 0, 0));
    let mode = program.descriptor(encode_descriptor(op::ADD_Q, 0, 0, 0, 0, 0, 3));
    let offset = program.descriptor(encode_descriptor(op::ADD_Q, 0, 0, 0, 0, 8, 0));
    let end = ebpf::MM_RODATA_START + program.rodata.len() as u64;
    let unsupported = "Err(UnsupportedInstruction)".to_string();
    let violation = |address: u64| format!("Err(AccessViolation(Load, {address}, 16, ");
    let cases = [
        (valid, "Ok(0)".to_string()),
        (reserved, unsupported.clone()),
        (register, unsupported.clone()),
        (wide, unsupported.clone()),
        (unknown, unsupported.clone()),
        (mode, unsupported.clone()),
        (offset, unsupported),
        (end - 8, violation(end - 8)),
        (0x7777_0000_0000, violation(0x7777_0000_0000)),
        (
            ebpf::MM_INPUT_START + 8,
            violation(ebpf::MM_INPUT_START + 8),
        ),
    ];
    for (address, expected) in cases {
        for constant in [true, false] {
            let mut p = program.clone();
            if constant {
                p.extend(lddw(1, address));
            } else {
                p.extend(lddw(1, address ^ 1));
                p.push(insn(ebpf::XOR64_IMM, 1, 0, 0, 1));
            }
            p.extend([call(), mov64_imm(0, 0), exit()]);
            let outcome = run_both(&verified(&p, config(true)), &[0u8; 16], 100);
            assert!(
                outcome.result.starts_with(&expected),
                "{:#x} {} {}",
                address,
                constant,
                outcome.result
            );
        }
    }
}

/// The JIT follows copies of the descriptor address and loads hoisted out of a loop
#[test]
fn test_descriptor_address_copies() {
    let mut program = Program::default();
    let add = program.descriptor(desc(op::ADD_Q, 0, 0, 0, 1));
    program.extend(lddw(2, 1));
    program.op(desc(op::SPLAT_Q, 0, 1, 0, 0));
    program.extend(lddw(6, add));
    program.push(mov64_imm(7, 100));
    // loop: r1 = r6, v0 += v1, r7 -= 1, jne r7, 0, loop
    program.push(mov64_reg(1, 6));
    program.push(call());
    program.push(add64_imm(7, -1));
    program.push(insn(ebpf::JNE64_IMM, 7, 0, -4, 0));
    program.op(desc(op::EXTRACT_Q, 0, 0, 0, 0));
    program.push(exit());
    let outcome = run_both(&verified(&program, config(true)), &[], 1000);
    assert_eq!(outcome.result, "Ok(100)");
}

/// The CUs: with vector_pair_one_cu a `lddw r1` directly followed by the call is free, so the pair
/// costs the CUs of the op. Ops of more than one CU fail before any effect if they do not fit.
#[test]
fn test_descriptor_instruction_meter() {
    let gather = op_info(op::GATHER_QQ).unwrap();
    let cu = gather.cu(0);
    assert!(cu > 1);
    let mut adjacent = Program::default();
    adjacent.push(mov64_reg(2, 1));
    adjacent.op(encode_descriptor(op::GATHER_QQ, 0, 1, 0, 0, 0, 0));
    adjacent.op(desc(op::EXTRACT_Q, 0, 0, 1, 0));
    adjacent.push(exit());
    let mut apart = Program::default();
    apart.push(mov64_reg(2, 1));
    let address = apart.descriptor(encode_descriptor(op::GATHER_QQ, 0, 1, 0, 0, 0, 0));
    apart.extend(lddw(1, address));
    apart.push(mov64_reg(3, 3));
    apart.push(call());
    apart.op(desc(op::EXTRACT_Q, 0, 0, 1, 0));
    apart.push(exit());
    let mut slow = apart.clone();
    slow.words[1] = insn(ebpf::LD_DW_IMM, 1, 0, 0, address as i32 + 1);
    slow.words[3] = add64_imm(1, -1);
    let input: Vec<u8> = (0..64).collect();
    for (program, pair_one_cu, total) in [
        (&adjacent, true, 1 + cu + 1 + 1),
        (&adjacent, false, 1 + 1 + cu + 2 + 1),
        (&apart, true, 1 + 1 + 1 + cu + 1 + 1),
        (&slow, true, 1 + 1 + 1 + cu + 1 + 1),
    ] {
        let config = Config {
            vector_pair_one_cu: pair_one_cu,
            ..config(true)
        };
        let executable = verified(program, config);
        for budget in 0..total + 2 {
            let outcome = run_both(&executable, &input, budget);
            if budget < total {
                assert!(
                    outcome.result.contains("ExceededMaxInstructions"),
                    "{budget} {}",
                    outcome.result
                );
                if budget >= total - 2 {
                    // Ran out at the extract, after the gather
                    assert_eq!(outcome.vector_registers[1].0[..8], input[..8]);
                }
            } else {
                let gathered = u64::from_le_bytes(input[..8].try_into().unwrap());
                assert_eq!(outcome.result, format!("Ok({gathered})"), "{budget}");
                assert_eq!(outcome.instruction_count, total, "{budget}");
            }
        }
    }
}

/// `call sol_avx512` needs `dst` and `off` 0, and is only a vector op if enabled
#[test]
fn test_descriptor_verifier() {
    let mut program = Program::default();
    let address = program.descriptor(desc(op::ZERO, 0, 0, 0, 0));
    program.extend(lddw(1, address));
    program.push(call());
    program.push(exit());
    verified(&program, config(true));
    for word in [call() | 1 << 8, call() | 1 << 16] {
        let mut invalid = program.clone();
        invalid.words[2] = word;
        assert_eq!(
            executable(&invalid, config(true))
                .verify::<RequisiteVerifier>()
                .unwrap_err()
                .to_string(),
            EbpfError::VerifierError(VerifierError::InvalidVectorOp(
                2,
                "call sol_avx512 must have dst and off 0"
            ))
            .to_string()
        );
    }
    // The deploy checks see a registered syscall, without the flag an unknown one
    executable(&program, config(true))
        .verify::<LocalVerifier>()
        .unwrap();
    let result = executable(&program, config(false)).verify::<LocalVerifier>();
    assert!(
        matches!(
            result,
            Err(EbpfError::VerifierError(VerifierError::InvalidSyscall(_)))
        ),
        "{:?}",
        result
    );
}

/// Whether the JIT turned the instruction at `pc` into the inside of a memory run: callx and
/// jumps to it then enter through `xor r11d, r11d; jmp`
fn inside_memory_run(executable: &Executable<TestContextObject>, pc: usize) -> bool {
    let program = executable.get_compiled_program().unwrap();
    let code = &program.text_section()[program.pc_section()[pc] as usize..];
    code.starts_with(&[0x45, 0x31, 0xdb, 0xe9]) || code.starts_with(&[0x45, 0x31, 0xdb, 0x90, 0xe9])
}

/// Consecutive inline `call sol_avx512` memory accesses form a memory run, also with callx into
/// the middle, faults, partly mapped spans and every budget
#[test]
fn test_descriptor_memory_runs() {
    let mut program = Program::default();
    program.push(mov64_reg(2, 1));
    let first = program.words.len();
    program.op(encode_descriptor(op::LD, 0, 0, 0, 0, 0, 0));
    program.op(encode_descriptor(op::LD, 0, 1, 0, 0, 64, 0));
    program.op(desc(op::ADD_Q, 0, 2, 0, 1));
    let store = program.words.len();
    program.op(encode_descriptor(op::ST, 0, 0, 2, 0, 128, 0));
    program.op(encode_descriptor(op::LD, 0, 3, 0, 0, 256, 0));
    program.op(encode_descriptor(op::ST, 0, 0, 3, 0, 192, 0));
    program.push(mov64_imm(0, 0));
    program.push(exit());
    let executable = verified(&program, config(true));
    for len in [512, 320, 300, 200, 64] {
        let input: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let full = run_both(&executable, &input, 1000);
        assert_eq!(
            full.result.starts_with("Ok("),
            len >= 320,
            "{len} {}",
            full.result
        );
        for budget in 0..20 {
            run_both(&executable, &input, budget);
        }
    }
    if jit_available() && !cfg!(feature = "tracer") {
        // The run starts at the first access, the later lddw and calls are inside
        assert!(!inside_memory_run(&executable, first + 2));
        for pc in [first + 3, first + 5, store, store + 2, store + 8] {
            assert!(inside_memory_run(&executable, pc), "{}", pc);
        }
    }
    // callx to the store inside the run, with r1 = the store (v2 = 0) or another descriptor (v1)
    let other = program.descriptor(encode_descriptor(op::ST, 0, 0, 1, 0, 128, 0));
    let [lo, hi] = [program.words[store], program.words[store + 1]];
    let store_descriptor = (lo >> 32) | (hi >> 32) << 32;
    let input: Vec<u8> = (0..512).map(|i| (i * 7 + 3) as u8).collect();
    let v1 = u64::from_le_bytes(input[64..72].try_into().unwrap());
    for (r1, expected) in [(store_descriptor, 0), (other, v1)] {
        let mut callx = Program {
            rodata: program.rodata.clone(),
            ..Program::default()
        };
        callx.extend([mov64_reg(2, 1), mov64_reg(6, 1)]);
        callx.op(desc(op::ZERO, 0, 2, 0, 0));
        callx.op(encode_descriptor(op::LD, 0, 1, 0, 0, 64, 0));
        let offset = callx.words.len() + 2 + 2 + 1 + 1 + 1;
        callx.extend(lddw(
            7,
            ebpf::MM_BYTECODE_START + 8 * (offset + store + 2) as u64,
        ));
        callx.extend(lddw(1, r1));
        callx.push(insn(ebpf::CALL_REG, 7, 0, 0, 0));
        callx.push(insn(ebpf::LD_DW_REG, 0, 6, 128, 0));
        callx.push(exit());
        assert_eq!(callx.words.len(), offset);
        callx.extend(program.words.iter().copied());
        let executable = verified(&callx, config(true));
        if jit_available() && !cfg!(feature = "tracer") {
            executable.jit_compile().unwrap();
            assert!(inside_memory_run(&executable, offset + store + 2));
        }
        let outcome = run_both(&executable, &input, 1000);
        assert_eq!(outcome.result, format!("Ok({expected})"));
    }
}

/// Prints CUs and JIT time per op of loops of ops in both encodings, and JIT compile times:
/// cargo test --release --test vector_rodata bench -- --ignored --nocapture
#[test]
#[ignore]
fn bench_descriptor_vs_src2() {
    use std::time::Instant;
    const ITERATIONS: i32 = 4096;
    const UNROLL: usize = 16;
    // (name, ops as (op, imm8, out, in0, in1, off), input bytes)
    type Op = (u8, u8, u8, u8, u8, i16);
    let kernels: [(&str, Vec<Op>); 4] = [
        ("add.q chain", vec![(op::ADD_Q, 0, 0, 0, 1, 0)]),
        (
            "add.q independent",
            (0..8).map(|k| (op::ADD_Q, 0, k, k, 8, 0)).collect(),
        ),
        (
            "ld, ld, add, st",
            vec![
                (op::LD, 0, 0, 0, 0, 0),
                (op::LD, 0, 1, 0, 0, 64),
                (op::ADD_Q, 0, 2, 0, 1, 0),
                (op::ST, 0, 0, 2, 0, 128),
            ],
        ),
        ("vpternlogq", vec![(op::TERNLOG, 0x96, 0, 1, 2, 0)]),
    ];
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Encoding {
        Src2,
        Descriptor,
        SlowPath,
    }
    let build = |ops: &[Op], encoding: Encoding, body: bool| {
        let mut program = Program::default();
        program.push(mov64_reg(2, 1));
        program.push(mov64_reg(9, 1));
        program.push(mov64_imm(7, ITERATIONS));
        let loop_start = program.words.len();
        if body {
            for k in 0..UNROLL {
                let (op, imm8, out, in0, in1, off) = ops[k % ops.len()];
                let bytes = encode_descriptor(op, imm8, out, in0, in1, off, 0);
                match encoding {
                    Encoding::Src2 => program.push(encode_masked(op, imm8, out, in0, in1, off, 0)),
                    Encoding::Descriptor => program.op(bytes),
                    Encoding::SlowPath => {
                        let address = program.descriptor(bytes);
                        program.extend(lddw(1, address + 1));
                        program.push(add64_imm(1, -1));
                        program.push(call());
                    }
                }
            }
        }
        let back = loop_start as i64 - program.words.len() as i64 - 2;
        program.push(add64_imm(7, -1));
        program.push(insn(ebpf::JNE64_IMM, 7, 0, back as i16, 0));
        program.push(mov64_imm(0, 0));
        program.push(exit());
        program
    };
    let input = vec![0u8; 256];
    let time = |program: &Program| {
        let executable = verified(program, config(true));
        executable.jit_compile().unwrap();
        let mut samples: Vec<(u128, u64)> = (0..15)
            .map(|_| {
                let start = Instant::now();
                let outcome = run(&executable, &input, u64::MAX >> 8, true);
                assert_eq!(outcome.result, "Ok(0)");
                (start.elapsed().as_nanos(), outcome.instruction_count)
            })
            .collect();
        samples.sort();
        samples[samples.len() / 2]
    };
    println!("| kernel | encoding | CU / op | JIT ns / op |\n|---|---|---|---|");
    let ops = (ITERATIONS as usize * UNROLL) as f64;
    for (name, kernel) in &kernels {
        let (empty_ns, empty_cu) = time(&build(kernel, Encoding::Src2, false));
        for encoding in [Encoding::Src2, Encoding::Descriptor, Encoding::SlowPath] {
            let (ns, cu) = time(&build(kernel, encoding, true));
            println!(
                "| {name} | {encoding:?} | {:.2} | {:.2} |",
                (cu - empty_cu) as f64 / ops,
                (ns as f64 - empty_ns as f64) / ops
            );
        }
    }
    // JIT compile time of a straight line of 20000 ops
    println!("\n| encoding | ops | JIT compile µs |\n|---|---|---|");
    for encoding in [Encoding::Src2, Encoding::Descriptor] {
        let mut program = Program::default();
        program.push(mov64_reg(2, 1));
        for k in 0..20000usize {
            let (op, imm8, out, in0, in1, off) = match k % 4 {
                0 => (op::LD, 0, (k % 8) as u8, 0, 0, 0),
                1 => (op::ADD_Q, 0, 1, 2, 3, 0),
                2 => (op::TERNLOG, 0x96, 4, 5, 6, 0),
                _ => (op::ST, 0, 0, 7, 0, 64),
            };
            match encoding {
                Encoding::Src2 => program.push(encode_masked(op, imm8, out, in0, in1, off, 0)),
                _ => program.op(encode_descriptor(op, imm8, out, in0, in1, off, 0)),
            }
        }
        program.push(exit());
        let mut samples: Vec<u128> = (0..9)
            .map(|_| {
                let executable = verified(&program, config(true));
                let start = Instant::now();
                executable.jit_compile().unwrap();
                start.elapsed().as_micros()
            })
            .collect();
        samples.sort();
        println!("| {encoding:?} | 20000 | {} |", samples[samples.len() / 2]);
    }
}
