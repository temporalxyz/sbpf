//! AVX-512 lowering of the vector ops (child module of [crate::jit])
//!
//! Design:
//! - Vector register `vN` is pinned to `zmmN`. `zmm30`, `zmm31` and `k1` are scratch.
//! - The JIT pre-scans the program for the set of vector registers it uses (the "used set"). If
//!   the set is empty nothing in the generated code changes.
//! - [EbpfVm::vector_registers] is the canonical state outside of JIT code. The entry prologue
//!   loads the used set from it, the epilogue stores it back. Every call into host code
//!   ([JitCompiler::emit_rust_call]) stores the used set before and loads it again afterwards,
//!   because host code may clobber all vector registers. `vzeroupper` precedes every host call
//!   and the return to the host to avoid AVX/SSE transition penalties.
//! - All memory accesses (vector and scalar) of a program using vector ops go through the
//!   translation subroutines below. They translate the address inline using only general
//!   purpose registers if the memory mapping is aligned and the access is plain (in bounds,
//!   not gapped, permitted). Everything else goes to [MemoryMapping::map_with_access_violation_handler]
//!   which is exactly what the scalar path calls. So results, errors and CUs match.
//! - A run of vector ops with several memory accesses translates the address range they cover
//!   once, see [MemoryRuns].

use super::*;
use crate::{
    error::StableResult,
    memory_region::{jit_region_layout, AccessType, JitRegionTable},
    vector::{op, VectorInsn, VECTOR_OP_SRC, VECTOR_REGISTERS, VECTOR_SIZE},
    x86_evex::{kmovq_to_gpr, EvexInstruction, Length, Map::*, Prefix::*, Rm, VZEROUPPER},
};
use std::convert::TryFrom;

/// Machine code length of the vector subroutines (prologue and translation), upper bound
pub(super) const MAX_VECTOR_SUBROUTINES_MACHINE_CODE_LENGTH: usize = 32 * 1024;

/// Lengths of the memory accesses which have a translation subroutine
const TRANSLATION_LENGTHS: [u64; 5] = [1, 2, 4, 8, VECTOR_SIZE as u64];

/// Machine code length of the fallback and the entry of one instruction of a [MemoryRun], upper bound
const MAX_MEMORY_RUN_TRAILER_LENGTH_PER_INSTRUCTION: usize = 80;

/// Scratch vector registers
const ZMM30: u8 = 30;
const ZMM31: u8 = 31;
/// Scratch mask register
const K1: u8 = 1;

/// Returns the name of a host CPU feature the lowering needs but is missing
pub(super) fn missing_host_feature() -> Option<&'static str> {
    [
        (is_x86_feature_detected!("avx512f"), "avx512f"),
        (is_x86_feature_detected!("avx512bw"), "avx512bw"),
        (is_x86_feature_detected!("avx512dq"), "avx512dq"),
        (is_x86_feature_detected!("avx512vl"), "avx512vl"),
        (is_x86_feature_detected!("avx512vnni"), "avx512vnni"),
        (is_x86_feature_detected!("avx512ifma"), "avx512ifma"),
        (is_x86_feature_detected!("avx512bitalg"), "avx512bitalg"),
        (
            is_x86_feature_detected!("avx512vpopcntdq"),
            "avx512vpopcntdq",
        ),
        (is_x86_feature_detected!("vpclmulqdq"), "vpclmulqdq"),
    ]
    .iter()
    .find_map(|&(detected, name)| (!detected).then_some(name))
}

/// Returns the anchor of the translation subroutine for the given access
fn translation_anchor(access_type: AccessType, len: u64) -> usize {
    let index = TRANSLATION_LENGTHS.iter().position(|l| *l == len).unwrap();
    ANCHOR_VECTOR_TRANSLATE
        + (access_type == AccessType::Store) as usize * TRANSLATION_LENGTHS.len()
        + index
}

/// Slow path of the translation subroutines
///
/// Writes `Ok(host_address)` or the error of the access to `result`.
extern "C" fn translate_slow_path(
    result: &mut ProgramResult,
    memory_mapping: &mut MemoryMapping,
    vm_addr: u64,
    len: u64,
    access_type: u64,
) {
    #[cfg(test)]
    SLOW_PATH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let access_type = if access_type == 0 {
        AccessType::Load
    } else {
        AccessType::Store
    };
    *result = match memory_mapping.map_with_access_violation_handler(access_type, vm_addr, len) {
        StableResult::Ok(host_buffer) => ProgramResult::Ok(host_buffer.ptr().cast::<u8>() as u64),
        StableResult::Err(err) => ProgramResult::Err(err),
    };
}

/// Memory runs: consecutive vector ops (nothing else in between, so `r1` is unchanged and no host
/// code runs) with two or more memory accesses, no jump target after the first access and no
/// instruction meter checkpoint inside.
///
/// The first access calls a span translation subroutine for the range all accesses of the run
/// cover. If the range lies in one plain region (the condition the translation fast path checks
/// for every single access) REGISTER_SCRATCH holds its host address (minus a key) for the rest of
/// the run and each access is `[REGISTER_SCRATCH + displacement]` after the usual instruction
/// meter check. Otherwise REGISTER_SCRATCH is 0 and every access jumps to its fallback, which is
/// exactly the code a single access emits (and leaves REGISTER_SCRATCH 0). So faults, the access
/// violation handler and CUs behave as without runs. Entering a run anywhere but at its start
/// (`callx` may target any instruction) goes through an entry which zeroes REGISTER_SCRATCH.
#[derive(Default)]
pub(super) struct MemoryRuns {
    /// Text section offsets of the span translation subroutines for loads only and for runs with
    /// stores, if the region layout allows inline translation
    span_translation: Option<[usize; 2]>,
    /// Whether a pc may be reached other than from the previous instruction, filled on first use
    jump_targets: Vec<bool>,
    /// The run being emitted
    run: Option<MemoryRun>,
}

/// See [MemoryRuns]
struct MemoryRun {
    /// pc of the first memory access
    start: usize,
    /// pc of the last memory access
    end: usize,
    /// Smallest displacement of the accesses
    min_off: i16,
    /// The accesses emitted so far
    fallbacks: Vec<MemoryRunFallback>,
}

/// An access of a [MemoryRun] whose fallback is emitted at the end of the run
struct MemoryRunFallback {
    pc: usize,
    insn: VectorInsn,
    /// Text section offset of the rel32 of the jump to the fallback
    jump: usize,
    /// Text section offset of the code after the access
    resume: usize,
}

/// One host instruction of a lowering
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostInstruction {
    /// AVX-512 instruction
    Evex(EvexInstruction),
    /// `kmovq r64, k`
    KmovqToGpr(X86Register, u8),
}

impl HostInstruction {
    /// Machine code bytes
    #[cfg(test)]
    pub(crate) fn encode(&self) -> Vec<u8> {
        match self {
            Self::Evex(instruction) => {
                let (bytes, len) = instruction.encode();
                bytes[..len].to_vec()
            }
            Self::KmovqToGpr(destination, mask) => kmovq_to_gpr(*destination, *mask).to_vec(),
        }
    }
}

/// The lowering of a vector op
///
/// If `memory` is set, the address in `r1` is first translated into `REGISTER_SCRATCH` (R11)
/// and the instructions access memory through it.
pub(crate) struct Lowering {
    pub(crate) memory: Option<(AccessType, u64)>,
    pub(crate) instructions: Vec<HostInstruction>,
}

/// Lowers a vector op to AVX-512, see the avx-512 column of the spec
pub(crate) fn lower(v: &VectorInsn) -> Lowering {
    use HostInstruction::Evex;
    type E = EvexInstruction;
    let (o, a, b, imm8) = (v.out, v.in0, v.in1, v.imm8);
    let r0 = REGISTER_MAP[0];
    let r1 = REGISTER_MAP[1] as u8;
    // out ← f(in0, in1)
    let binary = |map, w, opcode| vec![Evex(E::rrr(P66, map, w, opcode, o, a, b))];
    // out ← f(in0)
    let unary = |prefix, w, opcode| vec![Evex(E::rr(prefix, M0F38, w, opcode, o, a))];
    // out ← f(in0, imm8), ModRM.reg is an opcode extension and EVEX.vvvv the destination
    let shift =
        |w, opcode, extension| vec![Evex(E::rrr(P66, M0F, w, opcode, extension, o, a).imm(imm8))];
    // k1 ← compare(in0, in1); out ← vpmovm2*(k1)
    let compare =
        |compare: E, w, opcode| vec![Evex(compare), Evex(E::rr(PF3, M0F38, w, opcode, o, K1))];
    // out ← narrowing conversion of in0, which is the ModRM.reg operand
    let narrow = |opcode| vec![Evex(E::rr(PF3, M0F38, false, opcode, a, o))];
    let mut memory = None;
    let mut load = |access_type: AccessType, len: u64, instruction: E| {
        memory = Some((access_type, len));
        vec![Evex(instruction.mem(REGISTER_SCRATCH, 0))]
    };
    let instructions = match v.info.op {
        op::LD => load(AccessType::Load, 64, E::rr(PF3, M0F, true, 0x6f, o, 0)), // vmovdqu64
        op::ST => load(AccessType::Store, 64, E::rr(PF3, M0F, true, 0x7f, a, 0)), // vmovdqu64
        op::LDBC_D => load(AccessType::Load, 4, E::rr(P66, M0F38, false, 0x58, o, 0)), // vpbroadcastd
        op::LDBC_Q => load(AccessType::Load, 8, E::rr(P66, M0F38, true, 0x59, o, 0)), // vpbroadcastq
        op::SPLAT_B => vec![Evex(E::rr(P66, M0F38, false, 0x7a, o, r1))], // vpbroadcastb
        op::SPLAT_W => vec![Evex(E::rr(P66, M0F38, false, 0x7b, o, r1))], // vpbroadcastw
        op::SPLAT_D => vec![Evex(E::rr(P66, M0F38, false, 0x7c, o, r1))], // vpbroadcastd
        op::SPLAT_Q => vec![Evex(E::rr(P66, M0F38, true, 0x7c, o, r1))],  // vpbroadcastq
        op::ZERO => vec![Evex(E::rrr(P66, M0F, false, 0xef, o, o, o))],   // vpxord
        op::MOV => vec![Evex(E::rr(P66, M0F, true, 0x6f, o, a))],         // vmovdqa64
        op::ADD_B => binary(M0F, false, 0xfc),                            // vpaddb
        op::ADD_W => binary(M0F, false, 0xfd),                            // vpaddw
        op::ADD_D => binary(M0F, false, 0xfe),                            // vpaddd
        op::ADD_Q => binary(M0F, true, 0xd4),                             // vpaddq
        op::SUB_B => binary(M0F, false, 0xf8),                            // vpsubb
        op::SUB_W => binary(M0F, false, 0xf9),                            // vpsubw
        op::SUB_D => binary(M0F, false, 0xfa),                            // vpsubd
        op::SUB_Q => binary(M0F, true, 0xfb),                             // vpsubq
        op::MULLO_W => binary(M0F, false, 0xd5),                          // vpmullw
        op::MULLO_D => binary(M0F38, false, 0x40),                        // vpmulld
        op::MULLO_Q => binary(M0F38, true, 0x40),                         // vpmullq
        op::MULWU => binary(M0F, true, 0xf4),                             // vpmuludq
        op::MULWS => binary(M0F38, true, 0x28),                           // vpmuldq
        op::MULHI_W => binary(M0F, false, 0xe5),                          // vpmulhw
        op::MULHIU_W => binary(M0F, false, 0xe4),                         // vpmulhuw
        op::MAX_D => binary(M0F38, false, 0x3d),                          // vpmaxsd
        op::MIN_D => binary(M0F38, false, 0x39),                          // vpminsd
        op::MAXU_D => binary(M0F38, false, 0x3f),                         // vpmaxud
        op::MINU_D => binary(M0F38, false, 0x3b),                         // vpminud
        op::MAX_W => binary(M0F, false, 0xee),                            // vpmaxsw
        op::MIN_W => binary(M0F, false, 0xea),                            // vpminsw
        op::MAXU_Q => binary(M0F38, true, 0x3f),                          // vpmaxuq
        op::MINU_Q => binary(M0F38, true, 0x3b),                          // vpminuq
        op::MAXU_W => binary(M0F38, false, 0x3e),                         // vpmaxuw
        op::MINU_W => binary(M0F38, false, 0x3a),                         // vpminuw
        op::AND => binary(M0F, true, 0xdb),                               // vpandq
        op::OR => binary(M0F, true, 0xeb),                                // vporq
        op::XOR => binary(M0F, true, 0xef),                               // vpxorq
        op::ANDN => binary(M0F, true, 0xdf),                              // vpandnq
        op::TERNLOG => vec![Evex(E::rrr(P66, M0F3A, true, 0x25, o, a, b).imm(imm8))], // vpternlogq
        op::SHLI_W => shift(false, 0x71, 6),                              // vpsllw
        op::SHLI_D => shift(false, 0x72, 6),                              // vpslld
        op::SHLI_Q => shift(true, 0x73, 6),                               // vpsllq
        op::SHRI_W => shift(false, 0x71, 2),                              // vpsrlw
        op::SHRI_D => shift(false, 0x72, 2),                              // vpsrld
        op::SHRI_Q => shift(true, 0x73, 2),                               // vpsrlq
        op::SARI_W => shift(false, 0x71, 4),                              // vpsraw
        op::SARI_D => shift(false, 0x72, 4),                              // vpsrad
        op::SARI_Q => shift(true, 0x72, 4),                               // vpsraq
        op::ROLI_D => shift(false, 0x72, 1),                              // vprold
        op::ROLI_Q => shift(true, 0x72, 1),                               // vprolq
        op::SHLV_D => binary(M0F38, false, 0x47),                         // vpsllvd
        op::SHLV_Q => binary(M0F38, true, 0x47),                          // vpsllvq
        op::SHRV_D => binary(M0F38, false, 0x45),                         // vpsrlvd
        op::SHRV_Q => binary(M0F38, true, 0x45),                          // vpsrlvq
        op::ROLV_D => binary(M0F38, false, 0x15),                         // vprolvd
        op::ROLV_Q => binary(M0F38, true, 0x15),                          // vprolvq
        op::CMPEQ_B => compare(E::rrr(P66, M0F, false, 0x74, K1, a, b), false, 0x28), // vpcmpeqb, vpmovm2b
        op::CMPEQ_D => compare(E::rrr(P66, M0F, false, 0x76, K1, a, b), false, 0x38), // vpcmpeqd, vpmovm2d
        op::CMPEQ_Q => compare(E::rrr(P66, M0F38, true, 0x29, K1, a, b), true, 0x38), // vpcmpeqq, vpmovm2q
        op::CMPGT_D => compare(E::rrr(P66, M0F, false, 0x66, K1, a, b), false, 0x38), // vpcmpgtd, vpmovm2d
        op::CMPGT_Q => compare(E::rrr(P66, M0F38, true, 0x37, K1, a, b), true, 0x38), // vpcmpgtq, vpmovm2q
        op::CMPGTU_Q => compare(E::rrr(P66, M0F3A, true, 0x1e, K1, a, b).imm(6), true, 0x38), // vpcmpnleuq, vpmovm2q
        op::DPBUSD => binary(M0F38, false, 0x50), // vpdpbusd
        op::DPWSSD => binary(M0F38, false, 0x52), // vpdpwssd
        op::MADDUBS => binary(M0F38, false, 0x04), // vpmaddubsw
        op::MADDWD => binary(M0F, false, 0xf5),   // vpmaddwd
        op::MADD52LO => binary(M0F38, true, 0xb4), // vpmadd52luq
        op::MADD52HI => binary(M0F38, true, 0xb5), // vpmadd52huq
        op::PACKS_D => binary(M0F, false, 0x6b),  // vpackssdw
        op::PACKS_W => binary(M0F, false, 0x63),  // vpacksswb
        op::PACKUS_W => binary(M0F, false, 0x67), // vpackuswb
        op::PERMD => binary(M0F38, false, 0x36),  // vpermd
        op::PERMQ => binary(M0F38, true, 0x36),   // vpermq
        op::SHUFB => binary(M0F38, false, 0x00),  // vpshufb
        op::PERM2_W => binary(M0F38, true, 0x7d), // vpermt2w (out is table 1)
        op::NARROWUS_D_B => narrow(0x11),         // vpmovusdb
        op::NARROWS_D_B => narrow(0x21),          // vpmovsdb
        op::NARROWS_D_W => narrow(0x23),          // vpmovsdw
        op::NARROW_D_B => narrow(0x31),           // vpmovdb
        op::ZX_B_D => unary(P66, false, 0x31),    // vpmovzxbd
        op::SX_B_D => unary(P66, false, 0x21),    // vpmovsxbd
        op::ZX_W_D => unary(P66, false, 0x33),    // vpmovzxwd
        op::SX_W_D => unary(P66, false, 0x23),    // vpmovsxwd
        op::ZX_D_Q => unary(P66, false, 0x35),    // vpmovzxdq
        op::SX_D_Q => unary(P66, false, 0x25),    // vpmovsxdq
        op::POPCNT_B => unary(P66, false, 0x54),  // vpopcntb
        op::POPCNT_Q => unary(P66, true, 0x55),   // vpopcntq
        op::HSUM_D | op::HSUM_Q => {
            let q = v.info.op == op::HSUM_Q;
            let add = if q { 0xd4 } else { 0xfe }; // vpaddq / vpaddd
            let mut sequence = vec![
                // ymm30 ← high half of in0 + low half of in0
                Evex(E::rr(P66, M0F3A, true, 0x3b, a, ZMM30).imm(1)), // vextracti64x4 ymm30, zmm_a, 1
                Evex(E::rrr(P66, M0F, q, add, ZMM30, ZMM30, a).length(Length::L256)),
                // xmm30 ← high half of ymm30 + low half of ymm30
                Evex(
                    E::rr(P66, M0F3A, false, 0x39, ZMM30, ZMM31)
                        .imm(1)
                        .length(Length::L256),
                ), // vextracti32x4 xmm31, ymm30, 1
                Evex(E::rrr(P66, M0F, q, add, ZMM30, ZMM30, ZMM31).length(Length::L128)),
                // Add the high qword of xmm30 to the low qword
                Evex(
                    E::rr(P66, M0F, false, 0x70, ZMM31, ZMM30)
                        .imm(0x4e)
                        .length(Length::L128),
                ), // vpshufd
                Evex(E::rrr(P66, M0F, q, add, ZMM30, ZMM30, ZMM31).length(Length::L128)),
            ];
            if !q {
                // Add the odd dword of xmm30 to the even dword
                sequence.push(Evex(
                    E::rr(P66, M0F, false, 0x70, ZMM31, ZMM30)
                        .imm(0xb1)
                        .length(Length::L128),
                ));
                sequence.push(Evex(
                    E::rrr(P66, M0F, false, add, ZMM30, ZMM30, ZMM31).length(Length::L128),
                ));
            }
            sequence.push(Evex(
                E::rr(P66, M0F, q, 0x7e, ZMM30, r0 as u8).length(Length::L128),
            )); // vmovd / vmovq
            sequence
        }
        op::EXTRACT_Q | op::EXTRACT_D => {
            let q = v.info.op == op::EXTRACT_Q;
            let per_128 = if q { 2 } else { 4 };
            vec![
                Evex(E::rr(P66, M0F3A, false, 0x39, a, ZMM30).imm(imm8 / per_128)), // vextracti32x4
                Evex(
                    E::rr(P66, M0F3A, q, 0x16, ZMM30, r0 as u8)
                        .imm(imm8 % per_128)
                        .length(Length::L128),
                ), // vpextrq / vpextrd
            ]
        }
        op::INSERT_Q => vec![
            Evex(E::rr(P66, M0F3A, false, 0x39, o, ZMM30).imm(imm8 / 2)), // vextracti32x4
            Evex(
                E::rrr(P66, M0F3A, true, 0x22, ZMM30, ZMM30, r1)
                    .imm(imm8 % 2)
                    .length(Length::L128),
            ), // vpinsrq
            Evex(E::rrr(P66, M0F3A, false, 0x38, o, o, ZMM30).imm(imm8 / 2)), // vinserti32x4
        ],
        op::MOVEMASK_B => vec![
            Evex(E::rr(PF3, M0F38, false, 0x29, K1, a)), // vpmovb2m
            HostInstruction::KmovqToGpr(r0, K1),
        ],
        op::CLMUL => vec![Evex(E::rrr(P66, M0F3A, false, 0x44, o, a, b).imm(imm8))], // vpclmulqdq
        _ => unreachable!("every op has a lowering"),
    };
    Lowering {
        memory,
        instructions,
    }
}

impl<C: ContextObject> JitCompiler<'_, C> {
    fn emit_host_instruction(&mut self, instruction: &HostInstruction) {
        match instruction {
            HostInstruction::Evex(instruction) => {
                let (bytes, len) = instruction.encode();
                for byte in &bytes[..len] {
                    self.emit::<u8>(*byte);
                }
            }
            HostInstruction::KmovqToGpr(destination, mask) => {
                for byte in kmovq_to_gpr(*destination, *mask) {
                    self.emit::<u8>(byte);
                }
            }
        }
        self.emit_random_noop();
    }

    /// Emits the lowering of the vector op `insn`
    pub(super) fn emit_vector_op(&mut self, insn: &ebpf::Insn) {
        let Ok(vector_insn) = VectorInsn::decode(insn) else {
            // Only reachable if the program was not verified. Same error as the interpreter.
            self.emit_ins(X86Instruction::load_immediate(
                REGISTER_SCRATCH,
                self.pc as i64,
            ));
            self.emit_ins(X86Instruction::jump_immediate(
                self.relative_to_anchor(ANCHOR_CALL_UNSUPPORTED_INSTRUCTION, 5),
            ));
            return;
        };
        let lowering = lower(&vector_insn);
        match lowering.memory {
            Some((access_type, len)) => {
                if self.vector_memory_runs.run.is_none() {
                    self.emit_memory_run_start();
                }
                if self.vector_memory_runs.run.is_some() {
                    self.emit_memory_run_access(vector_insn, &lowering);
                } else {
                    self.emit_r1_plus_off(vector_insn.off);
                    self.emit_vector_translation_call(access_type, len, self.pc);
                    for instruction in &lowering.instructions {
                        self.emit_host_instruction(instruction);
                    }
                }
            }
            None => {
                for instruction in &lowering.instructions {
                    self.emit_host_instruction(instruction);
                }
            }
        }
        let pc = self.pc;
        if let Some(run) = self.vector_memory_runs.run.take_if(|run| run.end == pc) {
            self.emit_memory_run_end(run);
        }
    }

    /// REGISTER_SCRATCH = r1 + off
    fn emit_r1_plus_off(&mut self, off: i16) {
        if off == 0 {
            self.emit_ins(X86Instruction::mov(
                OperandSize::S64,
                REGISTER_MAP[1],
                REGISTER_SCRATCH,
            ));
        } else {
            self.emit_vm_address(REGISTER_MAP[1], off as i64, true);
        }
    }

    /// Key which blinds the user provided displacements in [MemoryRuns]
    fn memory_run_key(&self) -> i64 {
        if self.config.sanitize_user_provided_values {
            // Small enough that displacement + key fits in 32 bits
            (self.immediate_value_key as i32 >> 1) as i64
        } else {
            0
        }
    }

    /// Returns which pcs may be reached other than from the previous instruction
    fn find_jump_targets(&self) -> Vec<bool> {
        let instruction_count = self.program.len() / ebpf::INSN_SIZE;
        let mut targets = vec![false; instruction_count];
        let mut mark = |pc: i64| {
            if let Some(target) = usize::try_from(pc).ok().and_then(|pc| targets.get_mut(pc)) {
                *target = true;
            }
        };
        mark(self.executable.get_entrypoint_instruction_offset() as i64);
        for (_, (_, pc)) in self.executable.get_function_registry().iter() {
            mark(pc as i64);
        }
        let mut pc = 0;
        while pc < instruction_count {
            let insn = ebpf::get_insn_unchecked(self.program, pc);
            // Over-approximates: every other jump class instruction marks pc + off + 1, calls also
            // pc + imm + 1
            let class = insn.opc & ebpf::BPF_CLS_MASK;
            let vector_op = insn.opc == ebpf::CALL_IMM && insn.src == VECTOR_OP_SRC;
            if (class == ebpf::BPF_JMP64 || class == ebpf::BPF_JMP32) && !vector_op {
                mark(pc as i64 + insn.off as i64 + 1);
                if insn.opc == ebpf::CALL_IMM {
                    mark((pc as i64).saturating_add(insn.imm).saturating_add(1));
                }
            }
            pc += if insn.opc == ebpf::LD_DW_IMM
                && !self.executable.get_sbpf_version().disable_lddw()
            {
                2
            } else {
                1
            };
        }
        targets
    }

    /// Starts a [MemoryRun] at the memory access at self.pc if there is one
    fn emit_memory_run_start(&mut self) {
        let Some(span_translation) = self.vector_memory_runs.span_translation else {
            return;
        };
        // The tracer clobbers REGISTER_SCRATCH before every instruction
        if cfg!(feature = "tracer") || !self.config.aligned_memory_mapping {
            return;
        }
        if self.vector_memory_runs.jump_targets.is_empty() {
            self.vector_memory_runs.jump_targets = self.find_jump_targets();
        }
        // An instruction meter checkpoint clobbers REGISTER_SCRATCH
        let limit = if self.config.enable_instruction_meter {
            self.last_instruction_meter_validation_pc
                .saturating_add(self.config.instruction_meter_checkpoint_distance)
        } else {
            usize::MAX
        };
        let (mut end, mut accesses, mut stores) = (self.pc, 0, false);
        let (mut min_off, mut max_end) = (i64::MAX, i64::MIN);
        let mut pc = self.pc;
        while (pc + 1) * ebpf::INSN_SIZE <= self.program.len()
            && (pc == self.pc || (pc < limit && !self.vector_memory_runs.jump_targets[pc]))
        {
            let insn = ebpf::get_insn_unchecked(self.program, pc);
            if insn.opc != ebpf::CALL_IMM || insn.src != VECTOR_OP_SRC {
                break;
            }
            let Ok(vector_insn) = VectorInsn::decode(&insn) else {
                break;
            };
            if let Some((access_type, len)) = lower(&vector_insn).memory {
                end = pc;
                accesses += 1;
                stores |= access_type == AccessType::Store;
                min_off = min_off.min(vector_insn.off as i64);
                max_end = max_end.max(vector_insn.off as i64 + len as i64);
            }
            pc += 1;
        }
        let worst_case_length = (end - self.pc + 3)
            * (MAX_MACHINE_CODE_LENGTH_PER_INSTRUCTION
                + MAX_MEMORY_RUN_TRAILER_LENGTH_PER_INSTRUCTION);
        if accesses < 2
            || self.offset_in_text_section + worst_case_length >= self.result.text_section().len()
        {
            return;
        }
        // REGISTER_SCRATCH = host address of the span - key, or 0
        let key = self.memory_run_key();
        self.emit_r1_plus_off(min_off as i16);
        self.emit_ins(X86Instruction::store_immediate(
            OperandSize::S64,
            RSP,
            X86IndirectAccess::OffsetIndexShift(-16, RSP, 0),
            max_end - min_off - 1 + key,
        ));
        self.emit_ins(X86Instruction::call_immediate(
            self.relative_to_offset(span_translation[stores as usize], 5),
        ));
        self.vector_memory_runs.run = Some(MemoryRun {
            start: self.pc,
            end,
            min_off: min_off as i16,
            fallbacks: Vec::new(),
        });
    }

    /// Emits a memory access of the current [MemoryRun]
    fn emit_memory_run_access(&mut self, insn: VectorInsn, lowering: &Lowering) {
        // Like the translation subroutines, never access memory beyond the instruction meter limit
        if self.config.enable_instruction_meter {
            self.emit_ins(X86Instruction::cmp_immediate(
                OperandSize::S64,
                REGISTER_INSTRUCTION_METER,
                self.pc as i64,
                None,
            ));
            self.emit_ins(X86Instruction::conditional_jump_immediate(
                0x86,
                self.relative_to_anchor(ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS, 6),
            ));
        }
        self.emit_ins(X86Instruction::test(
            OperandSize::S64,
            REGISTER_SCRATCH,
            REGISTER_SCRATCH,
            None,
        ));
        let jump = self.offset_in_text_section + 2;
        self.emit_ins(X86Instruction::conditional_jump_immediate(0x84, 0)); // jz fallback
        let run = self.vector_memory_runs.run.as_ref().unwrap();
        let displacement = (insn.off as i64 - run.min_off as i64 + self.memory_run_key()) as i32;
        for instruction in &lowering.instructions {
            let instruction = match instruction {
                HostInstruction::Evex(evex) if matches!(evex.rm, Rm::Mem(..)) => {
                    HostInstruction::Evex(evex.mem(REGISTER_SCRATCH, displacement))
                }
                other => *other,
            };
            self.emit_host_instruction(&instruction);
        }
        let resume = self.offset_in_text_section;
        let run = self.vector_memory_runs.run.as_mut().unwrap();
        run.fallbacks.push(MemoryRunFallback {
            pc: self.pc,
            insn,
            jump,
            resume,
        });
    }

    /// Emits the fallbacks and entries of a [MemoryRun] after its last access
    fn emit_memory_run_end(&mut self, run: MemoryRun) {
        let skip = self.offset_in_text_section + 1;
        self.emit_ins(X86Instruction::jump_immediate(0));
        for fallback in run.fallbacks {
            self.patch_rel32(fallback.jump);
            let lowering = lower(&fallback.insn);
            let (access_type, len) = lowering.memory.unwrap();
            self.emit_r1_plus_off(fallback.insn.off);
            self.emit_vector_translation_call(access_type, len, fallback.pc);
            for instruction in &lowering.instructions {
                self.emit_host_instruction(instruction);
            }
            // Keep using the fallbacks until the end of the run
            self.emit_ins(X86Instruction::alu(
                OperandSize::S32,
                0x31,
                REGISTER_SCRATCH,
                REGISTER_SCRATCH,
                None,
            ));
            self.emit_ins(X86Instruction::jump_immediate(
                self.relative_to_offset(fallback.resume, 5),
            ));
        }
        for pc in run.start + 1..=run.end {
            let code = self.result.pc_section()[pc] as usize;
            self.result.pc_section_mut()[pc] = self.offset_in_text_section as u32;
            self.emit_ins(X86Instruction::alu(
                OperandSize::S32,
                0x31,
                REGISTER_SCRATCH,
                REGISTER_SCRATCH,
                None,
            ));
            self.emit_ins(X86Instruction::jump_immediate(
                self.relative_to_offset(code, 5),
            ));
        }
        self.patch_rel32(skip);
    }

    /// Points the rel32 at `location` to the current offset
    fn patch_rel32(&mut self, location: usize) {
        let relative = self.offset_in_text_section as i32 - (location + 4) as i32;
        self.result.text_section_mut()[location..location + 4]
            .copy_from_slice(&relative.to_le_bytes());
    }

    /// Stores (`to_vm == true`) or loads the used vector registers to / from [EbpfVm::vector_registers]
    pub(super) fn emit_vector_registers_spill(&mut self, to_vm: bool) {
        let base = self.slot_in_vm(RuntimeEnvironmentSlot::VectorRegisters);
        for reg in 0..VECTOR_REGISTERS as u8 {
            if self.vector_registers_used & (1 << reg) != 0 {
                let instruction = EvexInstruction::rr(
                    PF3,
                    M0F,
                    true,
                    if to_vm { 0x7f } else { 0x6f },
                    reg,
                    0,
                ) // vmovdqu64
                .mem(REGISTER_PTR_TO_VM, base + reg as i32 * VECTOR_SIZE as i32);
                self.emit_host_instruction(&HostInstruction::Evex(instruction));
            }
        }
        if to_vm {
            for byte in VZEROUPPER {
                self.emit::<u8>(byte);
            }
        }
    }

    /// Computes `reg + constant` (wrapping, like the scalar path) into REGISTER_SCRATCH
    fn emit_vm_address(&mut self, reg: X86Register, constant: i64, user_provided: bool) {
        if user_provided && self.should_sanitize_constant(constant) {
            self.emit_sanitized_load_immediate(REGISTER_SCRATCH, constant);
        } else {
            self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, constant));
        }
        self.emit_ins(X86Instruction::alu(
            OperandSize::S64,
            0x01,
            reg,
            REGISTER_SCRATCH,
            None,
        ));
    }

    /// Emits a call to the translation subroutine, the vm address must be in REGISTER_SCRATCH
    ///
    /// Afterwards REGISTER_SCRATCH contains the host address. All other registers are preserved.
    fn emit_vector_translation_call(&mut self, access_type: AccessType, len: u64, pc: usize) {
        // Store the pc in the stack slot the slow path expects (see emit_address_translation)
        self.emit_ins(X86Instruction::store_immediate(
            OperandSize::S64,
            RSP,
            X86IndirectAccess::OffsetIndexShift(-16, RSP, 0),
            pc as i64,
        ));
        self.emit_ins(X86Instruction::call_immediate(
            self.relative_to_anchor(translation_anchor(access_type, len), 5),
        ));
    }

    /// Replaces [JitCompiler::emit_address_translation] in programs using vector ops
    pub(super) fn emit_vector_address_translation(
        &mut self,
        dst: Option<X86Register>,
        vm_addr: Value,
        len: u64,
        value: Option<Value>,
    ) {
        match vm_addr {
            Value::RegisterPlusConstant64(reg, constant, user_provided) => {
                self.emit_vm_address(reg, constant, user_provided);
            }
            _ => {
                #[cfg(debug_assertions)]
                unreachable!();
            }
        }
        self.emit_vector_translation_call(
            if dst.is_some() {
                AccessType::Load
            } else {
                AccessType::Store
            },
            len,
            self.pc,
        );
        let size = match len {
            1 => OperandSize::S8,
            2 => OperandSize::S16,
            4 => OperandSize::S32,
            _ => OperandSize::S64,
        };
        let host_address = X86IndirectAccess::Offset(0);
        match (dst, value) {
            (Some(dst), _) => self.emit_ins(X86Instruction::load(
                size,
                REGISTER_SCRATCH,
                dst,
                host_address,
            )),
            (None, Some(Value::Register(src))) => self.emit_ins(X86Instruction::store(
                size,
                src,
                REGISTER_SCRATCH,
                host_address,
            )),
            (None, Some(Value::Constant64(constant, user_provided))) => {
                if user_provided && self.should_sanitize_constant(constant) {
                    // Borrow a guest register, the red zone below RSP is free here
                    let temporary = REGISTER_MAP[0];
                    let stack_slot = X86IndirectAccess::OffsetIndexShift(-8, RSP, 0);
                    self.emit_ins(X86Instruction::store(
                        OperandSize::S64,
                        temporary,
                        RSP,
                        stack_slot,
                    ));
                    self.emit_sanitized_load_immediate(temporary, constant);
                    self.emit_ins(X86Instruction::store(
                        size,
                        temporary,
                        REGISTER_SCRATCH,
                        host_address,
                    ));
                    self.emit_ins(X86Instruction::load(
                        OperandSize::S64,
                        RSP,
                        temporary,
                        stack_slot,
                    ));
                } else {
                    self.emit_ins(X86Instruction::store_immediate(
                        size,
                        REGISTER_SCRATCH,
                        host_address,
                        constant,
                    ));
                }
            }
            _ => {
                #[cfg(debug_assertions)]
                unreachable!();
            }
        }
    }

    fn relative_to_offset(&self, target_offset: usize, instruction_length: usize) -> i32 {
        target_offset as i32 - (self.offset_in_text_section + instruction_length) as i32
    }

    /// Emits the entry prologue and the translation subroutines (single accesses and spans)
    pub(super) fn emit_vector_subroutines(&mut self) {
        // Entry prologue: load the used vector registers, then jump to the entrypoint
        self.set_anchor(ANCHOR_VECTOR_PROLOGUE);
        self.emit_vector_registers_spill(false);
        let jump_offset =
            self.relative_to_target_pc(self.executable.get_entrypoint_instruction_offset(), 5);
        self.emit_ins(X86Instruction::jump_immediate(jump_offset));

        // Translation subroutines
        // Input: vm address in REGISTER_SCRATCH, guest pc in X86IndirectAccess::OffsetIndexShift(-8, RSP, 0)
        // Output: host address in REGISTER_SCRATCH, or throws the error of the access
        let layout = jit_region_layout();
        let virtual_address_bits = self.config.aligned_region_size.trailing_zeros() as i64;
        for access_type in [AccessType::Load, AccessType::Store] {
            for len in TRANSLATION_LENGTHS {
                // Slow path, emitted first so that the fast path can branch backwards to it
                let temporaries = [(RAX, -16), (RDX, -24)];
                let slow_path_restoring_temporaries = self.offset_in_text_section;
                if layout.is_some() {
                    for (reg, offset) in temporaries {
                        self.emit_ins(X86Instruction::load(
                            OperandSize::S64,
                            RSP,
                            reg,
                            X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
                        ));
                    }
                }
                if layout.is_none() {
                    self.set_anchor(translation_anchor(access_type, len));
                }
                if self.config.enable_instruction_meter {
                    self.emit_ins(X86Instruction::alu(
                        OperandSize::S64,
                        0x3B,
                        REGISTER_INSTRUCTION_METER,
                        RSP,
                        Some(X86IndirectAccess::OffsetIndexShift(-8, RSP, 0)),
                    )); // cmp REGISTER_INSTRUCTION_METER, pc (on rsp);
                    self.emit_ins(X86Instruction::conditional_jump_immediate(
                        0x86,
                        self.relative_to_anchor(ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS, 6),
                    ));
                }
                // skip over the pc slot pushed by the caller, we'll pop it before returning
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S64,
                    0x81,
                    5,
                    RSP,
                    8,
                    None,
                )); // RSP -= 8
                self.emit_rust_call(
                    Value::Constant64(translate_slow_path as *const u8 as i64, false),
                    &[
                        Argument {
                            index: 2,
                            value: Value::Register(REGISTER_SCRATCH),
                        }, // Specify first as the src register could be overwritten by other arguments
                        Argument {
                            index: 3,
                            value: Value::Constant64(len as i64, false),
                        },
                        Argument {
                            index: 4,
                            value: Value::Constant64(
                                (access_type == AccessType::Store) as i64,
                                false,
                            ),
                        },
                        Argument {
                            index: 1,
                            value: Value::RegisterIndirect(
                                REGISTER_PTR_TO_VM,
                                self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping),
                                false,
                            ),
                        },
                        Argument {
                            index: 0,
                            value: Value::RegisterPlusConstant32(
                                REGISTER_PTR_TO_VM,
                                self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult),
                                false,
                            ),
                        },
                    ],
                    None,
                );
                // Throw error if the result indicates one
                self.emit_check_result_is_ok();
                self.emit_ins(X86Instruction::pop(REGISTER_SCRATCH)); // REGISTER_SCRATCH = pc
                self.emit_ins(X86Instruction::conditional_jump_immediate(
                    0x85,
                    self.relative_to_anchor(ANCHOR_THROW_EXCEPTION, 6),
                ));
                // unwrap() the host address into REGISTER_SCRATCH
                self.emit_ins(X86Instruction::load(
                    OperandSize::S64,
                    REGISTER_PTR_TO_VM,
                    REGISTER_SCRATCH,
                    X86IndirectAccess::Offset(
                        self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult)
                            + std::mem::size_of::<u64>() as i32,
                    ),
                ));
                self.emit_ins(X86Instruction::return_near());

                let Some(layout) = layout else {
                    continue;
                };
                // Fast path, mirrors MemoryMapping::map() for AlignedMemoryMapping and non-gapped regions
                self.set_anchor(translation_anchor(access_type, len));
                // Like the slow path, never access memory beyond the instruction meter limit
                if self.config.enable_instruction_meter {
                    self.emit_ins(X86Instruction::alu(
                        OperandSize::S64,
                        0x3B,
                        REGISTER_INSTRUCTION_METER,
                        RSP,
                        Some(X86IndirectAccess::OffsetIndexShift(-8, RSP, 0)),
                    )); // cmp REGISTER_INSTRUCTION_METER, pc (on rsp);
                    self.emit_ins(X86Instruction::conditional_jump_immediate(
                        0x86,
                        self.relative_to_anchor(ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS, 6),
                    ));
                }
                for (reg, offset) in temporaries {
                    self.emit_ins(X86Instruction::store(
                        OperandSize::S64,
                        reg,
                        RSP,
                        X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
                    ));
                }
                let table = std::mem::offset_of!(MemoryMapping, jit_region_table) as i32;
                let slow_path = |jit: &mut Self, condition: u8| {
                    let offset = jit.relative_to_offset(slow_path_restoring_temporaries, 6);
                    jit.emit_ins(X86Instruction::conditional_jump_immediate(
                        condition, offset,
                    ));
                };
                // RAX = &MemoryMapping
                self.emit_ins(X86Instruction::load(
                    OperandSize::S64,
                    REGISTER_PTR_TO_VM,
                    RAX,
                    X86IndirectAccess::Offset(
                        self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping),
                    ),
                ));
                // The region size must be the one this code was compiled for
                self.emit_ins(X86Instruction::cmp_immediate(
                    OperandSize::S64,
                    RAX,
                    virtual_address_bits,
                    Some(X86IndirectAccess::Offset(
                        table + std::mem::offset_of!(JitRegionTable, virtual_address_bits) as i32,
                    )),
                ));
                slow_path(self, 0x85); // jne
                                       // RDX = region index = vm_addr >> virtual_address_bits, must be < table.len
                self.emit_ins(X86Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, RDX));
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S64,
                    0xc1,
                    5,
                    RDX,
                    virtual_address_bits,
                    None,
                ));
                self.emit_ins(X86Instruction::cmp(
                    OperandSize::S64,
                    RDX,
                    RAX,
                    Some(X86IndirectAccess::Offset(
                        table + std::mem::offset_of!(JitRegionTable, len) as i32,
                    )),
                ));
                slow_path(self, 0x86); // jbe: table.len <= index
                                       // RDX = &regions[index]
                self.emit_imul_rdx_immediate(layout.size as i32);
                self.emit_ins(X86Instruction::alu(
                    OperandSize::S64,
                    0x03,
                    RDX,
                    RAX,
                    Some(X86IndirectAccess::Offset(
                        table + std::mem::offset_of!(JitRegionTable, regions) as i32,
                    )),
                ));
                // Gapped regions take the slow path
                self.emit_ins(X86Instruction::cmp_immediate(
                    OperandSize::S8,
                    RDX,
                    63,
                    Some(X86IndirectAccess::Offset(layout.vm_gap_shift as i32)),
                ));
                slow_path(self, 0x85); // jne
                if access_type == AccessType::Store {
                    // Stores need HostBuffer::Mutable
                    self.emit_ins(X86Instruction::cmp_immediate(
                        OperandSize::S64,
                        RDX,
                        1,
                        Some(X86IndirectAccess::Offset(layout.host_tag as i32)),
                    ));
                    slow_path(self, 0x85); // jne
                }
                // RAX = begin_offset = vm_addr - region.vm_addr, must not underflow
                self.emit_ins(X86Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, RAX));
                self.emit_ins(X86Instruction::alu(
                    OperandSize::S64,
                    0x2b,
                    RAX,
                    RDX,
                    Some(X86IndirectAccess::Offset(layout.vm_addr as i32)),
                ));
                slow_path(self, 0x82); // jb
                                       // RAX = end_offset = begin_offset + len, must not overflow and be <= host length
                self.emit_ins(X86Instruction::alu_immediate(
                    OperandSize::S64,
                    0x81,
                    0,
                    RAX,
                    len as i64,
                    None,
                ));
                slow_path(self, 0x82); // jc
                self.emit_ins(X86Instruction::alu(
                    OperandSize::S64,
                    0x3b,
                    RAX,
                    RDX,
                    Some(X86IndirectAccess::Offset(layout.host_len as i32)),
                ));
                slow_path(self, 0x87); // ja
                                       // REGISTER_SCRATCH = host_ptr + end_offset - len
                self.emit_ins(X86Instruction::alu(
                    OperandSize::S64,
                    0x03,
                    RAX,
                    RDX,
                    Some(X86IndirectAccess::Offset(layout.host_ptr as i32)),
                ));
                self.emit_ins(X86Instruction::lea(
                    OperandSize::S64,
                    RAX,
                    REGISTER_SCRATCH,
                    Some(X86IndirectAccess::Offset(-(len as i32))),
                ));
                for (reg, offset) in temporaries {
                    self.emit_ins(X86Instruction::load(
                        OperandSize::S64,
                        RSP,
                        reg,
                        X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
                    ));
                }
                self.emit_ins(X86Instruction::return_near());
            }
        }

        // Span translation subroutines, see MemoryRuns
        // Input: first vm address of the span in REGISTER_SCRATCH, span length - 1 + key in X86IndirectAccess::OffsetIndexShift(-8, RSP, 0)
        // Output: host address of the span - key in REGISTER_SCRATCH, 0 if it is not in one plain region
        let Some(layout) = layout else {
            return;
        };
        let key = self.memory_run_key();
        let mut span_translation = [0; 2];
        for (access_type, offset) in [AccessType::Load, AccessType::Store]
            .iter()
            .zip(&mut span_translation)
        {
            let temporaries = [(RAX, -16), (RDX, -24)];
            let restore_temporaries = |jit: &mut Self| {
                for (reg, offset) in temporaries {
                    jit.emit_ins(X86Instruction::load(
                        OperandSize::S64,
                        RSP,
                        reg,
                        X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
                    ));
                }
            };
            let span_length_minus_one = X86IndirectAccess::OffsetIndexShift(-8, RSP, 0);
            // Not plain, emitted first so that the checks can branch backwards to it
            let not_plain = self.offset_in_text_section;
            restore_temporaries(self);
            self.emit_ins(X86Instruction::alu(
                OperandSize::S32,
                0x31,
                REGISTER_SCRATCH,
                REGISTER_SCRATCH,
                None,
            ));
            self.emit_ins(X86Instruction::return_near());
            let not_plain_if = |jit: &mut Self, condition: u8| {
                let offset = jit.relative_to_offset(not_plain, 6);
                jit.emit_ins(X86Instruction::conditional_jump_immediate(
                    condition, offset,
                ));
            };

            *offset = self.offset_in_text_section;
            for (reg, offset) in temporaries {
                self.emit_ins(X86Instruction::store(
                    OperandSize::S64,
                    reg,
                    RSP,
                    X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
                ));
            }
            // RAX = last vm address of the span, must not overflow and have the region index of the first
            self.emit_ins(X86Instruction::load(
                OperandSize::S64,
                RSP,
                RAX,
                span_length_minus_one,
            ));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0x81,
                0,
                RAX,
                -key,
                None,
            ));
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x01,
                REGISTER_SCRATCH,
                RAX,
                None,
            ));
            not_plain_if(self, 0x82); // jc
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x31,
                REGISTER_SCRATCH,
                RAX,
                None,
            ));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0xc1,
                5,
                RAX,
                virtual_address_bits,
                None,
            ));
            not_plain_if(self, 0x85); // jne

            // The checks of the translation fast path for the first vm address and len = span length
            let table = std::mem::offset_of!(MemoryMapping, jit_region_table) as i32;
            self.emit_ins(X86Instruction::load(
                OperandSize::S64,
                REGISTER_PTR_TO_VM,
                RAX,
                X86IndirectAccess::Offset(self.slot_in_vm(RuntimeEnvironmentSlot::MemoryMapping)),
            ));
            self.emit_ins(X86Instruction::cmp_immediate(
                OperandSize::S64,
                RAX,
                virtual_address_bits,
                Some(X86IndirectAccess::Offset(
                    table + std::mem::offset_of!(JitRegionTable, virtual_address_bits) as i32,
                )),
            ));
            not_plain_if(self, 0x85); // jne
            self.emit_ins(X86Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, RDX));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0xc1,
                5,
                RDX,
                virtual_address_bits,
                None,
            ));
            self.emit_ins(X86Instruction::cmp(
                OperandSize::S64,
                RDX,
                RAX,
                Some(X86IndirectAccess::Offset(
                    table + std::mem::offset_of!(JitRegionTable, len) as i32,
                )),
            ));
            not_plain_if(self, 0x86); // jbe
            self.emit_imul_rdx_immediate(layout.size as i32);
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x03,
                RDX,
                RAX,
                Some(X86IndirectAccess::Offset(
                    table + std::mem::offset_of!(JitRegionTable, regions) as i32,
                )),
            ));
            self.emit_ins(X86Instruction::cmp_immediate(
                OperandSize::S8,
                RDX,
                63,
                Some(X86IndirectAccess::Offset(layout.vm_gap_shift as i32)),
            ));
            not_plain_if(self, 0x85); // jne
            if *access_type == AccessType::Store {
                self.emit_ins(X86Instruction::cmp_immediate(
                    OperandSize::S64,
                    RDX,
                    1,
                    Some(X86IndirectAccess::Offset(layout.host_tag as i32)),
                ));
                not_plain_if(self, 0x85); // jne
            }
            // RAX = begin_offset, must not underflow
            self.emit_ins(X86Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, RAX));
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x2b,
                RAX,
                RDX,
                Some(X86IndirectAccess::Offset(layout.vm_addr as i32)),
            ));
            not_plain_if(self, 0x82); // jb

            // begin_offset + span length - 1 must be < host length (no overflow, checked above)
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x03,
                RAX,
                RSP,
                Some(span_length_minus_one),
            ));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0x81,
                0,
                RAX,
                -key,
                None,
            ));
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x3b,
                RAX,
                RDX,
                Some(X86IndirectAccess::Offset(layout.host_len as i32)),
            ));
            not_plain_if(self, 0x83); // jae

            // REGISTER_SCRATCH = host_ptr + begin_offset - key
            self.emit_ins(X86Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, RAX));
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x2b,
                RAX,
                RDX,
                Some(X86IndirectAccess::Offset(layout.vm_addr as i32)),
            ));
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x03,
                RAX,
                RDX,
                Some(X86IndirectAccess::Offset(layout.host_ptr as i32)),
            ));
            self.emit_ins(X86Instruction::lea(
                OperandSize::S64,
                RAX,
                REGISTER_SCRATCH,
                Some(X86IndirectAccess::Offset(-key as i32)),
            ));
            restore_temporaries(self);
            self.emit_ins(X86Instruction::return_near());
        }
        self.vector_memory_runs.span_translation = Some(span_translation);
    }

    /// `imul rdx, rdx, immediate`
    fn emit_imul_rdx_immediate(&mut self, immediate: i32) {
        self.emit::<u8>(0x48); // REX.W
        self.emit::<u8>(0x69);
        self.emit::<u8>(0xd2); // ModRM: rdx, rdx
        self.emit::<i32>(immediate);
        self.emit_random_noop();
    }
}

#[cfg(test)]
thread_local! {
    /// Number of calls of the slow path on this thread
    static SLOW_PATH_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aligned_memory::AlignedMemory,
        memory_region::MemoryRegion,
        program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
        vector::{encode, encode_with_offset, Kind, OPS},
        vm::{CallFrame, ExecutionMode},
    };
    use std::{io::Write, process::Command, sync::Arc};

    fn decode(word: u64) -> VectorInsn {
        VectorInsn::decode(&ebpf::get_insn(&word.to_le_bytes(), 0)).unwrap()
    }

    /// Expected Intel syntax disassembly of `lower(v)` as printed by LLVM
    fn expected_disassembly(v: &VectorInsn) -> Vec<String> {
        let (o, a, b, i) = (v.out, v.in0, v.in1, v.imm8);
        let binary = |mnemonic: &str| vec![format!("{mnemonic} zmm{o}, zmm{a}, zmm{b}")];
        let unary = |mnemonic: &str, source: &str| vec![format!("{mnemonic} zmm{o}, {source}{a}")];
        let shift = |mnemonic: &str| vec![format!("{mnemonic} zmm{o}, zmm{a}, {i}")];
        let compare = |compare: &str, movm: &str| {
            vec![
                format!("{compare} k1, zmm{a}, zmm{b}"),
                format!("{movm} zmm{o}, k1"),
            ]
        };
        let narrow = |mnemonic: &str, destination: &str| {
            vec![format!("{mnemonic} {destination}{o}, zmm{a}")]
        };
        match v.info.op {
            op::LD => vec![format!("vmovdqu64 zmm{o}, zmmword ptr [r11]")],
            op::ST => vec![format!("vmovdqu64 zmmword ptr [r11], zmm{a}")],
            op::LDBC_D => vec![format!("vpbroadcastd zmm{o}, dword ptr [r11]")],
            op::LDBC_Q => vec![format!("vpbroadcastq zmm{o}, qword ptr [r11]")],
            op::SPLAT_B => vec![format!("vpbroadcastb zmm{o}, esi")],
            op::SPLAT_W => vec![format!("vpbroadcastw zmm{o}, esi")],
            op::SPLAT_D => vec![format!("vpbroadcastd zmm{o}, esi")],
            op::SPLAT_Q => vec![format!("vpbroadcastq zmm{o}, rsi")],
            op::ZERO => vec![format!("vpxord zmm{o}, zmm{o}, zmm{o}")],
            op::MOV => vec![format!("vmovdqa64 zmm{o}, zmm{a}")],
            op::ADD_B => binary("vpaddb"),
            op::ADD_W => binary("vpaddw"),
            op::ADD_D => binary("vpaddd"),
            op::ADD_Q => binary("vpaddq"),
            op::SUB_B => binary("vpsubb"),
            op::SUB_W => binary("vpsubw"),
            op::SUB_D => binary("vpsubd"),
            op::SUB_Q => binary("vpsubq"),
            op::MULLO_W => binary("vpmullw"),
            op::MULLO_D => binary("vpmulld"),
            op::MULLO_Q => binary("vpmullq"),
            op::MULWU => binary("vpmuludq"),
            op::MULWS => binary("vpmuldq"),
            op::MULHI_W => binary("vpmulhw"),
            op::MULHIU_W => binary("vpmulhuw"),
            op::MAX_D => binary("vpmaxsd"),
            op::MIN_D => binary("vpminsd"),
            op::MAXU_D => binary("vpmaxud"),
            op::MINU_D => binary("vpminud"),
            op::MAX_W => binary("vpmaxsw"),
            op::MIN_W => binary("vpminsw"),
            op::MAXU_Q => binary("vpmaxuq"),
            op::MINU_Q => binary("vpminuq"),
            op::MAXU_W => binary("vpmaxuw"),
            op::MINU_W => binary("vpminuw"),
            op::AND => binary("vpandq"),
            op::OR => binary("vporq"),
            op::XOR => binary("vpxorq"),
            op::ANDN => binary("vpandnq"),
            op::TERNLOG => vec![format!("vpternlogq zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::SHLI_W => shift("vpsllw"),
            op::SHLI_D => shift("vpslld"),
            op::SHLI_Q => shift("vpsllq"),
            op::SHRI_W => shift("vpsrlw"),
            op::SHRI_D => shift("vpsrld"),
            op::SHRI_Q => shift("vpsrlq"),
            op::SARI_W => shift("vpsraw"),
            op::SARI_D => shift("vpsrad"),
            op::SARI_Q => shift("vpsraq"),
            op::ROLI_D => shift("vprold"),
            op::ROLI_Q => shift("vprolq"),
            op::SHLV_D => binary("vpsllvd"),
            op::SHLV_Q => binary("vpsllvq"),
            op::SHRV_D => binary("vpsrlvd"),
            op::SHRV_Q => binary("vpsrlvq"),
            op::ROLV_D => binary("vprolvd"),
            op::ROLV_Q => binary("vprolvq"),
            op::CMPEQ_B => compare("vpcmpeqb", "vpmovm2b"),
            op::CMPEQ_D => compare("vpcmpeqd", "vpmovm2d"),
            op::CMPEQ_Q => compare("vpcmpeqq", "vpmovm2q"),
            op::CMPGT_D => compare("vpcmpgtd", "vpmovm2d"),
            op::CMPGT_Q => compare("vpcmpgtq", "vpmovm2q"),
            op::CMPGTU_Q => compare("vpcmpnleuq", "vpmovm2q"),
            op::DPBUSD => binary("vpdpbusd"),
            op::DPWSSD => binary("vpdpwssd"),
            op::MADDUBS => binary("vpmaddubsw"),
            op::MADDWD => binary("vpmaddwd"),
            op::MADD52LO => binary("vpmadd52luq"),
            op::MADD52HI => binary("vpmadd52huq"),
            op::PACKS_D => binary("vpackssdw"),
            op::PACKS_W => binary("vpacksswb"),
            op::PACKUS_W => binary("vpackuswb"),
            op::PERMD => binary("vpermd"),
            op::PERMQ => binary("vpermq"),
            op::SHUFB => binary("vpshufb"),
            op::PERM2_W => binary("vpermt2w"),
            op::NARROWUS_D_B => narrow("vpmovusdb", "xmm"),
            op::NARROWS_D_B => narrow("vpmovsdb", "xmm"),
            op::NARROWS_D_W => narrow("vpmovsdw", "ymm"),
            op::NARROW_D_B => narrow("vpmovdb", "xmm"),
            op::ZX_B_D => unary("vpmovzxbd", "xmm"),
            op::SX_B_D => unary("vpmovsxbd", "xmm"),
            op::ZX_W_D => unary("vpmovzxwd", "ymm"),
            op::SX_W_D => unary("vpmovsxwd", "ymm"),
            op::ZX_D_Q => unary("vpmovzxdq", "ymm"),
            op::SX_D_Q => unary("vpmovsxdq", "ymm"),
            op::POPCNT_B => unary("vpopcntb", "zmm"),
            op::POPCNT_Q => unary("vpopcntq", "zmm"),
            op::HSUM_D => vec![
                format!("vextracti64x4 ymm30, zmm{a}, 1"),
                format!("vpaddd ymm30, ymm30, ymm{a}"),
                "vextracti32x4 xmm31, ymm30, 1".into(),
                "vpaddd xmm30, xmm30, xmm31".into(),
                "vpshufd xmm31, xmm30, 78".into(),
                "vpaddd xmm30, xmm30, xmm31".into(),
                "vpshufd xmm31, xmm30, 177".into(),
                "vpaddd xmm30, xmm30, xmm31".into(),
                "vmovd eax, xmm30".into(),
            ],
            op::HSUM_Q => vec![
                format!("vextracti64x4 ymm30, zmm{a}, 1"),
                format!("vpaddq ymm30, ymm30, ymm{a}"),
                "vextracti32x4 xmm31, ymm30, 1".into(),
                "vpaddq xmm30, xmm30, xmm31".into(),
                "vpshufd xmm31, xmm30, 78".into(),
                "vpaddq xmm30, xmm30, xmm31".into(),
                "vmovq rax, xmm30".into(),
            ],
            op::EXTRACT_Q => vec![
                format!("vextracti32x4 xmm30, zmm{a}, {}", i / 2),
                format!("vpextrq rax, xmm30, {}", i % 2),
            ],
            op::EXTRACT_D => vec![
                format!("vextracti32x4 xmm30, zmm{a}, {}", i / 4),
                format!("vpextrd eax, xmm30, {}", i % 4),
            ],
            op::INSERT_Q => vec![
                format!("vextracti32x4 xmm30, zmm{o}, {}", i / 2),
                format!("vpinsrq xmm30, xmm30, rsi, {}", i % 2),
                format!("vinserti32x4 zmm{o}, zmm{o}, xmm30, {}", i / 2),
            ],
            op::MOVEMASK_B => vec![format!("vpmovb2m k1, zmm{a}"), "kmovq rax, k1".into()],
            op::CLMUL => vec![format!("vpclmulqdq zmm{o}, zmm{a}, zmm{b}, {i}")],
            _ => unreachable!(),
        }
    }

    /// Disassembles `code` with LLVM into one normalized line per instruction
    fn llvm_disassemble(code: &[Vec<u8>]) -> Option<Vec<String>> {
        let llvm_mc = ["llvm-mc-19", "llvm-mc"]
            .iter()
            .find(|tool| Command::new(tool).arg("--version").output().is_ok())?;
        let mut child = Command::new(llvm_mc)
            .args(["--disassemble", "-triple=x86_64", "--output-asm-variant=1", "-mattr=+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512vnni,+avx512ifma,+avx512bitalg,+avx512vpopcntdq,+vpclmulqdq"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        let input: String = code
            .iter()
            .map(|bytes| {
                bytes
                    .iter()
                    .map(|b| format!("{b:#04x} "))
                    .collect::<String>()
                    + "\n"
            })
            .collect();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.stderr.is_empty(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .map(|line| {
                    line.split('#')
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .filter(|line| !line.is_empty() && !line.starts_with('.'))
                .collect(),
        )
    }

    /// Every EVEX (and VEX) encoding the JIT emits, checked against LLVM's disassembler
    #[test]
    fn test_encodings_against_llvm() {
        let mut code = vec![];
        let mut expected = vec![];
        for info in OPS {
            let kind = info.kind;
            let imm8s: Vec<u8> = (0..=255).filter(|i| info.imm8_is_valid(*i)).collect();
            for (k, (o, a, b)) in [
                (3, 4, 5),
                (29, 16, 8),
                (0, 15, 17),
                (24, 9, 1),
                (8, 23, 28),
                (7, 7, 7),
            ]
            .iter()
            .copied()
            .enumerate()
            {
                let (o, a, b) = (
                    o * kind.uses_out() as u8,
                    a * kind.uses_in0() as u8,
                    b * kind.uses_in1() as u8,
                );
                let imm8 = imm8s[(k * 37) % imm8s.len()];
                let v = decode(encode(info.op, imm8, o, a, b));
                for instruction in lower(&v).instructions {
                    code.push(instruction.encode());
                }
                expected.extend(expected_disassembly(&v));
            }
        }
        // Spilling, kmovq with other registers and vzeroupper
        for (reg, disp) in [(0u8, -0x1234), (17, 0x7fff_ffc0), (29, 64)] {
            for (opcode, text) in [
                (
                    0x7f,
                    format!(
                        "vmovdqu64 zmmword ptr [rdi {} {}], zmm{reg}",
                        if disp < 0 { '-' } else { '+' },
                        (disp as i64).abs()
                    ),
                ),
                (
                    0x6f,
                    format!(
                        "vmovdqu64 zmm{reg}, zmmword ptr [rdi {} {}]",
                        if disp < 0 { '-' } else { '+' },
                        (disp as i64).abs()
                    ),
                ),
            ] {
                code.push(
                    HostInstruction::Evex(
                        EvexInstruction::rr(PF3, M0F, true, opcode, reg, 0).mem(RDI, disp),
                    )
                    .encode(),
                );
                expected.push(text);
            }
        }
        for (reg, mask, text) in [(R15, 7, "kmovq r15, k7"), (RDX, 1, "kmovq rdx, k1")] {
            code.push(HostInstruction::KmovqToGpr(reg, mask).encode());
            expected.push(text.to_string());
        }
        code.push(VZEROUPPER.to_vec());
        expected.push("vzeroupper".to_string());
        // Memory operands with special base registers
        for (base, text) in [(RSP, "rsp"), (R12, "r12"), (RBP, "rbp"), (R13, "r13")] {
            code.push(
                HostInstruction::Evex(EvexInstruction::rr(PF3, M0F, true, 0x6f, 1, 0).mem(base, 0))
                    .encode(),
            );
            expected.push(format!("vmovdqu64 zmm1, zmmword ptr [{text}]"));
        }
        let Some(disassembly) = llvm_disassemble(&code) else {
            eprintln!("skipping, llvm-mc not found");
            return;
        };
        assert_eq!(disassembly.len(), expected.len());
        for (actual, expected) in disassembly.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }

    /// Every op has a lowering and only the memory ops translate an address
    #[test]
    fn test_lower_all_ops() {
        for info in OPS {
            let kind = info.kind;
            let imm8 = (0..=255).find(|i| info.imm8_is_valid(*i)).unwrap();
            let lowering = lower(&decode(encode(
                info.op,
                imm8,
                kind.uses_out() as u8,
                kind.uses_in0() as u8,
                kind.uses_in1() as u8,
            )));
            assert!(!lowering.instructions.is_empty());
            assert_eq!(
                lowering.memory.is_some(),
                matches!(kind, Kind::Ls | Kind::St) && info.op <= op::LDBC_Q,
                "{}",
                info.name
            );
        }
    }

    struct Context {
        remaining: u64,
        memory_mapping: MemoryMapping,
    }

    impl ContextObject for Context {
        fn consume(&mut self, amount: u64) {
            self.remaining = self.remaining.saturating_sub(amount);
        }
        fn get_remaining(&self) -> u64 {
            self.remaining
        }
        fn active_mapping_ptr(&mut self) -> std::ptr::NonNull<MemoryMapping> {
            std::ptr::NonNull::from_mut(&mut self.memory_mapping)
        }
    }

    /// In-bounds accesses of an aligned mapping never take the slow path
    #[test]
    fn test_inline_translation_fast_path() {
        if missing_host_feature().is_some() {
            return;
        }
        let ldxdw = |dst: u8, src: u8, off: i16| {
            ebpf::LD_DW_REG as u64 | ((dst | src << 4) as u64) << 8 | (off as u16 as u64) << 16
        };
        let stb = |dst: u8, off: i16| {
            ebpf::ST_B_IMM as u64 | (dst as u64) << 8 | (off as u16 as u64) << 16 | 0x7f << 32
        };
        let program = |offset: i16| -> Vec<u64> {
            vec![
                encode(op::LD, 0, 1, 0, 0),
                encode(op::ST, 0, 0, 1, 0),
                encode_with_offset(op::LDBC_D, 0, 2, 0, 0, 60),
                encode(op::LDBC_Q, 0, 3, 0, 0),
                ldxdw(0, 1, offset),
                stb(1, offset),
                ldxdw(4, 10, -8),
                ebpf::EXIT as u64,
            ]
        };
        for aligned_memory_mapping in [false, true] {
            for (offset, expected_slow_path_calls) in [(0, 0), (56, 0), (57, 1)] {
                let config = Config {
                    enable_vector_ops: true,
                    aligned_memory_mapping,
                    ..Config::default()
                };
                let bytes: Vec<u8> = program(offset)
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                let loader = Arc::new(BuiltinProgram::new_loader(config.clone()));
                let executable = Executable::<Context>::from_text_bytes(
                    &bytes,
                    loader.clone(),
                    SBPFVersion::V3,
                    FunctionRegistry::default(),
                )
                .unwrap();
                executable
                    .verify::<crate::verifier::RequisiteVerifier>()
                    .unwrap();
                executable.jit_compile().unwrap();
                let mut stack =
                    AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(config.stack_size());
                let mut input = [0u8; 64];
                let regions = vec![
                    executable.get_ro_region(),
                    MemoryRegion::new(&raw mut *stack.as_slice_mut(), ebpf::MM_STACK_START),
                    MemoryRegion::new(&raw mut input, ebpf::MM_INPUT_START),
                ];
                let mut context = Context {
                    remaining: 100,
                    memory_mapping: unsafe {
                        MemoryMapping::new(regions, &config, SBPFVersion::V3).unwrap()
                    },
                };
                let mut vm = EbpfVm::new(loader, SBPFVersion::V3, &mut context, stack.len());
                vm.registers[1] = ebpf::MM_INPUT_START;
                SLOW_PATH_CALLS.with(|calls| calls.set(0));
                let (_, result) = vm.execute_program(
                    &executable,
                    &mut ExecutionMode::Jit,
                    &mut [CallFrame::default()],
                );
                let slow_path_calls = SLOW_PATH_CALLS.with(|calls| calls.get());
                if aligned_memory_mapping {
                    assert_eq!(slow_path_calls, expected_slow_path_calls, "{offset}");
                } else {
                    // Every access translates through the slow path, up to the faulting one
                    assert_eq!(
                        slow_path_calls,
                        if offset <= 56 { 7 } else { 5 },
                        "{offset}"
                    );
                }
                assert_eq!(result.is_ok(), offset <= 56, "{result:?}");
            }
        }
    }
}
