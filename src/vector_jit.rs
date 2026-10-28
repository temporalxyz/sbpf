//! AVX-512 lowering of the vector ops (child module of [crate::jit])
//!
//! Design:
//! - Vector register `vN` is pinned to `zmmN`. `zmm30`, `zmm31` and `k1`-`k3` are scratch. Masked
//!   ops load `k2` from `r2`, compares write `k1`, gathers and scatters work on `k3`.
//! - Ops with more than one CU check the instruction meter for all of them at their pc.
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
//!   once, see [MemoryRuns]. `ldn` and `stn` translate their span with the same subroutine, if
//!   it is not plain they translate each register's access on its own, like single `ld` / `st`.
//! - Gathers and scatters whose lanes are not in one plain region run the interpreter semantics
//!   in [vector_slow_path].

use super::*;
use crate::{
    error::StableResult,
    memory_region::{jit_region_layout, AccessType, JitRegionTable},
    vector::{
        self, encode_masked, gather_scatter_shape, op, Kind, Mask, VectorInsn, VECTOR_OP_SRC,
        VECTOR_REGISTERS, VECTOR_SIZE,
    },
    vm::EncryptedHostAddressToEbpfVm,
    x86_evex::{
        kmovq_from_gpr, kmovq_to_gpr, kortest, kxnorq, EvexInstruction, LegacyInstruction, Length,
        Map::*,
        Prefix::{P66, PF2, PF3},
        Rm, VZEROUPPER,
    },
};
use std::convert::TryFrom;

/// Machine code length of the vector subroutines (prologue and translation), upper bound
pub(super) const MAX_VECTOR_SUBROUTINES_MACHINE_CODE_LENGTH: usize = 32 * 1024;

/// Lengths of the memory accesses which have a translation subroutine
const TRANSLATION_LENGTHS: [u64; 7] = [1, 2, 4, 8, 16, 32, VECTOR_SIZE as u64];

/// Scratch vector registers
const ZMM30: u8 = 30;
const ZMM31: u8 = 31;
/// Scratch mask register (compare results)
const K1: u8 = 1;
/// Mask register holding r2 for masked ops
const K2: u8 = 2;
/// Working mask of gathers and scatters
const K3: u8 = 3;
/// Red zone slots in which gathers and scatters save RAX and RCX (below the slots the
/// translation subroutines use)
const GATHER_SCATTER_SAVED_REGISTERS: [(X86Register, i32); 2] = [(RAX, -48), (RCX, -56)];

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
        (is_x86_feature_detected!("avx512cd"), "avx512cd"),
        (is_x86_feature_detected!("avx512vbmi"), "avx512vbmi"),
        (is_x86_feature_detected!("avx512vbmi2"), "avx512vbmi2"),
        (is_x86_feature_detected!("gfni"), "gfni"),
        (is_x86_feature_detected!("vaes"), "vaes"),
        (is_x86_feature_detected!("sha"), "sha"),
        (is_x86_feature_detected!("popcnt"), "popcnt"),
        (is_x86_feature_detected!("bmi1"), "bmi1"),
        (is_x86_feature_detected!("lzcnt"), "lzcnt"),
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

/// Slow path of gathers and scatters. Executes the op like the interpreter (without CUs) and
/// writes the result to [EbpfVm::program_result].
extern "C" fn vector_slow_path<C: ContextObject>(
    mut vm: EncryptedHostAddressToEbpfVm<C>,
    r1: u64,
    r2: u64,
    word: u64,
) {
    // SAFETY: called from the JIT code of this VM, which holds no other reference to it
    unsafe {
        vm.with_vm(|vm| {
            let insn = ebpf::get_insn(&word.to_le_bytes(), 0);
            let v = VectorInsn::decode(&insn).unwrap();
            vm.program_result = match vector::execute_decoded(vm, &v, r1, r2) {
                Ok(_) => ProgramResult::Ok(0),
                Err(err) => ProgramResult::Err(err),
            };
        })
    }
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
    /// Start of the code of the vector ops since the last one outside of a run
    budget_start: usize,
    /// Sum of [max_machine_code_length] of these ops
    budget: usize,
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
    /// `kmovq k, r64`
    KmovqFromGpr(u8, X86Register),
    /// Legacy SSE or scalar instruction
    Legacy(LegacyInstruction),
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
            Self::KmovqFromGpr(mask, source) => kmovq_from_gpr(*mask, *source).to_vec(),
            Self::Legacy(instruction) => {
                let (bytes, len) = instruction.encode();
                bytes[..len].to_vec()
            }
        }
    }
}

/// The lowering of a vector op
///
/// If `memory` is set, the address in `r1` is first translated into `REGISTER_SCRATCH` (R11)
/// and the instructions access memory through it. Gathers and scatters are the native
/// instruction only, see [JitCompiler::emit_gather_scatter].
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
    let r2 = REGISTER_MAP[2];
    let load = |instruction: E| vec![Evex(instruction.mem(REGISTER_SCRATCH, 0))];
    // vmovdqu8/16/32/64 for the lane width code in imm8
    let vmovdqu = |opcode, reg| {
        let prefix = if imm8 < 2 { PF2 } else { PF3 };
        E::rr(prefix, M0F, imm8 & 1 == 1, opcode, reg, 0).mem(REGISTER_SCRATCH, 0)
    };
    // vmovdqu64 of `count` registers from `reg` on to consecutive 64 byte slots
    let vmovdqu64s = |opcode, reg| {
        (0..v.register_count())
            .map(|k| {
                Evex(
                    E::rr(PF3, M0F, true, opcode, reg + k, 0)
                        .mem(REGISTER_SCRATCH, k as i32 * VECTOR_SIZE as i32),
                )
            })
            .collect()
    };
    let mut instructions = match v.info.op {
        op::LD => load(E::rr(PF3, M0F, true, 0x6f, o, 0)), // vmovdqu64
        op::ST => load(E::rr(PF3, M0F, true, 0x7f, a, 0)), // vmovdqu64
        op::LDBC_D => load(E::rr(P66, M0F38, false, 0x58, o, 0)), // vpbroadcastd
        op::LDBC_Q => load(E::rr(P66, M0F38, true, 0x59, o, 0)), // vpbroadcastq
        op::SPLAT_B => vec![Evex(E::rr(P66, M0F38, false, 0x7a, o, r1))], // vpbroadcastb
        op::SPLAT_W => vec![Evex(E::rr(P66, M0F38, false, 0x7b, o, r1))], // vpbroadcastw
        op::SPLAT_D => vec![Evex(E::rr(P66, M0F38, false, 0x7c, o, r1))], // vpbroadcastd
        op::SPLAT_Q => vec![Evex(E::rr(P66, M0F38, true, 0x7c, o, r1))], // vpbroadcastq
        op::ZERO => vec![Evex(E::rrr(P66, M0F, false, 0xef, o, o, o))], // vpxord
        op::MOV => vec![Evex(E::rr(P66, M0F, true, 0x6f, o, a))], // vmovdqa64
        op::ADD_B => binary(M0F, false, 0xfc),             // vpaddb
        op::ADD_W => binary(M0F, false, 0xfd),             // vpaddw
        op::ADD_D => binary(M0F, false, 0xfe),             // vpaddd
        op::ADD_Q => binary(M0F, true, 0xd4),              // vpaddq
        op::SUB_B => binary(M0F, false, 0xf8),             // vpsubb
        op::SUB_W => binary(M0F, false, 0xf9),             // vpsubw
        op::SUB_D => binary(M0F, false, 0xfa),             // vpsubd
        op::SUB_Q => binary(M0F, true, 0xfb),              // vpsubq
        op::MULLO_W => binary(M0F, false, 0xd5),           // vpmullw
        op::MULLO_D => binary(M0F38, false, 0x40),         // vpmulld
        op::MULLO_Q => binary(M0F38, true, 0x40),          // vpmullq
        op::MULWU => binary(M0F, true, 0xf4),              // vpmuludq
        op::MULWS => binary(M0F38, true, 0x28),            // vpmuldq
        op::MULHI_W => binary(M0F, false, 0xe5),           // vpmulhw
        op::MULHIU_W => binary(M0F, false, 0xe4),          // vpmulhuw
        op::MAX_D => binary(M0F38, false, 0x3d),           // vpmaxsd
        op::MIN_D => binary(M0F38, false, 0x39),           // vpminsd
        op::MAXU_D => binary(M0F38, false, 0x3f),          // vpmaxud
        op::MINU_D => binary(M0F38, false, 0x3b),          // vpminud
        op::MAX_W => binary(M0F, false, 0xee),             // vpmaxsw
        op::MIN_W => binary(M0F, false, 0xea),             // vpminsw
        op::MAXU_Q => binary(M0F38, true, 0x3f),           // vpmaxuq
        op::MINU_Q => binary(M0F38, true, 0x3b),           // vpminuq
        op::MAXU_W => binary(M0F38, false, 0x3e),          // vpmaxuw
        op::MINU_W => binary(M0F38, false, 0x3a),          // vpminuw
        op::AND => binary(M0F, true, 0xdb),                // vpandq
        op::OR => binary(M0F, true, 0xeb),                 // vporq
        op::XOR => binary(M0F, true, 0xef),                // vpxorq
        op::ANDN => binary(M0F, true, 0xdf),               // vpandnq
        op::TERNLOG => vec![Evex(E::rrr(P66, M0F3A, true, 0x25, o, a, b).imm(imm8))], // vpternlogq
        op::SHLI_W => shift(false, 0x71, 6),               // vpsllw
        op::SHLI_D => shift(false, 0x72, 6),               // vpslld
        op::SHLI_Q => shift(true, 0x73, 6),                // vpsllq
        op::SHRI_W => shift(false, 0x71, 2),               // vpsrlw
        op::SHRI_D => shift(false, 0x72, 2),               // vpsrld
        op::SHRI_Q => shift(true, 0x73, 2),                // vpsrlq
        op::SARI_W => shift(false, 0x71, 4),               // vpsraw
        op::SARI_D => shift(false, 0x72, 4),               // vpsrad
        op::SARI_Q => shift(true, 0x72, 4),                // vpsraq
        op::ROLI_D => shift(false, 0x72, 1),               // vprold
        op::ROLI_Q => shift(true, 0x72, 1),                // vprolq
        op::SHLV_D => binary(M0F38, false, 0x47),          // vpsllvd
        op::SHLV_Q => binary(M0F38, true, 0x47),           // vpsllvq
        op::SHRV_D => binary(M0F38, false, 0x45),          // vpsrlvd
        op::SHRV_Q => binary(M0F38, true, 0x45),           // vpsrlvq
        op::ROLV_D => binary(M0F38, false, 0x15),          // vprolvd
        op::ROLV_Q => binary(M0F38, true, 0x15),           // vprolvq
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
        op::MPOPCNT | op::MTZCNT | op::MLZCNT => vec![HostInstruction::Legacy(LegacyInstruction {
            prefix: Some(0xf3),
            rex_w: true,
            map: M0F,
            opcode: match v.info.op {
                op::MPOPCNT => 0xb8, // popcnt
                op::MTZCNT => 0xbc,  // tzcnt
                _ => 0xbd,           // lzcnt
            },
            reg: r0 as u8,
            rm: r2 as u8,
            imm8: None,
        })],
        op::BLENDM => vec![
            HostInstruction::KmovqFromGpr(K2, r2),
            Evex(
                E::rrr(
                    P66,
                    M0F38,
                    imm8 & 1 == 1,
                    [0x66, 0x66, 0x64, 0x64][imm8 as usize],
                    o,
                    a,
                    b,
                )
                .mask(K2, false),
            ), // vpblendmb/w/d/q
        ],
        op::BCAST_Q | op::BCAST_D | op::BCAST_W | op::BCAST_B => lower_broadcast(v),
        op::PERMQI => vec![Evex(E::rr(P66, M0F3A, true, 0x00, o, a).imm(imm8))], // vpermq
        op::EXTRACT256 => vec![Evex(E::rr(P66, M0F3A, true, 0x3b, a, o).imm(imm8))], // vextracti64x4
        op::PERM2I => vec![Evex(E::rrr(
            P66,
            M0F38,
            imm8 & 1 == 1,
            if imm8 < 2 { 0x75 } else { 0x76 },
            o,
            a,
            b,
        ))], // vpermi2b/w/d/q
        op::ALIGN_Q => vec![Evex(E::rrr(P66, M0F3A, true, 0x03, o, a, b).imm(imm8))], // valignq
        op::ALIGN_D => vec![Evex(E::rrr(P66, M0F3A, false, 0x03, o, a, b).imm(imm8))], // valignd
        op::EXTRACT128 => vec![Evex(E::rr(P66, M0F3A, false, 0x39, a, o).imm(imm8))], // vextracti32x4
        op::INSERT256 => vec![Evex(E::rrr(P66, M0F3A, true, 0x3a, o, a, b).imm(imm8))], // vinserti64x4
        op::INSERT128 => vec![Evex(E::rrr(P66, M0F3A, false, 0x38, o, a, b).imm(imm8))], // vinserti32x4
        op::PERM2T => vec![Evex(E::rrr(
            P66,
            M0F38,
            imm8 & 1 == 1,
            if imm8 < 2 { 0x7d } else { 0x7e },
            o,
            a,
            b,
        ))], // vpermt2b/d/q
        op::PERMB => binary(M0F38, false, 0x8d),                                         // vpermb
        op::PERMW => binary(M0F38, true, 0x8d),                                          // vpermw
        op::SHUFD => vec![Evex(E::rr(P66, M0F, false, 0x70, o, a).imm(imm8))],           // vpshufd
        op::SHUFHW => vec![Evex(E::rr(PF3, M0F, false, 0x70, o, a).imm(imm8))],          // vpshufhw
        op::SHUFLW => vec![Evex(E::rr(PF2, M0F, false, 0x70, o, a).imm(imm8))],          // vpshuflw
        op::SHUF128 => vec![Evex(E::rrr(P66, M0F3A, true, 0x43, o, a, b).imm(imm8))], // vshufi64x2
        op::SHUFPS => vec![Evex(
            E::rrr(crate::x86_evex::Prefix::None, M0F, false, 0xc6, o, a, b).imm(imm8),
        )], // vshufps
        op::ALIGNR => vec![Evex(E::rrr(P66, M0F3A, false, 0x0f, o, a, b).imm(imm8))], // vpalignr
        op::BSHLI => shift(false, 0x73, 7),                                           // vpslldq
        op::BSHRI => shift(false, 0x73, 3),                                           // vpsrldq
        op::UNPACKLO => binary(M0F, imm8 == 3, [0x60, 0x61, 0x62, 0x6c][imm8 as usize]), // vpunpckl*
        op::UNPACKHI => binary(M0F, imm8 == 3, [0x68, 0x69, 0x6a, 0x6d][imm8 as usize]), // vpunpckh*
        op::COMPRESS => vec![Evex(E::rr(
            P66,
            M0F38,
            imm8 & 1 == 1,
            if imm8 < 2 { 0x63 } else { 0x8b },
            a,
            o,
        ))], // vpcompressb/w/d/q
        op::EXPAND => vec![Evex(E::rr(
            P66,
            M0F38,
            imm8 & 1 == 1,
            if imm8 < 2 { 0x62 } else { 0x89 },
            o,
            a,
        ))], // vpexpandb/w/d/q
        op::PACKUS_D => binary(M0F38, false, 0x2b), // vpackusdw
        op::NARROW_D_W => narrow(0x33),             // vpmovdw
        op::NARROWUS_D_W => narrow(0x13),           // vpmovusdw
        op::NARROW_Q_B => narrow(0x32),             // vpmovqb
        op::NARROWS_Q_B => narrow(0x22),            // vpmovsqb
        op::NARROWUS_Q_B => narrow(0x12),           // vpmovusqb
        op::NARROW_Q_W => narrow(0x34),             // vpmovqw
        op::NARROWS_Q_W => narrow(0x24),            // vpmovsqw
        op::NARROWUS_Q_W => narrow(0x14),           // vpmovusqw
        op::NARROW_Q_D => narrow(0x35),             // vpmovqd
        op::NARROWS_Q_D => narrow(0x25),            // vpmovsqd
        op::NARROWUS_Q_D => narrow(0x15),           // vpmovusqd
        op::NARROW_W_B => narrow(0x30),             // vpmovwb
        op::NARROWS_W_B => narrow(0x20),            // vpmovswb
        op::NARROWUS_W_B => narrow(0x10),           // vpmovuswb
        op::ZX_B_W => unary(P66, false, 0x30),      // vpmovzxbw
        op::SX_B_W => unary(P66, false, 0x20),      // vpmovsxbw
        op::ZX_B_Q => unary(P66, false, 0x32),      // vpmovzxbq
        op::SX_B_Q => unary(P66, false, 0x22),      // vpmovsxbq
        op::ZX_W_Q => unary(P66, false, 0x34),      // vpmovzxwq
        op::SX_W_Q => unary(P66, false, 0x24),      // vpmovsxwq
        op::EXTRACT_B | op::EXTRACT_W => {
            let per_128 = if v.info.op == op::EXTRACT_B { 16 } else { 8 };
            let mut sequence = vec![];
            let mut source = a;
            if imm8 >= per_128 {
                sequence.push(Evex(
                    E::rr(P66, M0F3A, false, 0x39, a, ZMM30).imm(imm8 / per_128),
                )); // vextracti32x4
                source = ZMM30;
            }
            sequence.push(Evex(
                E::rr(
                    P66,
                    M0F3A,
                    false,
                    if per_128 == 16 { 0x14 } else { 0x15 },
                    source,
                    r0 as u8,
                )
                .imm(imm8 % per_128)
                .length(Length::L128),
            )); // vpextrb / vpextrw
            sequence
        }
        op::INSERT_B | op::INSERT_W | op::INSERT_D => {
            let (map, opcode, per_128) = match v.info.op {
                op::INSERT_B => (M0F3A, 0x20, 16), // vpinsrb
                op::INSERT_W => (M0F, 0xc4, 8),    // vpinsrw
                _ => (M0F3A, 0x22, 4),             // vpinsrd
            };
            vec![
                Evex(E::rr(P66, M0F3A, false, 0x39, o, ZMM30).imm(imm8 / per_128)), // vextracti32x4
                Evex(
                    E::rrr(P66, map, false, opcode, ZMM30, ZMM30, r1)
                        .imm(imm8 % per_128)
                        .length(Length::L128),
                ),
                Evex(E::rrr(P66, M0F3A, false, 0x38, o, o, ZMM30).imm(imm8 / per_128)), // vinserti32x4
            ]
        }
        op::MOVR_Q => vec![Evex(
            E::rr(P66, M0F, true, 0x6e, o, r1).length(Length::L128),
        )], // vmovq
        op::CMP => {
            // vpcmp[u]b/w/d/q k1, in0, in1, predicate
            let width = (imm8 >> 4) & 3;
            let unsigned = (imm8 >> 3) & 1;
            let opcode = [0x3f, 0x3f, 0x1f, 0x1f][width as usize] - unsigned;
            vec![
                Evex(E::rrr(P66, M0F3A, width & 1 == 1, opcode, K1, a, b).imm(imm8 & 7)),
                HostInstruction::KmovqToGpr(r0, K1),
            ]
        }
        op::BITSHUF => vec![
            Evex(E::rrr(P66, M0F38, false, 0x8f, K1, a, b)), // vpshufbitqmb
            HostInstruction::KmovqToGpr(r0, K1),
        ],
        op::MUL52LO | op::MUL52HI => {
            // vpmadd52luq / vpmadd52huq into a zeroed accumulator, through zmm30 if out is an
            // input or its old lanes are merged
            let opcode = if v.info.op == op::MUL52LO { 0xb4 } else { 0xb5 };
            let t = if v.mode == 1 || o == a || o == b {
                ZMM30
            } else {
                o
            };
            let mut sequence = vec![
                Evex(E::rrr(P66, M0F, true, 0xef, t, t, t)), // vpxorq
                Evex(E::rrr(P66, M0F38, true, opcode, t, a, b)),
            ];
            if t != o {
                sequence.push(Evex(E::rr(P66, M0F, true, 0x6f, o, t))); // vmovdqa64
            }
            sequence
        }
        op::ABS_Q => unary(P66, true, 0x1f),        // vpabsq
        op::MAX_Q => binary(M0F38, true, 0x3d),     // vpmaxsq
        op::MIN_Q => binary(M0F38, true, 0x39),     // vpminsq
        op::ABS_B => unary(P66, false, 0x1c),       // vpabsb
        op::ABS_W => unary(P66, false, 0x1d),       // vpabsw
        op::ABS_D => unary(P66, false, 0x1e),       // vpabsd
        op::MAX_B => binary(M0F38, false, 0x3c),    // vpmaxsb
        op::MIN_B => binary(M0F38, false, 0x38),    // vpminsb
        op::MAXU_B => binary(M0F, false, 0xde),     // vpmaxub
        op::MINU_B => binary(M0F, false, 0xda),     // vpminub
        op::ADDS_B => binary(M0F, false, 0xec),     // vpaddsb
        op::ADDS_W => binary(M0F, false, 0xed),     // vpaddsw
        op::ADDUS_B => binary(M0F, false, 0xdc),    // vpaddusb
        op::ADDUS_W => binary(M0F, false, 0xdd),    // vpaddusw
        op::SUBS_B => binary(M0F, false, 0xe8),     // vpsubsb
        op::SUBS_W => binary(M0F, false, 0xe9),     // vpsubsw
        op::SUBUS_B => binary(M0F, false, 0xd8),    // vpsubusb
        op::SUBUS_W => binary(M0F, false, 0xd9),    // vpsubusw
        op::AVG_B => binary(M0F, false, 0xe0),      // vpavgb
        op::AVG_W => binary(M0F, false, 0xe3),      // vpavgw
        op::SAD => binary(M0F, false, 0xf6),        // vpsadbw
        op::MULHRS_W => binary(M0F38, false, 0x0b), // vpmulhrsw
        op::DPBUSDS => binary(M0F38, false, 0x51),  // vpdpbusds
        op::DPWSSDS => binary(M0F38, false, 0x53),  // vpdpwssds
        op::TEST | op::TESTN => vec![
            Evex(E::rrr(
                if v.info.op == op::TEST { P66 } else { PF3 },
                M0F38,
                imm8 & 1 == 1,
                0x26 | imm8 >> 1,
                K1,
                a,
                b,
            )), // vptestm / vptestnm b/w/d/q
            HostInstruction::KmovqToGpr(r0, K1),
        ],
        op::MOVEMASK_W | op::MOVEMASK_D | op::MOVEMASK_Q => vec![
            Evex(E::rr(
                PF3,
                M0F38,
                v.info.op != op::MOVEMASK_D,
                if v.info.op == op::MOVEMASK_W {
                    0x29
                } else {
                    0x39
                },
                K1,
                a,
            )), // vpmovw2m / vpmovd2m / vpmovq2m
            HostInstruction::KmovqToGpr(r0, K1),
        ],
        op::MOVM_B | op::MOVM_W | op::MOVM_D | op::MOVM_Q => {
            let width = v.info.op - op::MOVM_B;
            vec![
                HostInstruction::KmovqFromGpr(K2, r2),
                Evex(E::rr(
                    PF3,
                    M0F38,
                    width & 1 == 1,
                    0x28 | (width >> 1) << 4,
                    o,
                    K2,
                )), // vpmovm2b/w/d/q
            ]
        }
        op::TERNLOG_D => {
            vec![Evex(E::rrr(P66, M0F3A, false, 0x25, o, a, b).imm(imm8))] // vpternlogd
        }
        op::SHLV_W => binary(M0F38, true, 0x12),  // vpsllvw
        op::SHRV_W => binary(M0F38, true, 0x10),  // vpsrlvw
        op::SARV_W => binary(M0F38, true, 0x11),  // vpsravw
        op::SARV_D => binary(M0F38, false, 0x46), // vpsravd
        op::SARV_Q => binary(M0F38, true, 0x46),  // vpsravq
        op::SHLDI_W | op::SHLDI_D | op::SHLDI_Q | op::SHRDI_W | op::SHRDI_D | op::SHRDI_Q => {
            // vpshldw/d/q, vpshrdw/d/q
            let (w, opcode) = match v.info.op {
                op::SHLDI_W => (true, 0x70),
                op::SHLDI_D => (false, 0x71),
                op::SHLDI_Q => (true, 0x71),
                op::SHRDI_W => (true, 0x72),
                op::SHRDI_D => (false, 0x73),
                _ => (true, 0x73),
            };
            vec![Evex(E::rrr(P66, M0F3A, w, opcode, o, a, b).imm(imm8))]
        }
        op::SHLDV_W => binary(M0F38, true, 0x70), // vpshldvw
        op::SHLDV_D => binary(M0F38, false, 0x71), // vpshldvd
        op::SHLDV_Q => binary(M0F38, true, 0x71), // vpshldvq
        op::SHRDV_W => binary(M0F38, true, 0x72), // vpshrdvw
        op::SHRDV_D => binary(M0F38, false, 0x73), // vpshrdvd
        op::SHRDV_Q => binary(M0F38, true, 0x73), // vpshrdvq
        op::CONFLICT_D => unary(P66, false, 0xc4), // vpconflictd
        op::CONFLICT_Q => unary(P66, true, 0xc4), // vpconflictq
        op::LZCNT_D => unary(P66, false, 0x44),   // vplzcntd
        op::LZCNT_Q => unary(P66, true, 0x44),    // vplzcntq
        op::POPCNT_W => unary(P66, true, 0x54),   // vpopcntw
        op::POPCNT_D => unary(P66, false, 0x55),  // vpopcntd
        op::MULTISHIFT => binary(M0F38, true, 0x83), // vpmultishiftqb
        op::GF2P8MUL => binary(M0F38, false, 0xcf), // vgf2p8mulb
        op::GF2P8AFFINE => vec![Evex(E::rrr(P66, M0F3A, true, 0xce, o, a, b).imm(imm8))], // vgf2p8affineqb
        op::GF2P8AFFINEINV => vec![Evex(E::rrr(P66, M0F3A, true, 0xcf, o, a, b).imm(imm8))], // vgf2p8affineinvqb
        op::AESENC => binary(M0F38, false, 0xdc), // vaesenc
        op::AESENCLAST => binary(M0F38, false, 0xdd), // vaesenclast
        op::AESDEC => binary(M0F38, false, 0xde), // vaesdec
        op::AESDECLAST => binary(M0F38, false, 0xdf), // vaesdeclast
        op::SHA256RNDS2
        | op::SHA256MSG1
        | op::SHA256MSG2
        | op::SHA1RNDS4
        | op::SHA1NEXTE
        | op::SHA1MSG1
        | op::SHA1MSG2 => lower_sha(v),
        op::GATHER_QQ | op::GATHER_DD | op::GATHER_DQ | op::GATHER_QD => {
            let (qword_index, element, _) = gather_scatter_shape(v.info.op);
            vec![Evex(
                E::rr(P66, M0F38, element == 8, 0x90 | qword_index as u8, o, 0)
                    .vsib(REGISTER_SCRATCH, a, imm8, 0)
                    .mask(K3, false),
            )] // vpgatherdd/dq/qd/qq
        }
        op::SCATTER_QQ | op::SCATTER_DD | op::SCATTER_DQ | op::SCATTER_QD => {
            let (qword_index, element, _) = gather_scatter_shape(v.info.op);
            vec![Evex(
                E::rr(P66, M0F38, element == 8, 0xa0 | qword_index as u8, b, 0)
                    .vsib(REGISTER_SCRATCH, a, imm8, 0)
                    .mask(K3, false),
            )] // vpscatterdd/dq/qd/qq
        }
        op::LDN => vmovdqu64s(0x6f, o),
        op::STN => vmovdqu64s(0x7f, a),
        op::LD_M => vec![
            HostInstruction::KmovqFromGpr(K2, r2),
            Evex(vmovdqu(0x6f, o).mask(K2, v.mode == 2)),
        ],
        op::ST_M => vec![
            HostInstruction::KmovqFromGpr(K2, r2),
            Evex(vmovdqu(0x7f, a).mask(K2, false)),
        ],
        op::LD_Y => load(E::rr(PF3, M0F, true, 0x6f, o, 0).length(Length::L256)), // vmovdqu64
        op::ST_Y => load(E::rr(PF3, M0F, true, 0x7f, a, 0).length(Length::L256)), // vmovdqu64
        op::LD_X => load(E::rr(PF3, M0F, false, 0x6f, o, 0).length(Length::L128)), // vmovdqu32
        op::ST_X => load(E::rr(PF3, M0F, false, 0x7f, a, 0).length(Length::L128)), // vmovdqu32
        op::LDBC_B => load(E::rr(P66, M0F38, false, 0x78, o, 0)),                 // vpbroadcastb
        op::LDBC_W => load(E::rr(P66, M0F38, false, 0x79, o, 0)),                 // vpbroadcastw
        op::LDBC_X => load(E::rr(P66, M0F38, false, 0x5a, o, 0)),                 // vbroadcasti32x4
        op::LDBC_Y => load(E::rr(P66, M0F38, true, 0x5b, o, 0)),                  // vbroadcasti64x4
        _ => unreachable!("every op has a lowering"),
    };
    // Mask modes 1 and 2 apply to the last instruction, which writes out
    if v.mode != 0
        && matches!(v.info.mask, Mask::Modes(..))
        && !matches!(v.info.kind, Kind::Lsi | Kind::Sti | Kind::Ga | Kind::Sc)
    {
        if let Some(Evex(last)) = instructions.last_mut() {
            *last = last.mask(K2, v.mode == 2);
        }
        instructions.insert(0, HostInstruction::KmovqFromGpr(K2, r2));
    }
    Lowering {
        memory: memory_access(v),
        instructions,
    }
}

/// The memory access of an op whose address the JIT translates before the lowering
fn memory_access(v: &VectorInsn) -> Option<(AccessType, u64)> {
    let span = VECTOR_SIZE as u64 * v.register_count() as u64;
    Some(match v.info.op {
        op::LD | op::LD_M | op::LDN => (AccessType::Load, span),
        op::ST | op::ST_M | op::STN => (AccessType::Store, span),
        op::LDBC_D => (AccessType::Load, 4),
        op::LDBC_Q => (AccessType::Load, 8),
        op::LDBC_B => (AccessType::Load, 1),
        op::LDBC_W => (AccessType::Load, 2),
        op::LD_X | op::LDBC_X => (AccessType::Load, 16),
        op::ST_X => (AccessType::Store, 16),
        op::LD_Y | op::LDBC_Y => (AccessType::Load, 32),
        op::ST_Y => (AccessType::Store, 32),
        _ => return None,
    })
}

/// `bcast.*`: the lane moves to the bottom of xmm30 (valign, or vextracti32x4 and vpsrldq for
/// bytes and words), then vpbroadcast
fn lower_broadcast(v: &VectorInsn) -> Vec<HostInstruction> {
    use HostInstruction::Evex;
    type E = EvexInstruction;
    let (o, a, imm8) = (v.out, v.in0, v.imm8);
    let (bytes, w, opcode) = match v.info.op {
        op::BCAST_Q => (8, true, 0x59),
        op::BCAST_D => (4, false, 0x58),
        op::BCAST_W => (2, false, 0x79),
        _ => (1, false, 0x78),
    };
    let mut sequence = vec![];
    let mut source = a;
    if bytes >= 4 {
        if imm8 != 0 {
            sequence.push(Evex(E::rrr(P66, M0F3A, w, 0x03, ZMM30, a, a).imm(imm8))); // valignq/d
            source = ZMM30;
        }
    } else {
        let per_128 = 16 / bytes;
        if imm8 >= per_128 {
            sequence.push(Evex(
                E::rr(P66, M0F3A, false, 0x39, a, ZMM30).imm(imm8 / per_128),
            )); // vextracti32x4
            source = ZMM30;
        }
        if imm8 % per_128 != 0 {
            sequence.push(Evex(
                E::rrr(P66, M0F, false, 0x73, 3, ZMM30, source)
                    .imm(imm8 % per_128 * bytes)
                    .length(Length::L128),
            )); // vpsrldq
            source = ZMM30;
        }
    }
    sequence.push(Evex(E::rr(P66, M0F38, w, opcode, o, source))); // vpbroadcastq/d/w/b
    sequence
}

/// SHA-NI only has legacy SSE encodings, which reach xmm0-15 and preserve bits 511:128 of the
/// destination. Operands in other registers (and for `sha256rnds2` WK, which must be in xmm0)
/// move through low temporary registers. Only the low 128 bits of a temporary change, they are
/// saved in the 128-bit lanes of zmm30.
fn lower_sha(v: &VectorInsn) -> Vec<HostInstruction> {
    use HostInstruction::Evex;
    type E = EvexInstruction;
    let (o, a, b) = (v.out, v.in0, v.in1);
    let rnds2 = v.info.op == op::SHA256RNDS2;
    let (map, opcode, imm8) = match v.info.op {
        op::SHA256RNDS2 => (M0F38, 0xcb, None),
        op::SHA256MSG1 => (M0F38, 0xcc, None),
        op::SHA256MSG2 => (M0F38, 0xcd, None),
        op::SHA1RNDS4 => (M0F3A, 0xcc, Some(v.imm8)),
        op::SHA1NEXTE => (M0F38, 0xc8, None),
        op::SHA1MSG1 => (M0F38, 0xc9, None),
        op::SHA1MSG2 => (M0F38, 0xca, None),
        _ => unreachable!(),
    };
    // vinserti32x4 zmm_destination, zmm_destination, xmm_source, lane
    let insert = |destination: u8, source: u8, lane: u8| {
        Evex(E::rrr(P66, M0F3A, false, 0x38, destination, destination, source).imm(lane))
    };
    let usable = |reg: u8| reg < 16 && !(rnds2 && reg == 0);
    let mut free = (rnds2 as u8..16).filter(|reg| ![o, a, b].contains(reg));
    let to = if usable(o) { o } else { free.next().unwrap() };
    let ta = if usable(a) {
        a
    } else if a == o {
        to
    } else {
        free.next().unwrap()
    };
    let mut saved = vec![];
    if to != o {
        saved.push(to);
    }
    if ta != a && ta != to {
        saved.push(ta);
    }
    if rnds2 && b != 0 {
        saved.push(0);
    }
    let lane_of = |reg: u8| saved.iter().position(|r| *r == reg).unwrap() as u8;
    let restore = |reg: u8| {
        let lane = lane_of(reg);
        if lane == 0 {
            vec![insert(reg, ZMM30, 0)]
        } else {
            vec![
                Evex(E::rr(P66, M0F3A, false, 0x39, ZMM30, ZMM31).imm(lane)), // vextracti32x4
                insert(reg, ZMM31, 0),
            ]
        }
    };
    let mut sequence: Vec<HostInstruction> = saved
        .iter()
        .enumerate()
        .map(|(lane, reg)| insert(ZMM30, *reg, lane as u8))
        .collect();
    if to != o {
        sequence.push(insert(to, o, 0));
    }
    if ta != a && ta != to {
        sequence.push(insert(ta, a, 0));
    }
    if rnds2 && b != 0 {
        sequence.push(insert(0, b, 0));
    }
    sequence.push(HostInstruction::Legacy(LegacyInstruction {
        prefix: None,
        rex_w: false,
        map,
        opcode,
        reg: to,
        rm: ta,
        imm8,
    }));
    if rnds2 && b != 0 {
        sequence.extend(restore(0));
    }
    if to != o {
        sequence.push(insert(o, to, 0));
    }
    for reg in saved.iter().rev().filter(|reg| **reg != 0 || !rnds2) {
        sequence.extend(restore(*reg));
    }
    sequence
}

/// Signed minimum (or maximum) of the active index lanes (k3) of a gather or scatter into the
/// general purpose register `destination` (low 32 bits for dword indices)
pub(crate) fn gather_scatter_bound(
    v: &VectorInsn,
    maximum: bool,
    destination: X86Register,
) -> Vec<HostInstruction> {
    use HostInstruction::Evex;
    type E = EvexInstruction;
    let (q, _, lanes) = gather_scatter_shape(v.info.op);
    let length = if q || lanes == 16 {
        Length::L512
    } else {
        Length::L256
    };
    let reduce = if maximum { 0x3d } else { 0x39 }; // vpmaxs / vpmins
    let mut sequence = vec![
        // zmm30 = all ones, then the neutral element: MAX (shift right by 1) or MIN (shift left)
        Evex(E::rrr(P66, M0F3A, false, 0x25, ZMM30, ZMM30, ZMM30).imm(0xff)), // vpternlogd
        Evex(
            E::rrr(
                P66,
                M0F,
                q,
                if q { 0x73 } else { 0x72 },
                if maximum { 6 } else { 2 },
                ZMM30,
                ZMM30,
            )
            .imm(if !maximum {
                1
            } else if q {
                63
            } else {
                31
            }),
        ), // vpsrlq/d or vpsllq/d
        Evex(
            E::rr(P66, M0F, q, 0x6f, ZMM30, v.in0)
                .mask(K3, false)
                .length(length),
        ), // vmovdqa64/32
    ];
    if length == Length::L512 {
        sequence.push(Evex(E::rr(P66, M0F3A, true, 0x3b, ZMM30, ZMM31).imm(1))); // vextracti64x4
        sequence.push(Evex(
            E::rrr(P66, M0F38, q, reduce, ZMM30, ZMM30, ZMM31).length(Length::L256),
        ));
    }
    sequence.push(Evex(
        E::rr(P66, M0F3A, false, 0x39, ZMM30, ZMM31)
            .imm(1)
            .length(Length::L256),
    )); // vextracti32x4
    sequence.push(Evex(
        E::rrr(P66, M0F38, q, reduce, ZMM30, ZMM30, ZMM31).length(Length::L128),
    ));
    for shuffle in if q { &[0x4e][..] } else { &[0x4e, 0xb1][..] } {
        sequence.push(Evex(
            E::rr(P66, M0F, false, 0x70, ZMM31, ZMM30)
                .imm(*shuffle)
                .length(Length::L128),
        )); // vpshufd
        sequence.push(Evex(
            E::rrr(P66, M0F38, q, reduce, ZMM30, ZMM30, ZMM31).length(Length::L128),
        ));
    }
    sequence.push(Evex(
        E::rr(P66, M0F, q, 0x7e, ZMM30, destination as u8).length(Length::L128),
    )); // vmovq / vmovd
    sequence
}

/// Upper bound of the machine code length of a vector op, including its share of a memory run
pub(super) fn max_machine_code_length(v: &VectorInsn, registers_used: u32) -> usize {
    // Longest EVEX encoding plus a noop
    const PER_INSTRUCTION: usize = 14;
    // CU check and meter adjustment
    const CU_CHECK: usize = 64;
    // r1 + off (sanitized), the pc or span length slot and the call
    const ACCESS: usize = 80;
    // Span translation call, access check, fallback jumps and entry
    const RUN: usize = 160;
    let lowering = lower(v);
    let instructions = lowering.instructions.len() * PER_INSTRUCTION;
    let host_call = 2 * (registers_used.count_ones() as usize + 1) * PER_INSTRUCTION + 192;
    let mut length = CU_CHECK + instructions;
    if let Some((_, len)) = lowering.memory {
        // The access in a run and its fallback
        length += RUN + ACCESS + instructions;
        if !TRANSLATION_LENGTHS.contains(&len) {
            // One access per register
            length += lowering.instructions.len() * (ACCESS + PER_INSTRUCTION);
        }
    }
    if matches!(v.info.kind, Kind::Ga | Kind::Sc) {
        length += 64 * PER_INSTRUCTION + 2 * ACCESS + host_call;
    }
    length
}

/// Sum of [max_machine_code_length] of the vector ops in `prog`
pub(super) fn machine_code_length_estimate(
    prog: &[u8],
    sbpf_version: crate::program::SBPFVersion,
    registers_used: u32,
) -> usize {
    let mut length = 0;
    let mut insn_ptr: usize = 0;
    while (insn_ptr + 1) * ebpf::INSN_SIZE <= prog.len() {
        let insn = ebpf::get_insn_unchecked(prog, insn_ptr);
        if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
            insn_ptr += 1;
        } else if vector::is_vector_op(&insn, true, sbpf_version) {
            if let Ok(v) = VectorInsn::decode(&insn) {
                length += max_machine_code_length(&v, registers_used);
            }
        }
        insn_ptr += 1;
    }
    length
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
                self.emit_bytes(&kmovq_to_gpr(*destination, *mask));
            }
            HostInstruction::KmovqFromGpr(mask, source) => {
                self.emit_bytes(&kmovq_from_gpr(*mask, *source));
            }
            HostInstruction::Legacy(instruction) => {
                let (bytes, len) = instruction.encode();
                self.emit_bytes(&bytes[..len]);
            }
        }
        self.emit_random_noop();
    }

    fn emit_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.emit::<u8>(*byte);
        }
    }

    /// Emits the lowering of the vector op `insn`
    pub(super) fn emit_vector_op(&mut self, insn: &ebpf::Insn) -> Result<(), EbpfError> {
        let Ok(vector_insn) = VectorInsn::decode(insn) else {
            // Only reachable if the program was not verified. Same error as the interpreter.
            self.emit_ins(X86Instruction::load_immediate(
                REGISTER_SCRATCH,
                self.pc as i64,
            ));
            self.emit_ins(X86Instruction::jump_immediate(
                self.relative_to_anchor(ANCHOR_CALL_UNSUPPORTED_INSTRUCTION, 5),
            ));
            return Ok(());
        };
        let budget = max_machine_code_length(&vector_insn, self.vector_registers_used);
        if self.offset_in_text_section + budget >= self.result.text_section().len() {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        {
            let runs = &mut self.vector_memory_runs;
            if runs.run.is_none() {
                runs.budget_start = self.offset_in_text_section;
                runs.budget = 0;
            }
            runs.budget += budget;
        }
        self.emit_vector_cu_check(&vector_insn);
        let lowering = lower(&vector_insn);
        if matches!(vector_insn.info.kind, Kind::Ga | Kind::Sc) {
            self.emit_gather_scatter(&vector_insn, &lowering);
        } else if lowering.memory.is_some() {
            if self.vector_memory_runs.run.is_none() {
                self.emit_memory_run_start();
            }
            if self.vector_memory_runs.run.is_some() {
                self.emit_memory_run_access(vector_insn, &lowering);
            } else {
                self.emit_vector_memory_access(&vector_insn, &lowering, self.pc);
            }
        } else {
            for instruction in &lowering.instructions {
                self.emit_host_instruction(instruction);
            }
        }
        let pc = self.pc;
        if let Some(run) = self.vector_memory_runs.run.take_if(|run| run.end == pc) {
            self.emit_memory_run_end(run);
        }
        // The text section was sized with these bounds
        let runs = &self.vector_memory_runs;
        assert!(
            runs.run.is_some() || self.offset_in_text_section - runs.budget_start <= runs.budget
        );
        Ok(())
    }

    /// Ops with more than one CU: fail at their pc if the instruction meter does not cover all of
    /// them (like the interpreter), otherwise charge the extra CUs
    fn emit_vector_cu_check(&mut self, v: &VectorInsn) {
        let cu = v.info.cu(v.imm8);
        if cu <= 1 || !self.config.enable_instruction_meter {
            return;
        }
        let pc = self.pc as i64;
        self.emit_ins(X86Instruction::cmp_immediate(
            OperandSize::S64,
            REGISTER_INSTRUCTION_METER,
            pc + cu as i64 - 1,
            None,
        ));
        let covered = self.offset_in_text_section + 2;
        self.emit_ins(X86Instruction::conditional_jump_immediate(0x87, 0)); // ja covered
                                                                            // Ran out before this op: the meter already holds the pc to report
        self.emit_ins(X86Instruction::cmp_immediate(
            OperandSize::S64,
            REGISTER_INSTRUCTION_METER,
            pc,
            None,
        ));
        self.emit_ins(X86Instruction::conditional_jump_immediate(
            0x86,
            self.relative_to_anchor(ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS, 6),
        ));
        self.emit_ins(X86Instruction::load_immediate(
            REGISTER_INSTRUCTION_METER,
            pc,
        ));
        self.emit_ins(X86Instruction::jump_immediate(
            self.relative_to_anchor(ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS, 5),
        ));
        self.patch_rel32(covered);
        self.emit_ins(X86Instruction::alu_immediate(
            OperandSize::S64,
            0x81,
            5,
            REGISTER_INSTRUCTION_METER,
            cu as i64 - 1,
            None,
        ));
    }

    /// A memory access outside of a [MemoryRun], also the fallback of one in a run
    fn emit_vector_memory_access(&mut self, v: &VectorInsn, lowering: &Lowering, pc: usize) {
        let (access_type, len) = lowering.memory.unwrap();
        if TRANSLATION_LENGTHS.contains(&len) {
            self.emit_r1_plus_off(v.off);
            self.emit_vector_translation_call(access_type, len, pc);
            for instruction in &lowering.instructions {
                self.emit_host_instruction(instruction);
            }
            return;
        }
        // ldn and stn: the span translation, one access per register if the span is not plain
        let Some(span_translation) = self.inline_span_translation() else {
            self.emit_vector_register_accesses(v, lowering, pc);
            return;
        };
        let key = self.memory_run_key();
        self.emit_r1_plus_off(v.off);
        self.emit_ins(X86Instruction::store_immediate(
            OperandSize::S64,
            RSP,
            X86IndirectAccess::OffsetIndexShift(-16, RSP, 0),
            len as i64 - 1 + key,
        ));
        self.emit_ins(X86Instruction::call_immediate(self.relative_to_offset(
            span_translation[(access_type == AccessType::Store) as usize],
            5,
        )));
        self.emit_ins(X86Instruction::test(
            OperandSize::S64,
            REGISTER_SCRATCH,
            REGISTER_SCRATCH,
            None,
        ));
        let slow_path = self.offset_in_text_section + 2;
        self.emit_ins(X86Instruction::conditional_jump_immediate(0x84, 0)); // jz slow_path
        self.emit_displaced(lowering, key as i32);
        let done = self.offset_in_text_section + 1;
        self.emit_ins(X86Instruction::jump_immediate(0));
        self.patch_rel32(slow_path);
        self.emit_vector_register_accesses(v, lowering, pc);
        self.patch_rel32(done);
    }

    /// `ldn` and `stn` as single register accesses in ascending order, each translated on its own
    fn emit_vector_register_accesses(&mut self, v: &VectorInsn, lowering: &Lowering, pc: usize) {
        let (access_type, _) = lowering.memory.unwrap();
        for (k, instruction) in lowering.instructions.iter().enumerate() {
            let HostInstruction::Evex(evex) = instruction else {
                unreachable!("ldn and stn are vmovdqu64 only");
            };
            let off = v.off as i64 + (k * VECTOR_SIZE) as i64;
            self.emit_vm_address(REGISTER_MAP[1], off, true);
            self.emit_vector_translation_call(access_type, VECTOR_SIZE as u64, pc);
            self.emit_host_instruction(&HostInstruction::Evex(evex.mem(REGISTER_SCRATCH, 0)));
        }
    }

    /// The span translation subroutines, if the region layout and the mapping allow them
    fn inline_span_translation(&self) -> Option<[usize; 2]> {
        self.vector_memory_runs
            .span_translation
            .filter(|_| self.config.aligned_memory_mapping)
    }

    /// Emits the lowering with the memory operands relative to REGISTER_SCRATCH + `displacement`
    fn emit_displaced(&mut self, lowering: &Lowering, displacement: i32) {
        for instruction in &lowering.instructions {
            let instruction = match instruction {
                HostInstruction::Evex(evex) => match evex.rm {
                    Rm::Mem(_, own) => {
                        HostInstruction::Evex(evex.mem(REGISTER_SCRATCH, displacement + own))
                    }
                    Rm::Vsib {
                        index, scale, disp, ..
                    } => HostInstruction::Evex(evex.vsib(
                        REGISTER_SCRATCH,
                        index,
                        scale,
                        displacement + disp,
                    )),
                    Rm::Reg(_) => *instruction,
                },
                other => *other,
            };
            self.emit_host_instruction(&instruction);
        }
    }

    /// Calls [vector_slow_path] for the op at `pc` and throws its error
    fn emit_vector_slow_path_call(&mut self, v: &VectorInsn, pc: usize) {
        let word = encode_masked(v.info.op, v.imm8, v.out, v.in0, v.in1, v.off, v.mode);
        self.emit_sanitized_load_immediate(REGISTER_SCRATCH, word as i64);
        self.emit_rust_call(
            Value::Constant64(vector_slow_path::<C> as *const u8 as i64, false),
            &[
                Argument {
                    index: 3,
                    value: Value::Register(REGISTER_SCRATCH),
                },
                Argument {
                    index: 2,
                    value: Value::Register(REGISTER_MAP[2]),
                },
                Argument {
                    index: 1,
                    value: Value::Register(REGISTER_MAP[1]),
                },
                Argument {
                    index: 0,
                    value: Value::Register(REGISTER_PTR_TO_VM),
                },
            ],
            None,
        );
        self.emit_check_result_is_ok();
        self.emit_ins(X86Instruction::load_immediate(REGISTER_SCRATCH, pc as i64));
        self.emit_ins(X86Instruction::conditional_jump_immediate(
            0x85,
            self.relative_to_anchor(ANCHOR_THROW_EXCEPTION, 6),
        ));
    }

    /// Gathers and scatters
    ///
    /// Fast path: if the addresses of all active lanes lie in one plain region, the native
    /// instruction runs on its host address. The span is [lo, hi) with lo = base + min index
    /// << scale and hi = base + max index << scale + element size, where the minimum and maximum
    /// are over the active lanes. Qword indices outside of the i32 range take the slow path,
    /// so these computations do not overflow. Otherwise the slow path does the per lane model.
    /// Overlapping lanes of scatters are written in ascending order by the hardware, as in the
    /// model.
    fn emit_gather_scatter(&mut self, v: &VectorInsn, lowering: &Lowering) {
        let (qword_index, element, lanes) = gather_scatter_shape(v.info.op);
        let gather = v.info.kind == Kind::Ga;
        let mut empty = None;
        if v.mode == 0 {
            self.emit_bytes(&kxnorq(K3, K3, K3));
        } else {
            self.emit_bytes(&kmovq_from_gpr(K3, REGISTER_MAP[2]));
            self.emit_bytes(&kortest(lanes, K3, K3));
            empty = Some(self.offset_in_text_section + 2);
            self.emit_ins(X86Instruction::conditional_jump_immediate(0x84, 0)); // jz empty
        }
        let mut done = vec![];
        if let Some(span_translation) = self.inline_span_translation() {
            let key = self.memory_run_key();
            for (reg, offset) in GATHER_SCATTER_SAVED_REGISTERS {
                self.emit_ins(X86Instruction::store(
                    OperandSize::S64,
                    reg,
                    RSP,
                    X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
                ));
            }
            // RAX = min index, RCX = max index
            for (maximum, reg) in [(false, RAX), (true, RCX)] {
                for instruction in gather_scatter_bound(v, maximum, reg) {
                    self.emit_host_instruction(&instruction);
                }
            }
            let mut slow_path = vec![];
            for reg in [RAX, RCX] {
                if qword_index {
                    self.emit_ins(X86Instruction::mov_with_sign_extension(
                        OperandSize::S64,
                        reg,
                        REGISTER_SCRATCH,
                    ));
                    self.emit_ins(X86Instruction::cmp(
                        OperandSize::S64,
                        REGISTER_SCRATCH,
                        reg,
                        None,
                    ));
                    slow_path.push(self.offset_in_text_section + 2);
                    self.emit_ins(X86Instruction::conditional_jump_immediate(0x85, 0));
                // jne
                } else {
                    self.emit_ins(X86Instruction::mov_with_sign_extension(
                        OperandSize::S64,
                        reg,
                        reg,
                    ));
                }
                if v.imm8 != 0 {
                    self.emit_ins(X86Instruction::alu_immediate(
                        OperandSize::S64,
                        0xc1,
                        4,
                        reg,
                        v.imm8 as i64,
                        None,
                    ));
                }
            }
            // REGISTER_SCRATCH = lo, span length - 1 + key on the stack
            self.emit_r1_plus_off(v.off);
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x01,
                RAX,
                REGISTER_SCRATCH,
                None,
            ));
            self.emit_ins(X86Instruction::alu(OperandSize::S64, 0x29, RAX, RCX, None));
            self.emit_ins(X86Instruction::alu_immediate(
                OperandSize::S64,
                0x81,
                0,
                RCX,
                element as i64 - 1 + key,
                None,
            ));
            self.emit_ins(X86Instruction::store(
                OperandSize::S64,
                RCX,
                RSP,
                X86IndirectAccess::OffsetIndexShift(-16, RSP, 0),
            ));
            self.emit_ins(X86Instruction::call_immediate(
                self.relative_to_offset(span_translation[!gather as usize], 5),
            ));
            self.emit_ins(X86Instruction::test(
                OperandSize::S64,
                REGISTER_SCRATCH,
                REGISTER_SCRATCH,
                None,
            ));
            slow_path.push(self.offset_in_text_section + 2);
            self.emit_ins(X86Instruction::conditional_jump_immediate(0x84, 0)); // jz
                                                                                // REGISTER_SCRATCH + key = host address of index 0
            self.emit_ins(X86Instruction::alu(
                OperandSize::S64,
                0x29,
                RAX,
                REGISTER_SCRATCH,
                None,
            ));
            self.emit_restore_gather_scatter_registers();
            // The destination of a gather must differ from the index
            let destination = if gather && v.out == v.in0 {
                if v.mode == 1 {
                    self.emit_host_instruction(&HostInstruction::Evex(EvexInstruction::rr(
                        P66, M0F, true, 0x6f, ZMM30, v.out,
                    ))); // vmovdqa64
                }
                ZMM30
            } else {
                v.out
            };
            if gather && v.mode == 2 {
                self.emit_host_instruction(&HostInstruction::Evex(EvexInstruction::rrr(
                    P66,
                    M0F,
                    false,
                    0xef,
                    destination,
                    destination,
                    destination,
                ))); // vpxord
            }
            let native = match lowering.instructions[0] {
                HostInstruction::Evex(evex) if gather => HostInstruction::Evex(EvexInstruction {
                    reg: destination,
                    ..evex
                }),
                other => other,
            };
            self.emit_displaced(
                &Lowering {
                    memory: None,
                    instructions: vec![native],
                },
                key as i32,
            );
            if destination != v.out {
                self.emit_host_instruction(&HostInstruction::Evex(EvexInstruction::rr(
                    P66, M0F, true, 0x6f, v.out, ZMM30,
                ))); // vmovdqa64
            }
            done.push(self.offset_in_text_section + 1);
            self.emit_ins(X86Instruction::jump_immediate(0));
            for jump in slow_path {
                self.patch_rel32(jump);
            }
            self.emit_restore_gather_scatter_registers();
        }
        self.emit_vector_slow_path_call(v, self.pc);
        if let Some(empty) = empty {
            done.push(self.offset_in_text_section + 1);
            self.emit_ins(X86Instruction::jump_immediate(0));
            self.patch_rel32(empty);
            if gather && v.mode == 2 {
                self.emit_host_instruction(&HostInstruction::Evex(EvexInstruction::rrr(
                    P66, M0F, false, 0xef, v.out, v.out, v.out,
                ))); // vpxord
            } else if gather && lanes * element < VECTOR_SIZE {
                // gather.qd: the bits above the result are 0 in every mode
                self.emit_host_instruction(&HostInstruction::Evex(
                    EvexInstruction::rr(P66, M0F, false, 0x6f, v.out, v.out).length(Length::L256),
                )); // vmovdqa32
            }
        }
        for jump in done {
            self.patch_rel32(jump);
        }
    }

    fn emit_restore_gather_scatter_registers(&mut self) {
        for (reg, offset) in GATHER_SCATTER_SAVED_REGISTERS {
            self.emit_ins(X86Instruction::load(
                OperandSize::S64,
                RSP,
                reg,
                X86IndirectAccess::OffsetIndexShift(offset, RSP, 0),
            ));
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
        let mut worst_case_length = 0;
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
            // Gathers and scatters use REGISTER_SCRATCH
            if matches!(vector_insn.info.kind, Kind::Ga | Kind::Sc) {
                break;
            }
            worst_case_length += max_machine_code_length(&vector_insn, self.vector_registers_used);
            if let Some((access_type, len)) = memory_access(&vector_insn) {
                end = pc;
                accesses += 1;
                stores |= access_type == AccessType::Store;
                min_off = min_off.min(vector_insn.off as i64);
                max_end = max_end.max(vector_insn.off as i64 + len as i64);
            }
            pc += 1;
        }
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
        self.emit_displaced(lowering, displacement);
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
            self.emit_vector_memory_access(&fallback.insn, &lowering, fallback.pc);
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
        vector::{encode, encode_with_offset, OPS},
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
            op::MPOPCNT => vec!["popcnt rax, rdx".into()],
            op::MTZCNT => vec!["tzcnt rax, rdx".into()],
            op::MLZCNT => vec!["lzcnt rax, rdx".into()],
            op::BLENDM => vec![
                "kmovq k2, rdx".into(),
                format!("vpblendm{} zmm{o} {{k2}}, zmm{a}, zmm{b}", width(i)),
            ],
            op::CMP => {
                let predicate = ["eq", "lt", "le", "", "neq", "nlt", "nle"][(i & 7) as usize];
                let unsigned = if i & 8 != 0 { "u" } else { "" };
                vec![
                    format!(
                        "vpcmp{predicate}{unsigned}{} k1, zmm{a}, zmm{b}",
                        width(i >> 4)
                    ),
                    "kmovq rax, k1".into(),
                ]
            }
            op::BITSHUF => vec![
                format!("vpshufbitqmb k1, zmm{a}, zmm{b}"),
                "kmovq rax, k1".into(),
            ],
            op::MUL52LO | op::MUL52HI => {
                let lo = if v.info.op == op::MUL52LO { "l" } else { "h" };
                let t = if v.mode == 1 || o == a || o == b {
                    30
                } else {
                    o
                };
                let mut lines = vec![
                    format!("vpxorq zmm{t}, zmm{t}, zmm{t}"),
                    format!("vpmadd52{lo}uq zmm{t}, zmm{a}, zmm{b}"),
                ];
                if t != o {
                    lines.push(format!("vmovdqa64 zmm{o}, zmm30"));
                }
                lines
            }
            op::ABS_Q => unary("vpabsq", "zmm"),
            op::MAX_Q => binary("vpmaxsq"),
            op::MIN_Q => binary("vpminsq"),
            op::ABS_B => unary("vpabsb", "zmm"),
            op::ABS_W => unary("vpabsw", "zmm"),
            op::ABS_D => unary("vpabsd", "zmm"),
            op::MAX_B => binary("vpmaxsb"),
            op::MIN_B => binary("vpminsb"),
            op::MAXU_B => binary("vpmaxub"),
            op::MINU_B => binary("vpminub"),
            op::ADDS_B => binary("vpaddsb"),
            op::ADDS_W => binary("vpaddsw"),
            op::ADDUS_B => binary("vpaddusb"),
            op::ADDUS_W => binary("vpaddusw"),
            op::SUBS_B => binary("vpsubsb"),
            op::SUBS_W => binary("vpsubsw"),
            op::SUBUS_B => binary("vpsubusb"),
            op::SUBUS_W => binary("vpsubusw"),
            op::AVG_B => binary("vpavgb"),
            op::AVG_W => binary("vpavgw"),
            op::SAD => binary("vpsadbw"),
            op::MULHRS_W => binary("vpmulhrsw"),
            op::DPBUSDS => binary("vpdpbusds"),
            op::DPWSSDS => binary("vpdpwssds"),
            op::TEST | op::TESTN => vec![
                format!(
                    "vptest{}m{} k1, zmm{a}, zmm{b}",
                    if v.info.op == op::TESTN { "n" } else { "" },
                    width(i)
                ),
                "kmovq rax, k1".into(),
            ],
            op::MOVEMASK_W | op::MOVEMASK_D | op::MOVEMASK_Q => vec![
                format!(
                    "vpmov{}2m k1, zmm{a}",
                    match v.info.op {
                        op::MOVEMASK_W => "w",
                        op::MOVEMASK_D => "d",
                        _ => "q",
                    }
                ),
                "kmovq rax, k1".into(),
            ],
            op::MOVM_B | op::MOVM_W | op::MOVM_D | op::MOVM_Q => vec![
                "kmovq k2, rdx".into(),
                format!("vpmovm2{} zmm{o}, k2", width(v.info.op - op::MOVM_B)),
            ],
            op::TERNLOG_D => vec![format!("vpternlogd zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::SHLV_W => binary("vpsllvw"),
            op::SHRV_W => binary("vpsrlvw"),
            op::SARV_W => binary("vpsravw"),
            op::SARV_D => binary("vpsravd"),
            op::SARV_Q => binary("vpsravq"),
            op::SHLDI_W | op::SHLDI_D | op::SHLDI_Q => vec![format!(
                "vpshld{} zmm{o}, zmm{a}, zmm{b}, {i}",
                width(v.info.op - op::SHLDI_W + 1)
            )],
            op::SHRDI_W | op::SHRDI_D | op::SHRDI_Q => vec![format!(
                "vpshrd{} zmm{o}, zmm{a}, zmm{b}, {i}",
                width(v.info.op - op::SHRDI_W + 1)
            )],
            op::SHLDV_W | op::SHLDV_D | op::SHLDV_Q => {
                binary(&format!("vpshldv{}", width(v.info.op - op::SHLDV_W + 1)))
            }
            op::SHRDV_W | op::SHRDV_D | op::SHRDV_Q => {
                binary(&format!("vpshrdv{}", width(v.info.op - op::SHRDV_W + 1)))
            }
            op::CONFLICT_D => unary("vpconflictd", "zmm"),
            op::CONFLICT_Q => unary("vpconflictq", "zmm"),
            op::LZCNT_D => unary("vplzcntd", "zmm"),
            op::LZCNT_Q => unary("vplzcntq", "zmm"),
            op::POPCNT_W => unary("vpopcntw", "zmm"),
            op::POPCNT_D => unary("vpopcntd", "zmm"),
            op::MULTISHIFT => binary("vpmultishiftqb"),
            op::GF2P8MUL => binary("vgf2p8mulb"),
            op::GF2P8AFFINE => vec![format!("vgf2p8affineqb zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::GF2P8AFFINEINV => vec![format!("vgf2p8affineinvqb zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::AESENC => binary("vaesenc"),
            op::AESENCLAST => binary("vaesenclast"),
            op::AESDEC => binary("vaesdec"),
            op::AESDECLAST => binary("vaesdeclast"),
            op::SHA256RNDS2
            | op::SHA256MSG1
            | op::SHA256MSG2
            | op::SHA1RNDS4
            | op::SHA1NEXTE
            | op::SHA1MSG1
            | op::SHA1MSG2 => lower(v)
                .instructions
                .iter()
                .map(|instruction| match instruction {
                    HostInstruction::Evex(e) if e.opcode == 0x38 => format!(
                        "vinserti32x4 zmm{}, zmm{}, xmm{}, {}",
                        e.reg,
                        e.vvvv,
                        match e.rm {
                            Rm::Reg(reg) => reg,
                            _ => unreachable!(),
                        },
                        e.imm8.unwrap()
                    ),
                    HostInstruction::Evex(e) if e.opcode == 0x39 => format!(
                        "vextracti32x4 xmm{}, zmm{}, {}",
                        match e.rm {
                            Rm::Reg(reg) => reg,
                            _ => unreachable!(),
                        },
                        e.reg,
                        e.imm8.unwrap()
                    ),
                    HostInstruction::Legacy(l) => match v.info.op {
                        op::SHA256RNDS2 => format!("sha256rnds2 xmm{}, xmm{}, xmm0", l.reg, l.rm),
                        op::SHA1RNDS4 => format!("sha1rnds4 xmm{}, xmm{}, {i}", l.reg, l.rm),
                        _ => format!("{} xmm{}, xmm{}", v.info.name, l.reg, l.rm),
                    },
                    _ => unreachable!(),
                })
                .collect(),
            op::GATHER_QQ => vec![format!(
                "vpgatherqq zmm{o} {{k3}}, zmmword ptr [r11 + {}zmm{a}]",
                scale(i)
            )],
            op::SCATTER_QQ => vec![format!(
                "vpscatterqq zmmword ptr [r11 + {}zmm{a}] {{k3}}, zmm{b}",
                scale(i)
            )],
            op::GATHER_DD => vec![format!(
                "vpgatherdd zmm{o} {{k3}}, zmmword ptr [r11 + {}zmm{a}]",
                scale(i)
            )],
            op::GATHER_DQ => vec![format!(
                "vpgatherdq zmm{o} {{k3}}, zmmword ptr [r11 + {}ymm{a}]",
                scale(i)
            )],
            op::GATHER_QD => vec![format!(
                "vpgatherqd ymm{o} {{k3}}, ymmword ptr [r11 + {}zmm{a}]",
                scale(i)
            )],
            op::SCATTER_DD => vec![format!(
                "vpscatterdd zmmword ptr [r11 + {}zmm{a}] {{k3}}, zmm{b}",
                scale(i)
            )],
            op::SCATTER_DQ => vec![format!(
                "vpscatterdq zmmword ptr [r11 + {}ymm{a}] {{k3}}, zmm{b}",
                scale(i)
            )],
            op::SCATTER_QD => vec![format!(
                "vpscatterqd ymmword ptr [r11 + {}zmm{a}] {{k3}}, ymm{b}",
                scale(i)
            )],
            op::LD_Y => vec![format!("vmovdqu64 ymm{o}, ymmword ptr [r11]")],
            op::ST_Y => vec![format!("vmovdqu64 ymmword ptr [r11], ymm{a}")],
            op::LD_X => vec![format!("vmovdqu32 xmm{o}, xmmword ptr [r11]")],
            op::ST_X => vec![format!("vmovdqu32 xmmword ptr [r11], xmm{a}")],
            op::LDBC_B => vec![format!("vpbroadcastb zmm{o}, byte ptr [r11]")],
            op::LDBC_W => vec![format!("vpbroadcastw zmm{o}, word ptr [r11]")],
            op::LDBC_X => vec![format!("vbroadcasti32x4 zmm{o}, xmmword ptr [r11]")],
            op::LDBC_Y => vec![format!("vbroadcasti64x4 zmm{o}, ymmword ptr [r11]")],
            op::LDN => (0..i)
                .map(|k| {
                    format!(
                        "vmovdqu64 zmm{}, zmmword ptr [r11{}]",
                        o + k,
                        displacement(k)
                    )
                })
                .collect(),
            op::STN => (0..i)
                .map(|k| {
                    format!(
                        "vmovdqu64 zmmword ptr [r11{}], zmm{}",
                        displacement(k),
                        a + k
                    )
                })
                .collect(),
            op::LD_M => vec![
                "kmovq k2, rdx".into(),
                format!(
                    "vmovdqu{} zmm{o} {{k2}}{}, zmmword ptr [r11]",
                    8 << i,
                    if v.mode == 2 { " {z}" } else { "" }
                ),
            ],
            op::ST_M => vec![
                "kmovq k2, rdx".into(),
                format!("vmovdqu{} zmmword ptr [r11] {{k2}}, zmm{a}", 8 << i),
            ],
            op::BCAST_Q | op::BCAST_D | op::BCAST_W | op::BCAST_B => {
                let (bytes, name) = match v.info.op {
                    op::BCAST_Q => (8, "q"),
                    op::BCAST_D => (4, "d"),
                    op::BCAST_W => (2, "w"),
                    _ => (1, "b"),
                };
                let mut lines = vec![];
                let mut source = a;
                if bytes >= 4 && i != 0 {
                    lines.push(format!("valign{name} zmm30, zmm{a}, zmm{a}, {i}"));
                    source = 30;
                } else if bytes < 4 {
                    let per_128 = 16 / bytes;
                    if i >= per_128 {
                        lines.push(format!("vextracti32x4 xmm30, zmm{a}, {}", i / per_128));
                        source = 30;
                    }
                    if i % per_128 != 0 {
                        lines.push(format!(
                            "vpsrldq xmm30, xmm{source}, {}",
                            i % per_128 * bytes
                        ));
                        source = 30;
                    }
                }
                lines.push(format!("vpbroadcast{name} zmm{o}, xmm{source}"));
                lines
            }
            op::PERMQI => shift("vpermq"),
            op::EXTRACT256 => vec![format!("vextracti64x4 ymm{o}, zmm{a}, {i}")],
            op::PERM2I => binary(&format!("vpermi2{}", width(i))),
            op::ALIGN_Q => vec![format!("valignq zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::ALIGN_D => vec![format!("valignd zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::EXTRACT128 => vec![format!("vextracti32x4 xmm{o}, zmm{a}, {i}")],
            op::INSERT256 => vec![format!("vinserti64x4 zmm{o}, zmm{a}, ymm{b}, {i}")],
            op::INSERT128 => vec![format!("vinserti32x4 zmm{o}, zmm{a}, xmm{b}, {i}")],
            op::PERM2T => binary(&format!("vpermt2{}", width(i))),
            op::PERMB => binary("vpermb"),
            op::PERMW => binary("vpermw"),
            op::SHUFD => shift("vpshufd"),
            op::SHUFHW => shift("vpshufhw"),
            op::SHUFLW => shift("vpshuflw"),
            op::SHUF128 => vec![format!("vshufi64x2 zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::SHUFPS => vec![format!("vshufps zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::ALIGNR => vec![format!("vpalignr zmm{o}, zmm{a}, zmm{b}, {i}")],
            op::BSHLI => shift("vpslldq"),
            op::BSHRI => shift("vpsrldq"),
            op::UNPACKLO => {
                binary(["vpunpcklbw", "vpunpcklwd", "vpunpckldq", "vpunpcklqdq"][i as usize])
            }
            op::UNPACKHI => {
                binary(["vpunpckhbw", "vpunpckhwd", "vpunpckhdq", "vpunpckhqdq"][i as usize])
            }
            op::COMPRESS => unary(&format!("vpcompress{}", width(i)), "zmm"),
            op::EXPAND => unary(&format!("vpexpand{}", width(i)), "zmm"),
            op::PACKUS_D => binary("vpackusdw"),
            op::NARROW_D_W => narrow("vpmovdw", "ymm"),
            op::NARROWUS_D_W => narrow("vpmovusdw", "ymm"),
            op::NARROW_Q_B => narrow("vpmovqb", "xmm"),
            op::NARROWS_Q_B => narrow("vpmovsqb", "xmm"),
            op::NARROWUS_Q_B => narrow("vpmovusqb", "xmm"),
            op::NARROW_Q_W => narrow("vpmovqw", "xmm"),
            op::NARROWS_Q_W => narrow("vpmovsqw", "xmm"),
            op::NARROWUS_Q_W => narrow("vpmovusqw", "xmm"),
            op::NARROW_Q_D => narrow("vpmovqd", "ymm"),
            op::NARROWS_Q_D => narrow("vpmovsqd", "ymm"),
            op::NARROWUS_Q_D => narrow("vpmovusqd", "ymm"),
            op::NARROW_W_B => narrow("vpmovwb", "ymm"),
            op::NARROWS_W_B => narrow("vpmovswb", "ymm"),
            op::NARROWUS_W_B => narrow("vpmovuswb", "ymm"),
            op::ZX_B_W => unary("vpmovzxbw", "ymm"),
            op::SX_B_W => unary("vpmovsxbw", "ymm"),
            op::ZX_B_Q => unary("vpmovzxbq", "xmm"),
            op::SX_B_Q => unary("vpmovsxbq", "xmm"),
            op::ZX_W_Q => unary("vpmovzxwq", "xmm"),
            op::SX_W_Q => unary("vpmovsxwq", "xmm"),
            op::EXTRACT_B | op::EXTRACT_W => {
                let (name, per_128) = if v.info.op == op::EXTRACT_B {
                    ("b", 16)
                } else {
                    ("w", 8)
                };
                let mut lines = vec![];
                let mut source = a;
                if i >= per_128 {
                    lines.push(format!("vextracti32x4 xmm30, zmm{a}, {}", i / per_128));
                    source = 30;
                }
                lines.push(format!("vpextr{name} eax, xmm{source}, {}", i % per_128));
                lines
            }
            op::INSERT_B | op::INSERT_W | op::INSERT_D => {
                let (name, per_128) = match v.info.op {
                    op::INSERT_B => ("b", 16),
                    op::INSERT_W => ("w", 8),
                    _ => ("d", 4),
                };
                vec![
                    format!("vextracti32x4 xmm30, zmm{o}, {}", i / per_128),
                    format!("vpinsr{name} xmm30, xmm30, esi, {}", i % per_128),
                    format!("vinserti32x4 zmm{o}, zmm{o}, xmm30, {}", i / per_128),
                ]
            }
            op::MOVR_Q => vec![format!("vmovq xmm{o}, rsi")],
            _ => unreachable!(),
        }
    }

    fn width(code: u8) -> &'static str {
        ["b", "w", "d", "q"][(code & 3) as usize]
    }

    fn scale(log2: u8) -> String {
        if log2 == 0 {
            String::new()
        } else {
            format!("{}*", 1 << log2)
        }
    }

    fn displacement(k: u8) -> String {
        if k == 0 {
            String::new()
        } else {
            format!(" + {}", 64 * k as usize)
        }
    }

    /// [expected_disassembly] including the mask mode
    fn expected_masked_disassembly(v: &VectorInsn) -> Vec<String> {
        let mut lines = expected_disassembly(v);
        if v.mode != 0
            && matches!(v.info.mask, Mask::Modes(..))
            && !matches!(v.info.kind, Kind::Lsi | Kind::Sti | Kind::Ga | Kind::Sc)
        {
            let last = lines.pop().unwrap();
            let (destination, rest) = last.split_once(',').unwrap();
            let mask = if v.mode == 2 { " {k2} {z}" } else { " {k2}" };
            lines.insert(0, "kmovq k2, rdx".into());
            lines.push(format!("{destination}{mask},{rest}"));
        }
        lines
    }

    /// Expected disassembly of [gather_scatter_bound]
    fn expected_bound_disassembly(v: &VectorInsn, maximum: bool, destination: &str) -> Vec<String> {
        let (q, _, lanes) = gather_scatter_shape(v.info.op);
        let (w, bits) = if q { ("q", 64) } else { ("d", 32) };
        let reduce = if maximum { "vpmaxs" } else { "vpmins" };
        let a = v.in0;
        let mut lines = vec![
            "vpternlogd zmm30, zmm30, zmm30, 255".to_string(),
            if maximum {
                format!("vpsll{w} zmm30, zmm30, {}", bits - 1)
            } else {
                format!("vpsrl{w} zmm30, zmm30, 1")
            },
        ];
        if q || lanes == 16 {
            lines.push(format!("vmovdqa{bits} zmm30 {{k3}}, zmm{a}"));
            lines.push("vextracti64x4 ymm31, zmm30, 1".into());
            lines.push(format!("{reduce}{w} ymm30, ymm30, ymm31"));
        } else {
            lines.push(format!("vmovdqa{bits} ymm30 {{k3}}, ymm{a}"));
        }
        lines.push("vextracti32x4 xmm31, ymm30, 1".into());
        lines.push(format!("{reduce}{w} xmm30, xmm30, xmm31"));
        for shuffle in if q { &[78][..] } else { &[78, 177][..] } {
            lines.push(format!("vpshufd xmm31, xmm30, {shuffle}"));
            lines.push(format!("{reduce}{w} xmm30, xmm30, xmm31"));
        }
        lines.push(if q {
            format!("vmovq {destination}, xmm30")
        } else {
            format!("vmovd {}, xmm30", destination.replace('r', "e"))
        });
        lines
    }

    /// Disassembles `code` with LLVM into one normalized line per instruction
    fn llvm_disassemble(code: &[Vec<u8>]) -> Option<Vec<String>> {
        let llvm_mc = ["llvm-mc-19", "llvm-mc"]
            .iter()
            .find(|tool| Command::new(tool).arg("--version").output().is_ok())?;
        let mut child = Command::new(llvm_mc)
            .args(["--disassemble", "-triple=x86_64", "--output-asm-variant=1", "-mattr=+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512vnni,+avx512ifma,+avx512bitalg,+avx512vpopcntdq,+vpclmulqdq,+avx512cd,+avx512vbmi,+avx512vbmi2,+gfni,+vaes,+sha,+popcnt,+bmi,+lzcnt"])
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
        for info in OPS.iter().filter(|info| info.implemented) {
            let kind = info.kind;
            let imm8s: Vec<u8> = (0..=255).filter(|i| info.imm8_is_valid(*i)).collect();
            for (k, (o, a, b)) in [
                (3, 4, 5),
                (29, 16, 8),
                (0, 15, 17),
                (24, 9, 1),
                (8, 23, 28),
                (7, 7, 7),
                (0, 0, 0),
                (20, 0, 21),
                (1, 2, 0),
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
                for mode in (0..3).filter(|mode| info.modes() & (1 << mode) != 0) {
                    // The first valid imm8 from a varying start (register ranges of ldn / stn)
                    let Some(v) = (0..imm8s.len())
                        .map(|j| imm8s[(k * 37 + j) % imm8s.len()])
                        .find_map(|imm8| {
                            VectorInsn::decode(&ebpf::get_insn(
                                &encode_masked(info.op, imm8, o, a, b, 0, mode).to_le_bytes(),
                                0,
                            ))
                            .ok()
                        })
                    else {
                        continue;
                    };
                    for instruction in lower(&v).instructions {
                        code.push(instruction.encode());
                    }
                    expected.extend(expected_masked_disassembly(&v));
                    if matches!(kind, Kind::Ga | Kind::Sc) {
                        for (maximum, reg, text) in [(false, RAX, "rax"), (true, RCX, "rcx")] {
                            for instruction in gather_scatter_bound(&v, maximum, reg) {
                                code.push(instruction.encode());
                            }
                            expected.extend(expected_bound_disassembly(&v, maximum, text));
                        }
                    }
                }
            }
        }
        // Mask register instructions of gathers and scatters
        for (bytes, text) in [
            (kxnorq(K3, K3, K3), "kxnorq k3, k3, k3"),
            (kortest(8, K3, K3), "kortestb k3, k3"),
            (kortest(16, K3, K3), "kortestw k3, k3"),
            (kmovq_from_gpr(K3, RDX), "kmovq k3, rdx"),
        ] {
            code.push(bytes.to_vec());
            expected.push(text.to_string());
        }
        // gather.qd with an empty mask in mode 1 clears the upper half
        code.push(
            HostInstruction::Evex(
                EvexInstruction::rr(P66, M0F, false, 0x6f, 3, 3).length(Length::L256),
            )
            .encode(),
        );
        expected.push("vmovdqa32 ymm3, ymm3".to_string());
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
        // The foundation EVEX additions: opmask, zeroing, the new prefixes, VSIB and kmovq k, r64
        let push = |code: &mut Vec<Vec<u8>>,
                    expected: &mut Vec<String>,
                    e: EvexInstruction,
                    text: &str| {
            let (bytes, len) = e.encode();
            code.push(bytes[..len].to_vec());
            expected.push(text.to_string());
        };
        push(
            &mut code,
            &mut expected,
            EvexInstruction::rrr(P66, M0F38, true, 0x64, 3, 4, 5).mask(2, false),
            "vpblendmq zmm3 {k2}, zmm4, zmm5",
        ); // merge masking
        push(
            &mut code,
            &mut expected,
            EvexInstruction::rrr(P66, M0F, false, 0xfe, 7, 8, 9).mask(1, true),
            "vpaddd zmm7 {k1} {z}, zmm8, zmm9",
        ); // zeroing
        push(
            &mut code,
            &mut expected,
            EvexInstruction::rrr(crate::x86_evex::Prefix::None, M0F, false, 0xc6, 3, 4, 5)
                .imm(0x1b),
            "vshufps zmm3, zmm4, zmm5, 27",
        ); // no prefix (pp = 00)
        push(
            &mut code,
            &mut expected,
            EvexInstruction::rr(PF2, M0F, false, 0x6f, 3, 4).mask(2, true),
            "vmovdqu8 zmm3 {k2} {z}, zmm4",
        ); // F2 prefix
        push(
            &mut code,
            &mut expected,
            EvexInstruction::rr(P66, M0F38, true, 0x91, 3, 0)
                .vsib(RDI, 5, 3, 0)
                .mask(2, false),
            "vpgatherqq zmm3 {k2}, zmmword ptr [rdi + 8*zmm5]",
        ); // VSIB
        push(
            &mut code,
            &mut expected,
            EvexInstruction::rr(P66, M0F38, true, 0x91, 20, 0)
                .vsib(R12, 17, 2, 64)
                .mask(3, false),
            "vpgatherqq zmm20 {k3}, zmmword ptr [r12 + 4*zmm17 + 64]",
        ); // VSIB, index >= 16
        for (mask, reg, text) in [(2u8, RSI, "kmovq k2, rsi"), (5, R9, "kmovq k5, r9")] {
            code.push(crate::x86_evex::kmovq_from_gpr(mask, reg).to_vec());
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
        eprintln!("{} instructions checked", expected.len());
        assert_eq!(disassembly.len(), expected.len());
        for (actual, expected) in disassembly.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }

    /// Every op has a lowering and only the memory ops translate an address
    #[test]
    fn test_lower_all_ops() {
        for info in OPS.iter().filter(|info| info.implemented) {
            let kind = info.kind;
            let imm8 = (0..=255).find(|i| info.imm8_is_valid(*i)).unwrap();
            let mode = info.modes().trailing_zeros() as u8;
            let lowering = lower(&decode(encode_masked(
                info.op,
                imm8,
                kind.uses_out() as u8,
                kind.uses_in0() as u8,
                kind.uses_in1() as u8,
                0,
                mode,
            )));
            assert!(!lowering.instructions.is_empty());
            assert_eq!(
                lowering.memory.is_some(),
                info.accesses_memory() && !matches!(kind, Kind::Ga | Kind::Sc),
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
