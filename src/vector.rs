#![allow(clippy::arithmetic_side_effects)]
//! 512-bit vector ops
//!
//! A vector op is a `CALL_IMM` with `src = 2`. `dst` is the mask mode (0 = unmasked, 1 = merge,
//! 2 = zero, with the mask in `r2`). `off` is a signed displacement added to `r1` by the memory
//! ops, and must be 0 for every other op. The immediate holds:
//!
//! ```text
//! imm[31:24] op
//! imm[23:16] imm8
//! imm[15]    0
//! imm[14:10] out
//! imm[9:5]   in0
//! imm[4:0]   in1
//! ```
//!
//! Vector ops are only recognized if [`Config::enable_vector_ops`](crate::vm::Config) is set and
//! the SBPF version has static syscalls. This module contains the decoder, the verifier checks
//! and the portable reference semantics used by the interpreter. The x86-64 JIT lowering lives in
//! `vector_jit.rs`.

use crate::{
    ebpf,
    error::{EbpfError, StableResult},
    memory_region::AccessType,
    program::SBPFVersion,
    verifier::VerifierError,
    vm::{ContextObject, EbpfVm},
};

/// `CALL_IMM` source register value which marks a vector op
pub const VECTOR_OP_SRC: u8 = 2;
/// Number of vector registers
pub const VECTOR_REGISTERS: usize = 30;
/// Size of a vector register in bytes
pub const VECTOR_SIZE: usize = 64;

/// One 512-bit vector register, lanes are little-endian
#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VectorRegister(pub [u8; VECTOR_SIZE]);

impl Default for VectorRegister {
    fn default() -> Self {
        Self::ZERO
    }
}

impl VectorRegister {
    /// All bits cleared
    pub const ZERO: Self = Self([0; VECTOR_SIZE]);

    /// Reads lane `i` of width `W` bytes, zero extended
    #[inline]
    pub fn lane<const W: usize>(&self, i: usize) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[..W].copy_from_slice(&self.0[i * W..(i + 1) * W]);
        u64::from_le_bytes(bytes)
    }

    /// Writes the low `W` bytes of `value` to lane `i`
    #[inline]
    pub fn set_lane<const W: usize>(&mut self, i: usize, value: u64) {
        self.0[i * W..(i + 1) * W].copy_from_slice(&value.to_le_bytes()[..W]);
    }
}

/// The vector register file of a VM
pub type VectorRegisterFile = [VectorRegister; VECTOR_REGISTERS];

/// Which instruction fields an op uses (see the kinds table in the spec)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// out ← f(in0, in1)
    R3,
    /// out ← f(out, in0, in1)
    A3,
    /// out ← f(out, in0, in1, imm8)
    A3i,
    /// out ← f(in0, in1, imm8)
    R3i,
    /// out ← f(in0)
    R2,
    /// out ← f(in0, imm8)
    R2i,
    /// out ← constant
    R0,
    /// out ← f(r1) (load or splat from a scalar)
    Ls,
    /// mem\[r1\] ← in0
    St,
    /// r0 ← f(in0)
    Sr,
    /// r0 ← f(in0, imm8)
    Sri,
    /// out ← f(out, r1, imm8)
    Ai,
    /// r0 ← f(in0, in1)
    S3,
    /// r0 ← f(in0, in1, imm8)
    S3i,
    /// out ← f(out, in0)
    A2,
    /// out ← f(out, in0, imm8)
    A2i,
    /// out ← mem\[r1\] (`ldn`: out..out+n-1), imm8 per op
    Lsi,
    /// mem\[r1\] ← in0 (`stn`: in0..in0+n-1), imm8 per op
    Sti,
    /// Gather: out ← mem\[r1 + in0 << imm8\]
    Ga,
    /// Scatter: mem\[r1 + in0 << imm8\] ← in1
    Sc,
    /// r0 ← f(r2)
    S0,
}

impl Kind {
    /// Whether `out` names a register (read or written)
    pub const fn uses_out(self) -> bool {
        !matches!(
            self,
            Kind::St
                | Kind::Sr
                | Kind::Sri
                | Kind::S3
                | Kind::S3i
                | Kind::Sti
                | Kind::Sc
                | Kind::S0
        )
    }
    /// Whether `out` is also read (without masking)
    pub const fn reads_out(self) -> bool {
        matches!(self, Kind::A3 | Kind::A3i | Kind::Ai | Kind::A2 | Kind::A2i)
    }
    /// Whether `in0` is read
    pub const fn uses_in0(self) -> bool {
        matches!(
            self,
            Kind::R3
                | Kind::A3
                | Kind::A3i
                | Kind::R3i
                | Kind::R2
                | Kind::R2i
                | Kind::St
                | Kind::Sr
                | Kind::Sri
                | Kind::S3
                | Kind::S3i
                | Kind::A2
                | Kind::A2i
                | Kind::Sti
                | Kind::Ga
                | Kind::Sc
        )
    }
    /// Whether `in1` is read
    pub const fn uses_in1(self) -> bool {
        matches!(
            self,
            Kind::R3 | Kind::A3 | Kind::A3i | Kind::R3i | Kind::S3 | Kind::S3i | Kind::Sc
        )
    }
    /// Whether imm8 is used
    pub const fn uses_imm8(self) -> bool {
        matches!(
            self,
            Kind::A3i
                | Kind::R3i
                | Kind::R2i
                | Kind::Sri
                | Kind::Ai
                | Kind::S3i
                | Kind::A2i
                | Kind::Lsi
                | Kind::Sti
                | Kind::Ga
                | Kind::Sc
        )
    }
    /// Whether the result is written to `r0`
    pub const fn writes_r0(self) -> bool {
        matches!(self, Kind::Sr | Kind::Sri | Kind::S3 | Kind::S3i | Kind::S0)
    }
    /// Whether `r1` is read
    pub const fn reads_r1(self) -> bool {
        matches!(
            self,
            Kind::Ls | Kind::St | Kind::Ai | Kind::Lsi | Kind::Sti | Kind::Ga | Kind::Sc
        )
    }
}

/// Valid values of imm8
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Imm {
    /// imm8 must be 0
    Zero,
    /// Any imm8
    Any,
    /// imm8 < n
    Below(u8),
    /// imm8 ∈ {0x00, 0x01, 0x10, 0x11}
    Clmul,
    /// imm8 is a lane width code (0 = b, 1 = w, 2 = d, 3 = q) in the given bit set
    Widths(u8),
    /// imm8\[2:0\] predicate (not 3 or 7), imm8\[3\] unsigned, imm8\[5:4\] width, imm8\[7:6\] = 0
    Cmp,
    /// Register count n of `ldn` / `stn`: 2 ≤ n and base + n ≤ 30
    Count,
}

/// Lane width a mask bit applies to
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Granularity {
    /// u8 lanes
    B,
    /// u16 lanes
    W,
    /// u32 lanes
    D,
    /// u64 lanes
    Q,
    /// The width code in imm8
    Imm8,
}

/// How an op uses the mask in `r2`
///
/// The `dst` field of a vector op is its mask mode: 0 = unmasked, 1 = merge (lane i of the result
/// is kept if bit i of `r2` is set, else the old `out` lane), 2 = zero (else 0).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mask {
    /// Mode 0 only, `r2` is not read
    None,
    /// Mode 0 only, `r2` is an operand of the op
    Data,
    /// Bit set of the allowed modes and the lane width of the mask bits
    Modes(u8, Granularity),
}

/// CU charged by an op
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cu {
    /// A constant
    Fixed(u8),
    /// imm8 (`ldn`, `stn`)
    Imm8,
    /// See [op_cu]
    PerOp,
}

/// Static description of an op
#[derive(Clone, Copy, Debug)]
pub struct OpInfo {
    /// Op code (imm\[31:24\])
    pub op: u8,
    /// Mnemonic
    pub name: &'static str,
    /// Which fields the op uses
    pub kind: Kind,
    /// Use of the mask
    pub mask: Mask,
    /// Whether the op is implemented. Reserved ops are rejected by the verifier.
    pub implemented: bool,
    imm: Imm,
    cu: Cu,
}

impl OpInfo {
    /// Whether `imm8` is valid for this op
    pub const fn imm8_is_valid(&self, imm8: u8) -> bool {
        match self.imm {
            Imm::Zero => imm8 == 0,
            Imm::Any => true,
            Imm::Below(n) => imm8 < n,
            Imm::Clmul => matches!(imm8, 0x00 | 0x01 | 0x10 | 0x11),
            Imm::Widths(set) => imm8 < 4 && (set >> imm8) & 1 == 1,
            Imm::Cmp => imm8 >> 6 == 0 && !matches!(imm8 & 7, 3 | 7),
            Imm::Count => 2 <= imm8 && imm8 as usize <= VECTOR_REGISTERS,
        }
    }

    /// Whether imm8 is a register count (`ldn`, `stn`)
    pub const fn imm8_is_count(&self) -> bool {
        matches!(self.imm, Imm::Count)
    }

    /// Bit set of the allowed mask modes
    pub const fn modes(&self) -> u8 {
        match self.mask {
            Mask::None | Mask::Data => 1,
            Mask::Modes(modes, _) => modes,
        }
    }

    /// Lane width in bytes a mask bit applies to
    pub const fn mask_lane_bytes(&self, imm8: u8) -> usize {
        match self.mask {
            Mask::Modes(_, Granularity::B) => 1,
            Mask::Modes(_, Granularity::W) => 2,
            Mask::Modes(_, Granularity::D) => 4,
            Mask::Modes(_, Granularity::Imm8) => 1 << (imm8 & 3),
            _ => 8,
        }
    }

    /// CU charged by the op
    pub fn cu(&self, imm8: u8) -> u64 {
        match self.cu {
            Cu::Fixed(cu) => cu as u64,
            Cu::Imm8 => imm8 as u64,
            Cu::PerOp => op_cu(self.op, imm8),
        }
    }

    /// Whether the op accesses memory at `r1 + off`
    pub const fn accesses_memory(&self) -> bool {
        matches!(self.kind, Kind::Lsi | Kind::Sti | Kind::Ga | Kind::Sc)
            || matches!(
                self.op,
                op::LD
                    | op::ST
                    | op::LDBC_D
                    | op::LDBC_Q
                    | op::LD_Y
                    | op::ST_Y
                    | op::LD_X
                    | op::ST_X
                    | op::LDBC_B
                    | op::LDBC_W
                    | op::LDBC_X
                    | op::LDBC_Y
            )
    }
}

/// CU of the ops with [Cu::PerOp]: the instructions of the lowering
fn op_cu(op: u8, imm8: u8) -> u64 {
    match op {
        op::BCAST_Q | op::BCAST_D => 1 + (imm8 != 0) as u64,
        op::BCAST_W => 1 + (imm8 >= 8) as u64 + (imm8 & 7 != 0) as u64,
        _ => 1 + (imm8 >= 16) as u64 + (imm8 & 15 != 0) as u64,
    }
}

/// Size in bytes of the result of ops whose result is narrower than a register. The mask applies
/// to the lanes of the result, the bytes above it are 0 in every mask mode.
fn result_bytes(op: u8, imm8: u8) -> usize {
    match op {
        op::NARROWUS_D_B | op::NARROWS_D_B | op::NARROW_D_B => 16,
        op::NARROWS_D_W => 32,
        op::EXTRACT256
        | op::NARROW_D_W
        | op::NARROWUS_D_W
        | op::NARROW_Q_D
        | op::NARROWS_Q_D
        | op::NARROWUS_Q_D
        | op::NARROW_W_B
        | op::NARROWS_W_B
        | op::NARROWUS_W_B => 32,
        op::EXTRACT128 | op::NARROW_Q_W | op::NARROWS_Q_W | op::NARROWUS_Q_W => 16,
        op::NARROW_Q_B | op::NARROWS_Q_B | op::NARROWUS_Q_B => 8,
        _ => {
            let _ = imm8;
            VECTOR_SIZE
        }
    }
}

macro_rules! ops {
    ($($($pending:ident:)? $code:literal $name:ident $mnemonic:literal $kind:ident, $imm:expr, $mask:expr $(, $cu:expr)?;)*) => {
        /// Op codes
        #[allow(missing_docs)]
        pub mod op {
            $(pub const $name: u8 = $code;)*
        }
        /// All defined ops, including the reserved ones (see [OpInfo::implemented])
        pub const OPS: &[OpInfo] = &[
            $(OpInfo {
                op: $code,
                name: $mnemonic,
                kind: Kind::$kind,
                mask: $mask,
                implemented: ops!(@implemented $($pending)?),
                imm: $imm,
                cu: ops!(@cu $($cu)?),
            },)*
        ];
    };
    (@implemented) => { true };
    (@implemented pending) => { false };
    (@cu) => { Cu::Fixed(1) };
    (@cu $cu:expr) => { $cu };
}

const NONE: Mask = Mask::None;
const R2: Mask = Mask::Data;
/// Modes 0, 1 and 2
const fn m(granularity: Granularity) -> Mask {
    Mask::Modes(0b111, granularity)
}
/// Modes 1 and 2: the mask is the purpose of the op
const fn mreq(granularity: Granularity) -> Mask {
    Mask::Modes(0b110, granularity)
}
/// Modes 0 and 1 (scatter)
const fn m01(granularity: Granularity) -> Mask {
    Mask::Modes(0b011, granularity)
}
/// Mode 1 only (masked store)
const fn m1(granularity: Granularity) -> Mask {
    Mask::Modes(0b010, granularity)
}

use {
    Granularity::{Imm8, B, D, Q, W},
    Imm::{Any, Below, Clmul, Cmp, Count, Widths, Zero},
};
ops! {
    0x00 LD "ld" Ls, Zero, NONE;
    0x01 ST "st" St, Zero, NONE;
    0x02 LDBC_D "ldbc.d" Ls, Zero, m(D);
    0x03 LDBC_Q "ldbc.q" Ls, Zero, m(Q);
    0x04 SPLAT_B "splat.b" Ls, Zero, m(B);
    0x05 SPLAT_W "splat.w" Ls, Zero, m(W);
    0x06 SPLAT_D "splat.d" Ls, Zero, m(D);
    0x07 SPLAT_Q "splat.q" Ls, Zero, m(Q);
    0x08 ZERO "zero" R0, Zero, NONE;
    0x09 MOV "mov" R2, Zero, NONE;
    0x10 ADD_B "add.b" R3, Zero, m(B);
    0x11 ADD_W "add.w" R3, Zero, m(W);
    0x12 ADD_D "add.d" R3, Zero, m(D);
    0x13 ADD_Q "add.q" R3, Zero, m(Q);
    0x14 SUB_B "sub.b" R3, Zero, m(B);
    0x15 SUB_W "sub.w" R3, Zero, m(W);
    0x16 SUB_D "sub.d" R3, Zero, m(D);
    0x17 SUB_Q "sub.q" R3, Zero, m(Q);
    0x18 MULLO_W "mullo.w" R3, Zero, m(W);
    0x19 MULLO_D "mullo.d" R3, Zero, m(D);
    0x1a MULLO_Q "mullo.q" R3, Zero, m(Q);
    0x1b MULWU "mulwu" R3, Zero, m(Q);
    0x1c MULWS "mulws" R3, Zero, m(Q);
    0x1d MULHI_W "mulhi.w" R3, Zero, m(W);
    0x1e MULHIU_W "mulhiu.w" R3, Zero, m(W);
    0x20 MAX_D "max.d" R3, Zero, m(D);
    0x21 MIN_D "min.d" R3, Zero, m(D);
    0x22 MAXU_D "maxu.d" R3, Zero, m(D);
    0x23 MINU_D "minu.d" R3, Zero, m(D);
    0x24 MAX_W "max.w" R3, Zero, m(W);
    0x25 MIN_W "min.w" R3, Zero, m(W);
    0x26 MAXU_Q "maxu.q" R3, Zero, m(Q);
    0x27 MINU_Q "minu.q" R3, Zero, m(Q);
    0x28 AND "and" R3, Zero, m(Q);
    0x29 OR "or" R3, Zero, m(Q);
    0x2a XOR "xor" R3, Zero, m(Q);
    0x2b ANDN "andn" R3, Zero, m(Q);
    0x2c TERNLOG "ternlog" A3i, Any, m(Q);
    0x2e MAXU_W "maxu.w" R3, Zero, m(W);
    0x2f MINU_W "minu.w" R3, Zero, m(W);
    0x30 SHLI_W "shli.w" R2i, Below(16), m(W);
    0x31 SHLI_D "shli.d" R2i, Below(32), m(D);
    0x32 SHLI_Q "shli.q" R2i, Below(64), m(Q);
    0x33 SHRI_W "shri.w" R2i, Below(16), m(W);
    0x34 SHRI_D "shri.d" R2i, Below(32), m(D);
    0x35 SHRI_Q "shri.q" R2i, Below(64), m(Q);
    0x36 SARI_W "sari.w" R2i, Below(16), m(W);
    0x37 SARI_D "sari.d" R2i, Below(32), m(D);
    0x38 SARI_Q "sari.q" R2i, Below(64), m(Q);
    0x39 ROLI_D "roli.d" R2i, Below(32), m(D);
    0x3a ROLI_Q "roli.q" R2i, Below(64), m(Q);
    0x3b SHLV_D "shlv.d" R3, Zero, m(D);
    0x3c SHLV_Q "shlv.q" R3, Zero, m(Q);
    0x3d SHRV_D "shrv.d" R3, Zero, m(D);
    0x3e SHRV_Q "shrv.q" R3, Zero, m(Q);
    0x3f ROLV_D "rolv.d" R3, Zero, m(D);
    0x40 ROLV_Q "rolv.q" R3, Zero, m(Q);
    0x48 CMPEQ_B "cmpeq.b" R3, Zero, NONE;
    0x49 CMPEQ_D "cmpeq.d" R3, Zero, NONE;
    0x4a CMPEQ_Q "cmpeq.q" R3, Zero, NONE;
    0x4b CMPGT_D "cmpgt.d" R3, Zero, NONE;
    0x4c CMPGT_Q "cmpgt.q" R3, Zero, NONE;
    0x4d CMPGTU_Q "cmpgtu.q" R3, Zero, NONE;
    0x50 DPBUSD "dpbusd" A3, Zero, m(D);
    0x51 DPWSSD "dpwssd" A3, Zero, m(D);
    0x52 MADDUBS "maddubs" R3, Zero, m(W);
    0x53 MADDWD "maddwd" R3, Zero, m(D);
    0x54 MADD52LO "madd52lo" A3, Zero, m(Q);
    0x55 MADD52HI "madd52hi" A3, Zero, m(Q);
    0x58 PACKS_D "packs.d" R3, Zero, m(W);
    0x59 PACKS_W "packs.w" R3, Zero, m(B);
    0x5a PACKUS_W "packus.w" R3, Zero, m(B);
    0x5b PERMD "permd" R3, Zero, m(D);
    0x5c PERMQ "permq" R3, Zero, m(Q);
    0x5d SHUFB "shufb" R3, Zero, m(B);
    0x5e PERM2_W "perm2.w" A3, Zero, m(W);
    0x60 NARROWUS_D_B "narrowus.d.b" R2, Zero, m(B);
    0x61 NARROWS_D_B "narrows.d.b" R2, Zero, m(B);
    0x62 NARROWS_D_W "narrows.d.w" R2, Zero, m(W);
    0x63 NARROW_D_B "narrow.d.b" R2, Zero, m(B);
    0x64 ZX_B_D "zx.b.d" R2, Zero, m(D);
    0x65 SX_B_D "sx.b.d" R2, Zero, m(D);
    0x66 ZX_W_D "zx.w.d" R2, Zero, m(D);
    0x67 SX_W_D "sx.w.d" R2, Zero, m(D);
    0x68 ZX_D_Q "zx.d.q" R2, Zero, m(Q);
    0x69 SX_D_Q "sx.d.q" R2, Zero, m(Q);
    0x6a POPCNT_B "popcnt.b" R2, Zero, m(B);
    0x6b POPCNT_Q "popcnt.q" R2, Zero, m(Q);
    0x70 HSUM_D "hsum.d" Sr, Zero, NONE;
    0x71 HSUM_Q "hsum.q" Sr, Zero, NONE;
    0x72 EXTRACT_Q "extract.q" Sri, Below(8), NONE;
    0x73 EXTRACT_D "extract.d" Sri, Below(16), NONE;
    0x74 INSERT_Q "insert.q" Ai, Below(8), NONE;
    0x75 MOVEMASK_B "movemask.b" Sr, Zero, NONE;
    0x78 CLMUL "clmul" R3i, Clmul, NONE;
    // mask helpers
    0xf6 MPOPCNT "mpopcnt" S0, Zero, R2;
    0xf7 MTZCNT "mtzcnt" S0, Zero, R2;
    0xf8 MLZCNT "mlzcnt" S0, Zero, R2;
    // data movement
    0x79 BLENDM "blendm" R3i, Widths(0xf), R2;
    0x7a BCAST_Q "bcast.q" R2i, Below(8), m(Q), Cu::PerOp;
    0x7b BCAST_D "bcast.d" R2i, Below(16), m(D), Cu::PerOp;
    0x7c BCAST_W "bcast.w" R2i, Below(32), m(W), Cu::PerOp;
    0x7d BCAST_B "bcast.b" R2i, Below(64), m(B), Cu::PerOp;
    0x7e PERMQI "permqi" R2i, Any, m(Q);
    0x7f EXTRACT256 "extract256" R2i, Below(2), m(Q);
    0x5f PERM2I "perm2i" A3i, Widths(0xf), m(Imm8);
    0x80 ALIGN_Q "align.q" R3i, Below(8), m(Q);
    0x81 ALIGN_D "align.d" R3i, Below(16), m(D);
    0x82 EXTRACT128 "extract128" R2i, Below(4), m(D);
    0x83 INSERT256 "insert256" R3i, Below(2), m(Q);
    0x84 INSERT128 "insert128" R3i, Below(4), m(D);
    0x85 PERM2T "perm2t" A3i, Widths(0b1101), m(Imm8);
    0x86 PERMB "permb" R3, Zero, m(B);
    0x87 PERMW "permw" R3, Zero, m(W);
    0x88 SHUFD "shufd" R2i, Any, m(D);
    0x89 SHUFHW "shufhw" R2i, Any, m(W);
    0x8a SHUFLW "shuflw" R2i, Any, m(W);
    0x8b SHUF128 "shuf128" R3i, Any, m(Q);
    0x8c SHUFPS "shufps" R3i, Any, m(D);
    0x8d ALIGNR "alignr" R3i, Below(32), m(B);
    0x8e BSHLI "bshli" R2i, Below(16), NONE;
    0x8f BSHRI "bshri" R2i, Below(16), NONE;
    0x90 UNPACKLO "unpacklo" R3i, Widths(0xf), m(Imm8);
    0x91 UNPACKHI "unpackhi" R3i, Widths(0xf), m(Imm8);
    0x92 COMPRESS "compress" R2i, Widths(0xf), mreq(Imm8), Cu::Fixed(2);
    0x93 EXPAND "expand" R2i, Widths(0xf), mreq(Imm8), Cu::Fixed(2);
    0x94 PACKUS_D "packus.d" R3, Zero, m(W);
    0x95 NARROW_D_W "narrow.d.w" R2, Zero, m(W);
    0x96 NARROWUS_D_W "narrowus.d.w" R2, Zero, m(W);
    0x97 NARROW_Q_B "narrow.q.b" R2, Zero, m(B);
    0x98 NARROWS_Q_B "narrows.q.b" R2, Zero, m(B);
    0x99 NARROWUS_Q_B "narrowus.q.b" R2, Zero, m(B);
    0x9a NARROW_Q_W "narrow.q.w" R2, Zero, m(W);
    0x9b NARROWS_Q_W "narrows.q.w" R2, Zero, m(W);
    0x9c NARROWUS_Q_W "narrowus.q.w" R2, Zero, m(W);
    0x9d NARROW_Q_D "narrow.q.d" R2, Zero, m(D);
    0x9e NARROWS_Q_D "narrows.q.d" R2, Zero, m(D);
    0x9f NARROWUS_Q_D "narrowus.q.d" R2, Zero, m(D);
    0xa0 NARROW_W_B "narrow.w.b" R2, Zero, m(B);
    0xa1 NARROWS_W_B "narrows.w.b" R2, Zero, m(B);
    0xa2 NARROWUS_W_B "narrowus.w.b" R2, Zero, m(B);
    0xa3 ZX_B_W "zx.b.w" R2, Zero, m(W);
    0xa4 SX_B_W "sx.b.w" R2, Zero, m(W);
    0xa5 ZX_B_Q "zx.b.q" R2, Zero, m(Q);
    0xa6 SX_B_Q "sx.b.q" R2, Zero, m(Q);
    0xa7 ZX_W_Q "zx.w.q" R2, Zero, m(Q);
    0xa8 SX_W_Q "sx.w.q" R2, Zero, m(Q);
    0xa9 EXTRACT_B "extract.b" Sri, Below(64), NONE, Cu::Fixed(2);
    0xaa EXTRACT_W "extract.w" Sri, Below(32), NONE, Cu::Fixed(2);
    0xab INSERT_B "insert.b" Ai, Below(64), NONE, Cu::Fixed(3);
    0xac INSERT_W "insert.w" Ai, Below(32), NONE, Cu::Fixed(3);
    0xad INSERT_D "insert.d" Ai, Below(16), NONE, Cu::Fixed(3);
    0xae MOVR_Q "movr.q" Ls, Zero, NONE;
    // arithmetic
    0xaf MUL52LO "mul52lo" R3, Zero, m(Q), Cu::Fixed(2);
    0xb0 MUL52HI "mul52hi" R3, Zero, m(Q), Cu::Fixed(2);
    0xb1 ABS_Q "abs.q" R2, Zero, m(Q);
    0xb2 MAX_Q "max.q" R3, Zero, m(Q);
    0xb3 MIN_Q "min.q" R3, Zero, m(Q);
    0xb4 ABS_B "abs.b" R2, Zero, m(B);
    0xb5 ABS_W "abs.w" R2, Zero, m(W);
    0xb6 ABS_D "abs.d" R2, Zero, m(D);
    0xb7 MAX_B "max.b" R3, Zero, m(B);
    0xb8 MIN_B "min.b" R3, Zero, m(B);
    0xb9 MAXU_B "maxu.b" R3, Zero, m(B);
    0xba MINU_B "minu.b" R3, Zero, m(B);
    0xbb ADDS_B "adds.b" R3, Zero, m(B);
    0xbc ADDS_W "adds.w" R3, Zero, m(W);
    0xbd ADDUS_B "addus.b" R3, Zero, m(B);
    0xbe ADDUS_W "addus.w" R3, Zero, m(W);
    0xbf SUBS_B "subs.b" R3, Zero, m(B);
    0xc0 SUBS_W "subs.w" R3, Zero, m(W);
    0xc1 SUBUS_B "subus.b" R3, Zero, m(B);
    0xc2 SUBUS_W "subus.w" R3, Zero, m(W);
    0xc3 AVG_B "avg.b" R3, Zero, m(B);
    0xc4 AVG_W "avg.w" R3, Zero, m(W);
    0xc5 SAD "sad" R3, Zero, NONE;
    0x1f MULHRS_W "mulhrs.w" R3, Zero, m(W);
    0x56 DPBUSDS "dpbusds" A3, Zero, m(D);
    0x57 DPWSSDS "dpwssds" A3, Zero, m(D);
    // bits, masks, compares
    0x46 CMP "cmp" S3i, Cmp, NONE, Cu::Fixed(2);
    0x47 TEST "test" S3i, Widths(0xf), NONE, Cu::Fixed(2);
    0x4e TESTN "testn" S3i, Widths(0xf), NONE, Cu::Fixed(2);
    0x76 MOVEMASK_W "movemask.w" Sr, Zero, NONE, Cu::Fixed(2);
    0x77 MOVEMASK_D "movemask.d" Sr, Zero, NONE, Cu::Fixed(2);
    0xc6 MOVEMASK_Q "movemask.q" Sr, Zero, NONE, Cu::Fixed(2);
    0xc7 MOVM_B "movm.b" R0, Zero, R2;
    0xc8 MOVM_W "movm.w" R0, Zero, R2;
    0xc9 MOVM_D "movm.d" R0, Zero, R2;
    0xca MOVM_Q "movm.q" R0, Zero, R2;
    0x2d TERNLOG_D "ternlog.d" A3i, Any, m(D);
    0x41 SHLV_W "shlv.w" R3, Zero, m(W);
    0x42 SHRV_W "shrv.w" R3, Zero, m(W);
    0x43 SARV_W "sarv.w" R3, Zero, m(W);
    0x44 SARV_D "sarv.d" R3, Zero, m(D);
    0x45 SARV_Q "sarv.q" R3, Zero, m(Q);
    0xce SHLDI_W "shldi.w" R3i, Below(16), m(W);
    0xcf SHLDI_D "shldi.d" R3i, Below(32), m(D);
    0xd0 SHLDI_Q "shldi.q" R3i, Below(64), m(Q);
    0xd1 SHRDI_W "shrdi.w" R3i, Below(16), m(W);
    0xd2 SHRDI_D "shrdi.d" R3i, Below(32), m(D);
    0xd3 SHRDI_Q "shrdi.q" R3i, Below(64), m(Q);
    0xd4 SHLDV_W "shldv.w" A3, Zero, m(W);
    0xd5 SHLDV_D "shldv.d" A3, Zero, m(D);
    0xd6 SHLDV_Q "shldv.q" A3, Zero, m(Q);
    0xd7 SHRDV_W "shrdv.w" A3, Zero, m(W);
    0xd8 SHRDV_D "shrdv.d" A3, Zero, m(D);
    0xd9 SHRDV_Q "shrdv.q" A3, Zero, m(Q);
    0xcb CONFLICT_D "conflict.d" R2, Zero, m(D), Cu::Fixed(4);
    0xcc CONFLICT_Q "conflict.q" R2, Zero, m(Q), Cu::Fixed(3);
    0x6e LZCNT_D "lzcnt.d" R2, Zero, m(D);
    0x6f LZCNT_Q "lzcnt.q" R2, Zero, m(Q);
    0x6c POPCNT_W "popcnt.w" R2, Zero, m(W);
    0x6d POPCNT_D "popcnt.d" R2, Zero, m(D);
    0x4f BITSHUF "bitshuf" S3, Zero, NONE, Cu::Fixed(2);
    0xcd MULTISHIFT "multishift" R3, Zero, m(B);
    // crypto
    0xda GF2P8MUL "gf2p8mul" R3, Zero, m(B);
    0xdb GF2P8AFFINE "gf2p8affine" R3i, Any, m(B);
    0xdc GF2P8AFFINEINV "gf2p8affineinv" R3i, Any, m(B);
    0xdd AESENC "aesenc" R3, Zero, NONE;
    0xde AESENCLAST "aesenclast" R3, Zero, NONE;
    0xdf AESDEC "aesdec" R3, Zero, NONE;
    0xe0 AESDECLAST "aesdeclast" R3, Zero, NONE;
    0xe1 SHA256RNDS2 "sha256rnds2" A3, Zero, NONE, Cu::Fixed(2);
    0xe2 SHA256MSG1 "sha256msg1" A2, Zero, NONE;
    0xe3 SHA256MSG2 "sha256msg2" A2, Zero, NONE;
    0xe4 SHA1RNDS4 "sha1rnds4" A2i, Below(4), NONE;
    0xe5 SHA1NEXTE "sha1nexte" A2, Zero, NONE;
    0xe6 SHA1MSG1 "sha1msg1" A2, Zero, NONE;
    0xe7 SHA1MSG2 "sha1msg2" A2, Zero, NONE;
    // memory
    0xe8 GATHER_QQ "gather.qq" Ga, Below(4), m(Q), Cu::Fixed(8);
    0xe9 GATHER_DD "gather.dd" Ga, Below(4), m(D), Cu::Fixed(16);
    0xea GATHER_DQ "gather.dq" Ga, Below(4), m(Q), Cu::Fixed(8);
    0xeb GATHER_QD "gather.qd" Ga, Below(4), m(D), Cu::Fixed(8);
    0xec SCATTER_QQ "scatter.qq" Sc, Below(4), m01(Q), Cu::Fixed(8);
    0xed SCATTER_DD "scatter.dd" Sc, Below(4), m01(D), Cu::Fixed(16);
    0xee SCATTER_DQ "scatter.dq" Sc, Below(4), m01(Q), Cu::Fixed(8);
    0xef SCATTER_QD "scatter.qd" Sc, Below(4), m01(D), Cu::Fixed(8);
    0x0c LDN "ldn" Lsi, Count, NONE, Cu::Imm8;
    0x0d STN "stn" Sti, Count, NONE, Cu::Imm8;
    0x0a LD_M "ld.m" Lsi, Widths(0xf), mreq(Imm8), Cu::Fixed(2);
    0x0b ST_M "st.m" Sti, Widths(0xf), m1(Imm8), Cu::Fixed(2);
    0x0e LD_Y "ld.y" Ls, Zero, NONE;
    0x0f ST_Y "st.y" St, Zero, NONE;
    0xf0 LD_X "ld.x" Ls, Zero, NONE;
    0xf1 ST_X "st.x" St, Zero, NONE;
    0xf2 LDBC_B "ldbc.b" Ls, Zero, m(B);
    0xf3 LDBC_W "ldbc.w" Ls, Zero, m(W);
    0xf4 LDBC_X "ldbc.x" Ls, Zero, m(D);
    0xf5 LDBC_Y "ldbc.y" Ls, Zero, m(Q);
}

/// Op code → index into [`OPS`] (`u8::MAX` = reserved)
const OP_INDEX: [u8; 256] = {
    let mut table = [u8::MAX; 256];
    let mut i = 0;
    while i < OPS.len() {
        table[OPS[i].op as usize] = i as u8;
        i += 1;
    }
    table
};

/// Looks up an op, `None` if it is reserved
pub fn op_info(op: u8) -> Option<&'static OpInfo> {
    OPS.get(OP_INDEX[op as usize] as usize)
}

/// Encodes a vector op as one little-endian instruction word
pub const fn encode(op: u8, imm8: u8, out: u8, in0: u8, in1: u8) -> u64 {
    encode_with_offset(op, imm8, out, in0, in1, 0)
}

/// Encodes a vector op with an `off` field (nonzero is only valid for memory accesses)
pub const fn encode_with_offset(op: u8, imm8: u8, out: u8, in0: u8, in1: u8, off: i16) -> u64 {
    encode_masked(op, imm8, out, in0, in1, off, 0)
}

/// Encodes a vector op with an `off` field and a mask mode
pub const fn encode_masked(op: u8, imm8: u8, out: u8, in0: u8, in1: u8, off: i16, mode: u8) -> u64 {
    let imm = (op as u32) << 24
        | (imm8 as u32) << 16
        | (out as u32 & 0x1f) << 10
        | (in0 as u32 & 0x1f) << 5
        | (in1 as u32 & 0x1f);
    ebpf::CALL_IMM as u64
        | (mode as u64 & 0xf) << 8
        | (VECTOR_OP_SRC as u64) << 12
        | (off as u16 as u64) << 16
        | (imm as u64) << 32
}

/// Whether `insn` is a vector op in the given setting
#[inline]
pub fn is_vector_op(insn: &ebpf::Insn, enable_vector_ops: bool, sbpf_version: SBPFVersion) -> bool {
    insn.opc == ebpf::CALL_IMM
        && insn.src == VECTOR_OP_SRC
        && enable_vector_ops
        && sbpf_version.static_syscalls()
}

/// A decoded and validated vector op
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorInsn {
    /// Static description of the op
    pub info: &'static OpInfo,
    /// imm\[23:16\]
    pub imm8: u8,
    /// Output (or accumulator) register
    pub out: u8,
    /// First input register
    pub in0: u8,
    /// Second input register
    pub in1: u8,
    /// Displacement added to `r1` by memory accesses, 0 for all other ops
    pub off: i16,
    /// Mask mode (the `dst` field): 0 = unmasked, 1 = merge, 2 = zero
    pub mode: u8,
}

impl PartialEq for OpInfo {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op
    }
}
impl Eq for OpInfo {}

impl VectorInsn {
    /// Decodes and validates a vector op (`insn` must be a `CALL_IMM` with `src = 2`)
    pub fn decode(insn: &ebpf::Insn) -> Result<Self, &'static str> {
        let imm = insn.imm as u32;
        if imm & (1 << 15) != 0 {
            return Err("imm bit 15 must be 0");
        }
        let info = op_info((imm >> 24) as u8).ok_or("unknown op")?;
        if !info.implemented {
            return Err("unimplemented op");
        }
        if insn.off != 0 && !info.accesses_memory() {
            return Err("off must be 0");
        }
        let imm8 = (imm >> 16) as u8;
        let out = ((imm >> 10) & 0x1f) as u8;
        let in0 = ((imm >> 5) & 0x1f) as u8;
        let in1 = (imm & 0x1f) as u8;
        for (used, reg) in [
            (info.kind.uses_out(), out),
            (info.kind.uses_in0(), in0),
            (info.kind.uses_in1(), in1),
        ] {
            if !used && reg != 0 {
                return Err("unused register field must be 0");
            }
            if reg as usize >= VECTOR_REGISTERS {
                return Err("register field out of range");
            }
        }
        if !info.imm8_is_valid(imm8) {
            return Err("imm8 out of range");
        }
        if info.imm8_is_count() {
            let base = if info.kind.uses_out() { out } else { in0 };
            if base as usize + imm8 as usize > VECTOR_REGISTERS {
                return Err("register range out of range");
            }
        }
        let mode = insn.dst;
        if mode >= 3 {
            return Err("mask mode out of range");
        }
        if info.modes() & (1 << mode) == 0 {
            return Err("mask mode not allowed");
        }
        Ok(Self {
            info,
            imm8,
            out,
            in0,
            in1,
            off: insn.off,
            mode,
        })
    }

    /// Number of consecutive registers named by `out` or `in0` (`ldn`, `stn`: imm8)
    pub fn register_count(&self) -> u8 {
        if self.info.imm8_is_count() {
            self.imm8
        } else {
            1
        }
    }

    /// Whether `r2` is read
    pub fn reads_r2(&self) -> bool {
        self.mode != 0 || self.info.mask == Mask::Data
    }

    /// Whether the old value of `out` is read
    pub fn reads_out(&self) -> bool {
        self.info.kind.reads_out() || (self.mode == 1 && self.info.kind.uses_out())
    }

    /// Bit set of the vector registers this op reads or writes
    pub fn registers_used(&self) -> u32 {
        let kind = self.info.kind;
        let span = |reg: u8| (((1u64 << self.register_count()) - 1) as u32) << reg;
        (if kind.uses_out() { span(self.out) } else { 0 })
            | (if kind.uses_in0() {
                if kind.uses_out() {
                    1 << self.in0
                } else {
                    span(self.in0)
                }
            } else {
                0
            })
            | (kind.uses_in1() as u32) << self.in1
    }
}

impl std::fmt::Display for VectorInsn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let kind = self.info.kind;
        write!(f, "vec.{}", self.info.name)?;
        let mut separator = " ";
        let mut operand = |f: &mut std::fmt::Formatter, text: String| {
            let result = write!(f, "{separator}{text}");
            separator = ", ";
            result
        };
        let registers = |reg: u8| {
            if self.register_count() > 1 {
                format!("v{}..v{}", reg, reg + self.register_count() - 1)
            } else {
                format!("v{reg}")
            }
        };
        let mask = ["", "{r2}", "{r2}{z}"][self.mode as usize % 3];
        if kind.uses_out() {
            operand(f, format!("{}{mask}", registers(self.out)))?;
        }
        if kind.writes_r0() {
            operand(f, "r0".to_string())?;
        }
        if matches!(kind, Kind::Ga | Kind::Sc) {
            operand(
                f,
                format!(
                    "[r1{:+} + v{} << {}]{}",
                    self.off, self.in0, self.imm8, mask
                ),
            )?;
            if kind == Kind::Sc {
                operand(f, format!("v{}", self.in1))?;
            }
            return Ok(());
        }
        if kind.uses_in0() {
            operand(f, registers(self.in0))?;
        }
        if kind.uses_in1() {
            operand(f, format!("v{}", self.in1))?;
        }
        if self.info.accesses_memory() {
            let mask = if kind.uses_out() { "" } else { mask };
            operand(f, format!("[r1{:+}]{mask}", self.off))?;
        } else if matches!(kind, Kind::Ls | Kind::Ai) {
            operand(f, "r1".to_string())?;
        }
        if self.info.mask == Mask::Data {
            operand(f, "r2".to_string())?;
        }
        if kind.uses_imm8() && !self.info.imm8_is_count() {
            operand(f, format!("{:#x}", self.imm8))?;
        }
        Ok(())
    }
}

/// Verifier checks of all vector ops in `prog`, used if vector ops are enabled
pub(crate) fn verify(prog: &[u8], sbpf_version: SBPFVersion) -> Result<(), VerifierError> {
    let mut insn_ptr: usize = 0;
    while (insn_ptr + 1) * ebpf::INSN_SIZE <= prog.len() {
        let insn = ebpf::get_insn(prog, insn_ptr);
        if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
            insn_ptr += 1;
        } else if is_vector_op(&insn, true, sbpf_version) {
            VectorInsn::decode(&insn)
                .map_err(|reason| VerifierError::InvalidVectorOp(insn_ptr, reason))?;
        }
        insn_ptr += 1;
    }
    Ok(())
}

/// Set in [registers_used_by_program] if there is a vector op (some use no vector register)
pub(crate) const PROGRAM_USES_VECTOR_OPS: u32 = 1 << 31;

/// Bit set of the vector registers used by any vector op in `prog` plus
/// [PROGRAM_USES_VECTOR_OPS], 0 if there are none
pub(crate) fn registers_used_by_program(prog: &[u8], sbpf_version: SBPFVersion) -> u32 {
    let mut mask = 0;
    let mut insn_ptr: usize = 0;
    while (insn_ptr + 1) * ebpf::INSN_SIZE <= prog.len() {
        let insn = ebpf::get_insn_unchecked(prog, insn_ptr);
        if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
            insn_ptr += 1;
        } else if is_vector_op(&insn, true, sbpf_version) {
            if let Ok(vector_insn) = VectorInsn::decode(&insn) {
                mask |= vector_insn.registers_used() | PROGRAM_USES_VECTOR_OPS;
            }
        }
        insn_ptr += 1;
    }
    mask
}

// Portable reference semantics

type V = VectorRegister;

fn map1<const N: usize>(a: &V, f: impl Fn(u64) -> u64) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / N {
        r.set_lane::<N>(i, f(a.lane::<N>(i)));
    }
    r
}

fn map2<const N: usize>(a: &V, b: &V, f: impl Fn(u64, u64) -> u64) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / N {
        r.set_lane::<N>(i, f(a.lane::<N>(i), b.lane::<N>(i)));
    }
    r
}

fn map3<const N: usize>(c: &V, a: &V, b: &V, f: impl Fn(u64, u64, u64) -> u64) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / N {
        r.set_lane::<N>(i, f(c.lane::<N>(i), a.lane::<N>(i), b.lane::<N>(i)));
    }
    r
}

/// Sign extends the low `bits` of `x`
fn sx(x: u64, bits: u32) -> i64 {
    ((x << (64 - bits)) as i64) >> (64 - bits)
}

fn mask_of(condition: bool) -> u64 {
    if condition {
        u64::MAX
    } else {
        0
    }
}

fn sat(x: i64, min: i64, max: i64) -> u64 {
    x.clamp(min, max) as u64
}

/// Sum of `n` products of `bits` wide sub-lanes of `x` and `y` with the given signedness
fn dot(x: u64, y: u64, bits: u32, n: u32, x_signed: bool, y_signed: bool) -> i64 {
    let mask = (1u64 << bits) - 1;
    (0..n)
        .map(|j| {
            let (xs, ys) = ((x >> (j * bits)) & mask, (y >> (j * bits)) & mask);
            let xv = if x_signed { sx(xs, bits) } else { xs as i64 };
            let yv = if y_signed { sx(ys, bits) } else { ys as i64 };
            xv * yv
        })
        .sum()
}

fn clmul64(x: u64, y: u64) -> u128 {
    (0..64)
        .filter(|i| (y >> i) & 1 == 1)
        .fold(0u128, |acc, i| acc ^ ((x as u128) << i))
}

/// High `bits` of (`hi`:`lo`) << `n`, the lanes of `vpshld*`
fn shld(hi: u64, lo: u64, n: u64, bits: u32) -> u64 {
    ((((hi as u128) << bits | lo as u128) << n) >> bits) as u64
}

/// Low `bits` of (`hi`:`lo`) >> `n`, the lanes of `vpshrd*`
fn shrd(hi: u64, lo: u64, n: u64, bits: u32) -> u64 {
    (((hi as u128) << bits | lo as u128) >> n) as u64
}

/// Bit j of lane i is set if lane j < i equals lane i
fn conflict<const N: usize>(a: &V) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / N {
        let x = a.lane::<N>(i);
        r.set_lane::<N>(
            i,
            (0..i).fold(0, |bits, j| bits | ((a.lane::<N>(j) == x) as u64) << j),
        );
    }
    r
}

/// Reads lane `i` of `bytes` (1, 2, 4 or 8) bytes, zero extended
fn lane(v: &V, bytes: usize, i: usize) -> u64 {
    let mut lane = [0u8; 8];
    lane[..bytes].copy_from_slice(&v.0[i * bytes..(i + 1) * bytes]);
    u64::from_le_bytes(lane)
}

/// Writes the low `bytes` bytes of `value` to lane `i`
fn set_lane(v: &mut V, bytes: usize, i: usize, value: u64) {
    v.0[i * bytes..(i + 1) * bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
}

/// The dwords d0..d3 of the low 128 bits
fn low_dwords(v: &V) -> [u32; 4] {
    std::array::from_fn(|k| v.lane::<4>(k) as u32)
}

/// `c` with its low 128 bits replaced by the dwords d0..d3 (SHA ops leave the rest unchanged)
fn with_low_dwords(c: &V, d: [u32; 4]) -> V {
    let mut r = *c;
    for (k, x) in d.iter().enumerate() {
        r.set_lane::<4>(k, *x as u64);
    }
    r
}

/// Lane i (of `bytes` bytes) of the result is lane `f(i).1` of `f(i).0` for i < `lanes`, the rest
/// is 0
fn pick<'a>(bytes: usize, lanes: usize, f: impl Fn(usize) -> (&'a V, usize)) -> V {
    let mut r = V::ZERO;
    for i in 0..lanes {
        let (v, k) = f(i);
        set_lane(&mut r, bytes, i, lane(v, bytes, k));
    }
    r
}

/// Lane i of `x` (of `from` bytes) to lane i of `to` bytes: truncated (`saturate` 0), signed
/// saturated (1) or unsigned saturated (2). The rest of the result is 0.
fn narrow(x: &V, from: usize, to: usize, saturate: u8) -> V {
    let bits = 8 * to as u32;
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / from {
        let value = lane(x, from, i);
        let value = match saturate {
            0 => value,
            1 => sat(
                sx(value, 8 * from as u32),
                -(1i64 << (bits - 1)),
                (1i64 << (bits - 1)) - 1,
            ),
            _ => value.min(u64::MAX >> (64 - bits)),
        };
        set_lane(&mut r, to, i, value);
    }
    r
}

/// The low lanes of `x` (of `from` bytes) zero or sign extended to lanes of `to` bytes
fn extend(x: &V, from: usize, to: usize, signed: bool) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / to {
        let value = lane(x, from, i);
        let value = if signed {
            sx(value, 8 * from as u32) as u64
        } else {
            value
        };
        set_lane(&mut r, to, i, value);
    }
    r
}

fn sha256msg1(src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    let sigma0 = |w: u32| w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3);
    let w = [src1[0], src1[1], src1[2], src1[3], src2[0]];
    std::array::from_fn(|k| w[k].wrapping_add(sigma0(w[k + 1])))
}

fn sha256rnds2(src1: [u32; 4], src2: [u32; 4], wk: [u32; 2]) -> [u32; 4] {
    let [f, e, b, a] = src2;
    let [h, g, d, c] = src1;
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (a, b, c, d, e, f, g, h);
    for wk in wk {
        let ch = (e & f) ^ (!e & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let t1 = ch.wrapping_add(sigma1).wrapping_add(wk).wrapping_add(h);
        (h, g, f, e) = (g, f, e, t1.wrapping_add(d));
        (d, c, b, a) = (c, b, a, t1.wrapping_add(maj).wrapping_add(sigma0));
    }
    [f, e, b, a]
}

fn sha1rnds4(src1: [u32; 4], src2: [u32; 4], imm8: u8) -> [u32; 4] {
    let (f, k): (fn(u32, u32, u32) -> u32, u32) = match imm8 & 3 {
        0 => (|b, c, d| (b & c) ^ (!b & d), 0x5a82_7999),
        1 => (|b, c, d| b ^ c ^ d, 0x6ed9_eba1),
        2 => (|b, c, d| (b & c) ^ (b & d) ^ (c & d), 0x8f1b_bcdc),
        _ => (|b, c, d| b ^ c ^ d, 0xca62_c1d6),
    };
    let [mut d, mut c, mut b, mut a] = src1;
    let w = [src2[3], src2[2], src2[1], src2[0]];
    let mut e = 0;
    for (i, w) in w.iter().enumerate() {
        let t = f(b, c, d)
            .wrapping_add(a.rotate_left(5))
            .wrapping_add(*w)
            .wrapping_add(if i == 0 { 0 } else { e })
            .wrapping_add(k);
        (e, d, c, b, a) = (d, c, b.rotate_left(30), a, t);
    }
    [d, c, b, a]
}

fn sha256msg2(src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    let sigma1 = |w: u32| w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10);
    let w16 = src1[0].wrapping_add(sigma1(src2[2]));
    let w17 = src1[1].wrapping_add(sigma1(src2[3]));
    let w18 = src1[2].wrapping_add(sigma1(w16));
    let w19 = src1[3].wrapping_add(sigma1(w17));
    [w16, w17, w18, w19]
}

fn sha1nexte(src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    let [w0, w1, w2, w3] = src2;
    [w0, w1, w2, w3.wrapping_add(src1[3].rotate_left(30))]
}

fn sha1msg1(src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    let [w3, w2, w1, w0] = src1;
    let [_, _, w5, w4] = src2;
    [w5 ^ w3, w4 ^ w2, w3 ^ w1, w2 ^ w0]
}

fn sha1msg2(src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    let [w15, w14, w13, _] = src2;
    let w16 = (src1[3] ^ w13).rotate_left(1);
    let w17 = (src1[2] ^ w14).rotate_left(1);
    let w18 = (src1[1] ^ w15).rotate_left(1);
    let w19 = (src1[0] ^ w16).rotate_left(1);
    [w19, w18, w17, w16]
}

/// Product in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1
const fn gf_mul(mut x: u8, mut y: u8) -> u8 {
    let mut product = 0;
    while y != 0 {
        if y & 1 == 1 {
            product ^= x;
        }
        x = (x << 1) ^ if x & 0x80 != 0 { 0x1b } else { 0 };
        y >>= 1;
    }
    product
}

/// Inverses in GF(2^8) (x^254), 0 for 0
const GF_INVERSE: [u8; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let (mut power, mut square, mut e) = (1, i as u8, 254);
        while e != 0 {
            if e & 1 == 1 {
                power = gf_mul(power, square);
            }
            square = gf_mul(square, square);
            e >>= 1;
        }
        table[i] = power;
        i += 1;
    }
    table
};

/// The AES S-box: the affine map of FIPS-197 applied to the inverse
const AES_SBOX: [u8; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let s = GF_INVERSE[i];
        table[i] =
            s ^ s.rotate_left(1) ^ s.rotate_left(2) ^ s.rotate_left(3) ^ s.rotate_left(4) ^ 0x63;
        i += 1;
    }
    table
};

const AES_INVERSE_SBOX: [u8; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[AES_SBOX[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// Bit t of the result is the parity of `x & a.b[7 - t]`, XOR bit t of `c`
fn gf_affine(a: u64, x: u8, c: u8) -> u8 {
    (0..8).fold(c, |r, t| {
        r ^ (((a >> (8 * (7 - t))) as u8 & x).count_ones() as u8 & 1) << t
    })
}

/// One AES round on each 128-bit lane, `inverse` for the equivalent inverse cipher and `last`
/// without (Inv)MixColumns. Byte k of a lane is row k % 4 of column k / 4 of the state.
fn aes_round(state: &V, key: &V, inverse: bool, last: bool) -> V {
    let mut r = V::ZERO;
    for l in 0..4 {
        let s = &state.0[16 * l..16 * (l + 1)];
        // (Inv)ShiftRows rotates row i by i columns, (Inv)SubBytes commutes with it
        let mut t: [u8; 16] = std::array::from_fn(|k| {
            let (row, column) = (k % 4, k / 4);
            let from = if inverse {
                column + 4 - row
            } else {
                column + row
            } % 4;
            let x = s[row + 4 * from] as usize;
            if inverse {
                AES_INVERSE_SBOX[x]
            } else {
                AES_SBOX[x]
            }
        });
        if !last {
            let m = if inverse {
                [14, 11, 13, 9]
            } else {
                [2, 3, 1, 1]
            };
            for column in t.chunks_mut(4) {
                let x = [column[0], column[1], column[2], column[3]];
                for (row, y) in column.iter_mut().enumerate() {
                    *y = (0..4).fold(0, |sum, j| sum ^ gf_mul(x[j], m[(j + 4 - row) % 4]));
                }
            }
        }
        for (k, x) in t.iter().enumerate() {
            r.0[16 * l + k] = x ^ key.0[16 * l + k];
        }
    }
    r
}

/// Applies mask mode `mode` with the mask `r2` to `result`, the unmasked result of the op
///
/// Lane i of the result keeps its value if bit i of `r2` is set, otherwise it is the lane of
/// `old` (mode 1) or 0 (mode 2). Bytes above a narrower result stay 0.
pub fn apply_mask(info: &OpInfo, imm8: u8, mode: u8, old: &V, result: V, r2: u64) -> V {
    if mode == 0 {
        return result;
    }
    let bytes = info.mask_lane_bytes(imm8);
    let mut r = result;
    for i in 0..result_bytes(info.op, imm8) / bytes {
        if (r2 >> i) & 1 == 0 {
            let value = if mode == 1 { lane(old, bytes, i) } else { 0 };
            set_lane(&mut r, bytes, i, value);
        }
    }
    r
}

/// Reference semantics of all ops which only read and write vector registers
///
/// `c` is the old value of `out` (read by the accumulating kinds), `a` is `in0` and `b` is `in1`.
/// Returns `None` for ops which access memory or scalar registers.
pub fn execute_register_op(op: u8, imm8: u8, c: &V, a: &V, b: &V) -> Option<V> {
    const M52: u64 = (1 << 52) - 1;
    let i = imm8 as u32;
    Some(match op {
        op::ZERO => V::ZERO,
        op::MOV => *a,
        op::ADD_B => map2::<1>(a, b, u64::wrapping_add),
        op::ADD_W => map2::<2>(a, b, u64::wrapping_add),
        op::ADD_D => map2::<4>(a, b, u64::wrapping_add),
        op::ADD_Q => map2::<8>(a, b, u64::wrapping_add),
        op::SUB_B => map2::<1>(a, b, u64::wrapping_sub),
        op::SUB_W => map2::<2>(a, b, u64::wrapping_sub),
        op::SUB_D => map2::<4>(a, b, u64::wrapping_sub),
        op::SUB_Q => map2::<8>(a, b, u64::wrapping_sub),
        op::MULLO_W => map2::<2>(a, b, u64::wrapping_mul),
        op::MULLO_D => map2::<4>(a, b, u64::wrapping_mul),
        op::MULLO_Q => map2::<8>(a, b, u64::wrapping_mul),
        op::MULWU => map2::<8>(a, b, |x, y| (x as u32 as u64) * (y as u32 as u64)),
        op::MULWS => map2::<8>(a, b, |x, y| ((x as i32 as i64) * (y as i32 as i64)) as u64),
        op::MULHI_W => map2::<2>(a, b, |x, y| ((sx(x, 16) * sx(y, 16)) >> 16) as u64),
        op::MULHIU_W => map2::<2>(a, b, |x, y| (x * y) >> 16),
        op::MAX_D => map2::<4>(a, b, |x, y| sx(x, 32).max(sx(y, 32)) as u64),
        op::MIN_D => map2::<4>(a, b, |x, y| sx(x, 32).min(sx(y, 32)) as u64),
        op::MAXU_D => map2::<4>(a, b, u64::max),
        op::MINU_D => map2::<4>(a, b, u64::min),
        op::MAX_W => map2::<2>(a, b, |x, y| sx(x, 16).max(sx(y, 16)) as u64),
        op::MIN_W => map2::<2>(a, b, |x, y| sx(x, 16).min(sx(y, 16)) as u64),
        op::MAXU_Q => map2::<8>(a, b, u64::max),
        op::MINU_Q => map2::<8>(a, b, u64::min),
        op::MAXU_W => map2::<2>(a, b, u64::max),
        op::MINU_W => map2::<2>(a, b, u64::min),
        op::AND => map2::<8>(a, b, |x, y| x & y),
        op::OR => map2::<8>(a, b, |x, y| x | y),
        op::XOR => map2::<8>(a, b, |x, y| x ^ y),
        op::ANDN => map2::<8>(a, b, |x, y| !x & y),
        op::TERNLOG => map3::<8>(c, a, b, |z, x, y| {
            (0..8)
                .filter(|k| (imm8 >> k) & 1 == 1)
                .map(|k| {
                    let pick = |v: u64, bit: u32| if (k >> bit) & 1 == 1 { v } else { !v };
                    pick(z, 2) & pick(x, 1) & pick(y, 0)
                })
                .fold(0, |r, term| r | term)
        }),
        op::SHLI_W => map1::<2>(a, |x| x << i),
        op::SHLI_D => map1::<4>(a, |x| x << i),
        op::SHLI_Q => map1::<8>(a, |x| x << i),
        op::SHRI_W => map1::<2>(a, |x| x >> i),
        op::SHRI_D => map1::<4>(a, |x| x >> i),
        op::SHRI_Q => map1::<8>(a, |x| x >> i),
        op::SARI_W => map1::<2>(a, |x| (sx(x, 16) >> i) as u64),
        op::SARI_D => map1::<4>(a, |x| (sx(x, 32) >> i) as u64),
        op::SARI_Q => map1::<8>(a, |x| (sx(x, 64) >> i) as u64),
        op::ROLI_D => map1::<4>(a, |x| (x as u32).rotate_left(i) as u64),
        op::ROLI_Q => map1::<8>(a, |x| x.rotate_left(i)),
        op::SHLV_D => map2::<4>(a, b, |x, y| if y < 32 { x << y } else { 0 }),
        op::SHLV_Q => map2::<8>(a, b, |x, y| if y < 64 { x << y } else { 0 }),
        op::SHRV_D => map2::<4>(a, b, |x, y| if y < 32 { x >> y } else { 0 }),
        op::SHRV_Q => map2::<8>(a, b, |x, y| if y < 64 { x >> y } else { 0 }),
        op::ROLV_D => map2::<4>(a, b, |x, y| (x as u32).rotate_left((y % 32) as u32) as u64),
        op::ROLV_Q => map2::<8>(a, b, |x, y| x.rotate_left((y % 64) as u32)),
        op::CMPEQ_B => map2::<1>(a, b, |x, y| mask_of(x == y)),
        op::CMPEQ_D => map2::<4>(a, b, |x, y| mask_of(x == y)),
        op::CMPEQ_Q => map2::<8>(a, b, |x, y| mask_of(x == y)),
        op::CMPGT_D => map2::<4>(a, b, |x, y| mask_of(sx(x, 32) > sx(y, 32))),
        op::CMPGT_Q => map2::<8>(a, b, |x, y| mask_of(sx(x, 64) > sx(y, 64))),
        op::CMPGTU_Q => map2::<8>(a, b, |x, y| mask_of(x > y)),
        op::DPBUSD => map3::<4>(c, a, b, |z, x, y| {
            z.wrapping_add(dot(x, y, 8, 4, false, true) as u64)
        }),
        op::DPWSSD => map3::<4>(c, a, b, |z, x, y| {
            z.wrapping_add(dot(x, y, 16, 2, true, true) as u64)
        }),
        op::MADDUBS => map2::<2>(a, b, |x, y| {
            sat(
                dot(x, y, 8, 2, false, true),
                i16::MIN as i64,
                i16::MAX as i64,
            )
        }),
        op::MADDWD => map2::<4>(a, b, |x, y| dot(x, y, 16, 2, true, true) as u64),
        op::MADD52LO => map3::<8>(c, a, b, |z, x, y| {
            z.wrapping_add(((x & M52) as u128 * (y & M52) as u128) as u64 & M52)
        }),
        op::MADD52HI => map3::<8>(c, a, b, |z, x, y| {
            z.wrapping_add((((x & M52) as u128 * (y & M52) as u128) >> 52) as u64)
        }),
        op::PACKS_D | op::PACKS_W | op::PACKUS_W => {
            let mut r = V::ZERO;
            for l in 0..4 {
                for (half, src) in [a, b].iter().enumerate() {
                    if op == op::PACKS_D {
                        for k in 0..4 {
                            let x = sx(src.lane::<4>(4 * l + k), 32);
                            r.set_lane::<2>(
                                8 * l + 4 * half + k,
                                sat(x, i16::MIN as i64, i16::MAX as i64),
                            );
                        }
                    } else {
                        for k in 0..8 {
                            let x = sx(src.lane::<2>(8 * l + k), 16);
                            let (min, max) = if op == op::PACKS_W {
                                (i8::MIN as i64, i8::MAX as i64)
                            } else {
                                (0, u8::MAX as i64)
                            };
                            r.set_lane::<1>(16 * l + 8 * half + k, sat(x, min, max));
                        }
                    }
                }
            }
            r
        }
        op::PERMD => {
            let mut r = V::ZERO;
            for k in 0..16 {
                r.set_lane::<4>(k, b.lane::<4>((a.lane::<4>(k) & 15) as usize));
            }
            r
        }
        op::PERMQ => {
            let mut r = V::ZERO;
            for k in 0..8 {
                r.set_lane::<8>(k, b.lane::<8>((a.lane::<8>(k) & 7) as usize));
            }
            r
        }
        op::SHUFB => {
            let mut r = V::ZERO;
            for k in 0..64 {
                let control = b.0[k];
                r.0[k] = if control & 0x80 != 0 {
                    0
                } else {
                    a.0[(k & !15) + (control & 15) as usize]
                };
            }
            r
        }
        op::PERM2_W => {
            let mut r = V::ZERO;
            for i in 0..32 {
                let k = (a.lane::<2>(i) & 63) as usize;
                r.set_lane::<2>(
                    i,
                    if k < 32 {
                        c.lane::<2>(k)
                    } else {
                        b.lane::<2>(k - 32)
                    },
                );
            }
            r
        }
        op::NARROWUS_D_B | op::NARROWS_D_B | op::NARROWS_D_W | op::NARROW_D_B => {
            let mut r = V::ZERO;
            for k in 0..16 {
                let x = a.lane::<4>(k);
                match op {
                    op::NARROWUS_D_B => r.set_lane::<1>(k, x.min(u8::MAX as u64)),
                    op::NARROWS_D_B => {
                        r.set_lane::<1>(k, sat(sx(x, 32), i8::MIN as i64, i8::MAX as i64))
                    }
                    op::NARROWS_D_W => {
                        r.set_lane::<2>(k, sat(sx(x, 32), i16::MIN as i64, i16::MAX as i64))
                    }
                    _ => r.set_lane::<1>(k, x),
                }
            }
            r
        }
        op::ZX_B_D | op::SX_B_D | op::ZX_W_D | op::SX_W_D | op::ZX_D_Q | op::SX_D_Q => {
            let mut r = V::ZERO;
            match op {
                op::ZX_B_D => (0..16).for_each(|k| r.set_lane::<4>(k, a.lane::<1>(k))),
                op::SX_B_D => {
                    (0..16).for_each(|k| r.set_lane::<4>(k, sx(a.lane::<1>(k), 8) as u64))
                }
                op::ZX_W_D => (0..16).for_each(|k| r.set_lane::<4>(k, a.lane::<2>(k))),
                op::SX_W_D => {
                    (0..16).for_each(|k| r.set_lane::<4>(k, sx(a.lane::<2>(k), 16) as u64))
                }
                op::ZX_D_Q => (0..8).for_each(|k| r.set_lane::<8>(k, a.lane::<4>(k))),
                _ => (0..8).for_each(|k| r.set_lane::<8>(k, sx(a.lane::<4>(k), 32) as u64)),
            }
            r
        }
        op::POPCNT_B => map1::<1>(a, |x| x.count_ones() as u64),
        op::POPCNT_Q => map1::<8>(a, |x| x.count_ones() as u64),
        op::CLMUL => {
            let mut r = V::ZERO;
            for l in 0..4 {
                let x = a.lane::<8>(2 * l + (imm8 & 1) as usize);
                let y = b.lane::<8>(2 * l + ((imm8 >> 4) & 1) as usize);
                let product = clmul64(x, y);
                r.set_lane::<8>(2 * l, product as u64);
                r.set_lane::<8>(2 * l + 1, (product >> 64) as u64);
            }
            r
        }
        // data movement
        op::BCAST_Q => pick(8, 8, |_| (a, imm8 as usize)),
        op::BCAST_D => pick(4, 16, |_| (a, imm8 as usize)),
        op::BCAST_W => pick(2, 32, |_| (a, imm8 as usize)),
        op::BCAST_B => pick(1, 64, |_| (a, imm8 as usize)),
        op::PERMQI => pick(8, 8, |i| {
            (a, (i & !3) + (imm8 as usize >> (2 * (i & 3)) & 3))
        }),
        op::EXTRACT256 => pick(8, 4, |i| (a, 4 * imm8 as usize + i)),
        op::PERM2I | op::PERM2T => {
            let bytes = 1 << imm8;
            let lanes = VECTOR_SIZE / bytes;
            // perm2i: the indices are in out, perm2t: out is the first table
            let (index, table) = if op == op::PERM2I { (c, a) } else { (a, c) };
            pick(bytes, lanes, |i| {
                let k = lane(index, bytes, i) as usize & (2 * lanes - 1);
                if k < lanes {
                    (table, k)
                } else {
                    (b, k - lanes)
                }
            })
        }
        op::ALIGN_Q => pick(8, 8, |i| {
            let j = i + imm8 as usize;
            if j < 8 {
                (b, j)
            } else {
                (a, j - 8)
            }
        }),
        op::ALIGN_D => pick(4, 16, |i| {
            let j = i + imm8 as usize;
            if j < 16 {
                (b, j)
            } else {
                (a, j - 16)
            }
        }),
        op::EXTRACT128 => pick(4, 4, |i| (a, 4 * imm8 as usize + i)),
        op::INSERT256 => pick(8, 8, |i| {
            if i / 4 == imm8 as usize {
                (b, i % 4)
            } else {
                (a, i)
            }
        }),
        op::INSERT128 => pick(4, 16, |i| {
            if i / 4 == imm8 as usize {
                (b, i % 4)
            } else {
                (a, i)
            }
        }),
        op::PERMB => pick(1, 64, |i| (b, a.0[i] as usize & 63)),
        op::PERMW => pick(2, 32, |i| (b, a.lane::<2>(i) as usize & 31)),
        op::SHUFD => pick(4, 16, |i| {
            (a, (i & !3) + (imm8 as usize >> (2 * (i & 3)) & 3))
        }),
        op::SHUFHW => pick(2, 32, |i| {
            let j = i & 7;
            if j < 4 {
                (a, i)
            } else {
                (a, (i & !7) + 4 + (imm8 as usize >> (2 * (j - 4)) & 3))
            }
        }),
        op::SHUFLW => pick(2, 32, |i| {
            let j = i & 7;
            if j < 4 {
                (a, (i & !7) + (imm8 as usize >> (2 * j) & 3))
            } else {
                (a, i)
            }
        }),
        op::SHUF128 => pick(8, 8, |i| {
            let l = i / 2;
            let source = if l < 2 { a } else { b };
            (source, 2 * (imm8 as usize >> (2 * l) & 3) + (i & 1))
        }),
        op::SHUFPS => pick(4, 16, |i| {
            let j = i & 3;
            let source = if j < 2 { a } else { b };
            (source, (i & !3) + (imm8 as usize >> (2 * j) & 3))
        }),
        op::ALIGNR => pick(1, 64, |i| {
            let j = (i & 15) + imm8 as usize;
            if j < 16 {
                (b, (i & !15) + j)
            } else if j < 32 {
                (a, (i & !15) + j - 16)
            } else {
                (&V::ZERO, 0)
            }
        }),
        op::BSHLI => pick(1, 64, |i| {
            if i & 15 >= imm8 as usize {
                (a, i - imm8 as usize)
            } else {
                (&V::ZERO, 0)
            }
        }),
        op::BSHRI => pick(1, 64, |i| {
            if (i & 15) + (imm8 as usize) < 16 {
                (a, i + imm8 as usize)
            } else {
                (&V::ZERO, 0)
            }
        }),
        op::UNPACKLO | op::UNPACKHI => {
            let bytes = 1 << imm8;
            let per_128 = 16 / bytes;
            let high = if op == op::UNPACKHI { per_128 / 2 } else { 0 };
            pick(bytes, VECTOR_SIZE / bytes, |i| {
                let (l, j) = (i / per_128, i % per_128);
                let source = if j % 2 == 0 { a } else { b };
                (source, l * per_128 + high + j / 2)
            })
        }
        op::PACKUS_D => {
            let mut r = V::ZERO;
            for l in 0..4 {
                for (half, src) in [a, b].iter().enumerate() {
                    for k in 0..4 {
                        let x = sx(src.lane::<4>(4 * l + k), 32);
                        r.set_lane::<2>(8 * l + 4 * half + k, sat(x, 0, u16::MAX as i64));
                    }
                }
            }
            r
        }
        op::NARROW_D_W => narrow(a, 4, 2, 0),
        op::NARROWUS_D_W => narrow(a, 4, 2, 2),
        op::NARROW_Q_B => narrow(a, 8, 1, 0),
        op::NARROWS_Q_B => narrow(a, 8, 1, 1),
        op::NARROWUS_Q_B => narrow(a, 8, 1, 2),
        op::NARROW_Q_W => narrow(a, 8, 2, 0),
        op::NARROWS_Q_W => narrow(a, 8, 2, 1),
        op::NARROWUS_Q_W => narrow(a, 8, 2, 2),
        op::NARROW_Q_D => narrow(a, 8, 4, 0),
        op::NARROWS_Q_D => narrow(a, 8, 4, 1),
        op::NARROWUS_Q_D => narrow(a, 8, 4, 2),
        op::NARROW_W_B => narrow(a, 2, 1, 0),
        op::NARROWS_W_B => narrow(a, 2, 1, 1),
        op::NARROWUS_W_B => narrow(a, 2, 1, 2),
        op::ZX_B_W => extend(a, 1, 2, false),
        op::SX_B_W => extend(a, 1, 2, true),
        op::ZX_B_Q => extend(a, 1, 8, false),
        op::SX_B_Q => extend(a, 1, 8, true),
        op::ZX_W_Q => extend(a, 2, 8, false),
        op::SX_W_Q => extend(a, 2, 8, true),
        // arithmetic
        op::MUL52LO => map2::<8>(a, b, |x, y| {
            ((x & M52) as u128 * (y & M52) as u128) as u64 & M52
        }),
        op::MUL52HI => map2::<8>(a, b, |x, y| {
            (((x & M52) as u128 * (y & M52) as u128) >> 52) as u64
        }),
        op::ABS_B => map1::<1>(a, |x| sx(x, 8).unsigned_abs()),
        op::ABS_W => map1::<2>(a, |x| sx(x, 16).unsigned_abs()),
        op::ABS_D => map1::<4>(a, |x| sx(x, 32).unsigned_abs()),
        op::ABS_Q => map1::<8>(a, |x| sx(x, 64).unsigned_abs()),
        op::MAX_Q => map2::<8>(a, b, |x, y| sx(x, 64).max(sx(y, 64)) as u64),
        op::MIN_Q => map2::<8>(a, b, |x, y| sx(x, 64).min(sx(y, 64)) as u64),
        op::MAX_B => map2::<1>(a, b, |x, y| sx(x, 8).max(sx(y, 8)) as u64),
        op::MIN_B => map2::<1>(a, b, |x, y| sx(x, 8).min(sx(y, 8)) as u64),
        op::MAXU_B => map2::<1>(a, b, u64::max),
        op::MINU_B => map2::<1>(a, b, u64::min),
        op::ADDS_B => map2::<1>(a, b, |x, y| {
            sat(sx(x, 8) + sx(y, 8), i8::MIN as i64, i8::MAX as i64)
        }),
        op::ADDS_W => map2::<2>(a, b, |x, y| {
            sat(sx(x, 16) + sx(y, 16), i16::MIN as i64, i16::MAX as i64)
        }),
        op::ADDUS_B => map2::<1>(a, b, |x, y| (x + y).min(u8::MAX as u64)),
        op::ADDUS_W => map2::<2>(a, b, |x, y| (x + y).min(u16::MAX as u64)),
        op::SUBS_B => map2::<1>(a, b, |x, y| {
            sat(sx(x, 8) - sx(y, 8), i8::MIN as i64, i8::MAX as i64)
        }),
        op::SUBS_W => map2::<2>(a, b, |x, y| {
            sat(sx(x, 16) - sx(y, 16), i16::MIN as i64, i16::MAX as i64)
        }),
        op::SUBUS_B => map2::<1>(a, b, u64::saturating_sub),
        op::SUBUS_W => map2::<2>(a, b, u64::saturating_sub),
        op::AVG_B => map2::<1>(a, b, |x, y| (x + y + 1) >> 1),
        op::AVG_W => map2::<2>(a, b, |x, y| (x + y + 1) >> 1),
        op::SAD => map2::<8>(a, b, |x, y| {
            (0..8)
                .map(|k| ((x >> (8 * k)) & 0xff).abs_diff((y >> (8 * k)) & 0xff))
                .sum()
        }),
        op::MULHRS_W => map2::<2>(a, b, |x, y| {
            ((((sx(x, 16) * sx(y, 16)) >> 14) + 1) >> 1) as u64
        }),
        op::DPBUSDS => map3::<4>(c, a, b, |z, x, y| {
            sat(
                sx(z, 32) + dot(x, y, 8, 4, false, true),
                i32::MIN as i64,
                i32::MAX as i64,
            )
        }),
        op::DPWSSDS => map3::<4>(c, a, b, |z, x, y| {
            sat(
                sx(z, 32) + dot(x, y, 16, 2, true, true),
                i32::MIN as i64,
                i32::MAX as i64,
            )
        }),
        // bits, masks, compares
        op::TERNLOG_D => return execute_register_op(op::TERNLOG, imm8, c, a, b),
        op::SHLV_W => map2::<2>(a, b, |x, y| if y < 16 { x << y } else { 0 }),
        op::SHRV_W => map2::<2>(a, b, |x, y| if y < 16 { x >> y } else { 0 }),
        op::SARV_W => map2::<2>(a, b, |x, y| (sx(x, 16) >> y.min(15)) as u64),
        op::SARV_D => map2::<4>(a, b, |x, y| (sx(x, 32) >> y.min(31)) as u64),
        op::SARV_Q => map2::<8>(a, b, |x, y| (sx(x, 64) >> y.min(63)) as u64),
        op::SHLDI_W => map2::<2>(a, b, |x, y| shld(x, y, i as u64, 16)),
        op::SHLDI_D => map2::<4>(a, b, |x, y| shld(x, y, i as u64, 32)),
        op::SHLDI_Q => map2::<8>(a, b, |x, y| shld(x, y, i as u64, 64)),
        op::SHRDI_W => map2::<2>(a, b, |x, y| shrd(y, x, i as u64, 16)),
        op::SHRDI_D => map2::<4>(a, b, |x, y| shrd(y, x, i as u64, 32)),
        op::SHRDI_Q => map2::<8>(a, b, |x, y| shrd(y, x, i as u64, 64)),
        op::SHLDV_W => map3::<2>(c, a, b, |z, x, y| shld(z, x, y % 16, 16)),
        op::SHLDV_D => map3::<4>(c, a, b, |z, x, y| shld(z, x, y % 32, 32)),
        op::SHLDV_Q => map3::<8>(c, a, b, |z, x, y| shld(z, x, y % 64, 64)),
        op::SHRDV_W => map3::<2>(c, a, b, |z, x, y| shrd(x, z, y % 16, 16)),
        op::SHRDV_D => map3::<4>(c, a, b, |z, x, y| shrd(x, z, y % 32, 32)),
        op::SHRDV_Q => map3::<8>(c, a, b, |z, x, y| shrd(x, z, y % 64, 64)),
        op::CONFLICT_D => conflict::<4>(a),
        op::CONFLICT_Q => conflict::<8>(a),
        op::LZCNT_D => map1::<4>(a, |x| (x as u32).leading_zeros() as u64),
        op::LZCNT_Q => map1::<8>(a, |x| x.leading_zeros() as u64),
        op::POPCNT_W => map1::<2>(a, |x| x.count_ones() as u64),
        op::POPCNT_D => map1::<4>(a, |x| x.count_ones() as u64),
        op::MULTISHIFT => {
            let mut r = V::ZERO;
            for k in 0..64 {
                r.0[k] = b.lane::<8>(k / 8).rotate_right((a.0[k] & 63) as u32) as u8;
            }
            r
        }
        // crypto
        op::GF2P8MUL => map2::<1>(a, b, |x, y| gf_mul(x as u8, y as u8) as u64),
        op::GF2P8AFFINE | op::GF2P8AFFINEINV => {
            let mut r = V::ZERO;
            for k in 0..VECTOR_SIZE {
                let x = if op == op::GF2P8AFFINE {
                    a.0[k]
                } else {
                    GF_INVERSE[a.0[k] as usize]
                };
                r.0[k] = gf_affine(b.lane::<8>(k / 8), x, imm8);
            }
            r
        }
        op::AESENC => aes_round(a, b, false, false),
        op::AESENCLAST => aes_round(a, b, false, true),
        op::AESDEC => aes_round(a, b, true, false),
        op::AESDECLAST => aes_round(a, b, true, true),
        op::SHA256RNDS2 => {
            let wk = [b.lane::<4>(0) as u32, b.lane::<4>(1) as u32];
            with_low_dwords(c, sha256rnds2(low_dwords(c), low_dwords(a), wk))
        }
        op::SHA256MSG1 => with_low_dwords(c, sha256msg1(low_dwords(c), low_dwords(a))),
        op::SHA1RNDS4 => with_low_dwords(c, sha1rnds4(low_dwords(c), low_dwords(a), imm8)),
        op::SHA256MSG2 => with_low_dwords(c, sha256msg2(low_dwords(c), low_dwords(a))),
        op::SHA1NEXTE => with_low_dwords(c, sha1nexte(low_dwords(c), low_dwords(a))),
        op::SHA1MSG1 => with_low_dwords(c, sha1msg1(low_dwords(c), low_dwords(a))),
        op::SHA1MSG2 => with_low_dwords(c, sha1msg2(low_dwords(c), low_dwords(a))),
        _ => return None,
    })
}

/// Reference semantics of the vector ops which read `r2` as an operand or define their own use
/// of the mask mode (no [apply_mask] afterwards)
pub fn execute_mask_op(op: u8, imm8: u8, mode: u8, c: &V, a: &V, b: &V, r2: u64) -> Option<V> {
    let _ = (mode, c);
    Some(match op {
        // data movement
        op::BLENDM => {
            let bytes = 1 << imm8;
            let mut r = V::ZERO;
            for i in 0..VECTOR_SIZE / bytes {
                let source = if (r2 >> i) & 1 == 1 { b } else { a };
                set_lane(&mut r, bytes, i, lane(source, bytes, i));
            }
            r
        }
        op::COMPRESS | op::EXPAND => {
            let bytes = 1 << imm8;
            let mut r = if mode == 1 { *c } else { V::ZERO };
            let active = (0..VECTOR_SIZE / bytes).filter(|i| (r2 >> i) & 1 == 1);
            for (j, i) in active.enumerate() {
                if op == op::COMPRESS {
                    set_lane(&mut r, bytes, j, lane(a, bytes, i));
                } else {
                    set_lane(&mut r, bytes, i, lane(a, bytes, j));
                }
            }
            r
        }
        // bits, masks, compares
        op::MOVM_B | op::MOVM_W | op::MOVM_D | op::MOVM_Q => {
            let bytes = 1 << (op - op::MOVM_B);
            let mut r = V::ZERO;
            for i in 0..VECTOR_SIZE / bytes {
                set_lane(&mut r, bytes, i, mask_of((r2 >> i) & 1 == 1));
            }
            r
        }
        _ => return None,
    })
}

/// Reference semantics of the ops which write the scalar register `r0`
pub fn execute_scalar_output_op(op: u8, imm8: u8, a: &V, b: &V, r2: u64) -> Option<u64> {
    Some(match op {
        op::HSUM_D => (0..16).fold(0u32, |sum, k| sum.wrapping_add(a.lane::<4>(k) as u32)) as u64,
        op::HSUM_Q => (0..8).fold(0u64, |sum, k| sum.wrapping_add(a.lane::<8>(k))),
        op::EXTRACT_Q => a.lane::<8>(imm8 as usize),
        op::EXTRACT_D => a.lane::<4>(imm8 as usize),
        op::MOVEMASK_B => (0..64).fold(0u64, |mask, k| mask | ((a.0[k] >> 7) as u64) << k),
        op::MPOPCNT => r2.count_ones() as u64,
        op::MTZCNT => r2.trailing_zeros() as u64,
        op::MLZCNT => r2.leading_zeros() as u64,
        // data movement
        op::EXTRACT_B => a.lane::<1>(imm8 as usize),
        op::EXTRACT_W => a.lane::<2>(imm8 as usize),
        // bits, masks, compares
        op::CMP => {
            let bytes = 1 << ((imm8 >> 4) & 3);
            let bits = 8 * bytes as u32;
            let unsigned = (imm8 >> 3) & 1 == 1;
            (0..VECTOR_SIZE / bytes).fold(0, |mask, i| {
                let (x, y) = (lane(a, bytes, i), lane(b, bytes, i));
                let order = if unsigned {
                    x.cmp(&y)
                } else {
                    sx(x, bits).cmp(&sx(y, bits))
                };
                let condition = match imm8 & 7 {
                    0 => order.is_eq(),
                    1 => order.is_lt(),
                    2 => order.is_le(),
                    4 => order.is_ne(),
                    5 => order.is_ge(),
                    _ => order.is_gt(),
                };
                mask | (condition as u64) << i
            })
        }
        op::BITSHUF => (0..64).fold(0, |mask, i| {
            let bit = (a.lane::<8>(i / 8) >> (b.0[i] & 63)) & 1;
            mask | bit << i
        }),
        op::TEST | op::TESTN => {
            let bytes = 1 << imm8;
            (0..VECTOR_SIZE / bytes).fold(0, |mask, i| {
                let nonzero = lane(a, bytes, i) & lane(b, bytes, i) != 0;
                mask | ((nonzero == (op == op::TEST)) as u64) << i
            })
        }
        op::MOVEMASK_W => (0..32).fold(0, |mask, k| mask | (a.lane::<2>(k) >> 15) << k),
        op::MOVEMASK_D => (0..16).fold(0, |mask, k| mask | (a.lane::<4>(k) >> 31) << k),
        op::MOVEMASK_Q => (0..8).fold(0, |mask, k| mask | (a.lane::<8>(k) >> 63) << k),
        _ => return None,
    })
}

/// Reference semantics of the ops which read the scalar register `r1` but not memory
pub fn execute_scalar_input_op(op: u8, imm8: u8, c: &V, r1: u64) -> Option<V> {
    Some(match op {
        op::SPLAT_B => map1::<1>(&V::ZERO, |_| r1),
        op::SPLAT_W => map1::<2>(&V::ZERO, |_| r1),
        op::SPLAT_D => map1::<4>(&V::ZERO, |_| r1),
        op::SPLAT_Q => map1::<8>(&V::ZERO, |_| r1),
        op::INSERT_Q => {
            let mut r = *c;
            r.set_lane::<8>(imm8 as usize, r1);
            r
        }
        // data movement
        op::INSERT_B => {
            let mut r = *c;
            r.set_lane::<1>(imm8 as usize, r1);
            r
        }
        op::INSERT_W => {
            let mut r = *c;
            r.set_lane::<2>(imm8 as usize, r1);
            r
        }
        op::INSERT_D => {
            let mut r = *c;
            r.set_lane::<4>(imm8 as usize, r1);
            r
        }
        op::MOVR_Q => {
            let mut r = V::ZERO;
            r.set_lane::<8>(0, r1);
            r
        }
        _ => return None,
    })
}

/// Maps `len` bytes at `vm_addr` exactly like a scalar access of that size would
fn translate<C: ContextObject>(
    vm: &mut EbpfVm<C>,
    access_type: AccessType,
    vm_addr: u64,
    len: u64,
) -> Result<*mut u8, EbpfError> {
    match vm
        .memory()
        .map_with_access_violation_handler(access_type, vm_addr, len)
    {
        StableResult::Ok(host_buffer) => Ok(host_buffer.ptr().cast::<u8>().cast_mut()),
        StableResult::Err(err) => Err(err),
    }
}

/// Lanes of a gather or scatter: whether the indices are qwords, the element size and the
/// number of lanes
pub(crate) fn gather_scatter_shape(op: u8) -> (bool, usize, usize) {
    match op {
        op::GATHER_QQ | op::SCATTER_QQ => (true, 8, 8),
        op::GATHER_DD | op::SCATTER_DD => (false, 4, 16),
        op::GATHER_DQ | op::SCATTER_DQ => (false, 8, 8),
        op::GATHER_QD | op::SCATTER_QD => (true, 4, 8),
        _ => unreachable!("not a gather or scatter"),
    }
}

/// Gathers and scatters, the per lane model of both engines
///
/// Lane i accesses `base + (index i << imm8)` (wrapping) like a scalar access of the element
/// size. First every active lane is translated in ascending order, the first error ends the op
/// with nothing changed (except for the side effects of the access violation handler). Then
/// the active lanes are read, or written in ascending order (later lanes win on overlaps).
pub(crate) fn gather_scatter<C: ContextObject>(
    vm: &mut EbpfVm<C>,
    v: &VectorInsn,
    base: u64,
    r2: u64,
) -> Result<(), EbpfError> {
    let (qword_index, element, lanes) = gather_scatter_shape(v.info.op);
    let access_type = if v.info.kind == Kind::Ga {
        AccessType::Load
    } else {
        AccessType::Store
    };
    let index = vm.vector_registers[v.in0 as usize];
    let address = |i: usize| {
        let x = if qword_index {
            index.lane::<8>(i)
        } else {
            sx(index.lane::<4>(i), 32) as u64
        };
        base.wrapping_add(x << v.imm8)
    };
    let active = |i: usize| v.mode == 0 || (r2 >> i) & 1 == 1;
    for i in (0..lanes).filter(|i| active(*i)) {
        translate(vm, access_type, address(i), element as u64)?;
    }
    // The access violation handler of a later lane may have moved the region of an earlier one
    let mut host_addresses = [std::ptr::null_mut(); 16];
    for i in (0..lanes).filter(|i| active(*i)) {
        host_addresses[i] = match vm.memory().map(access_type, address(i), element as u64) {
            StableResult::Ok(host_buffer) => host_buffer.ptr().cast::<u8>().cast_mut(),
            StableResult::Err(err) => return Err(err),
        };
    }
    if access_type == AccessType::Load {
        let out = v.out as usize;
        let mut result = if v.mode == 1 {
            vm.vector_registers[out]
        } else {
            V::ZERO
        };
        // The result of gather.qd is the low 256 bits, the rest is 0 in every mode
        result.0[lanes * element..].fill(0);
        for i in (0..lanes).filter(|i| active(*i)) {
            // SAFETY: `map` checked that `element` bytes at the address are readable.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_addresses[i],
                    result.0[i * element..].as_mut_ptr(),
                    element,
                )
            };
        }
        vm.vector_registers[out] = result;
    } else {
        let data = vm.vector_registers[v.in1 as usize];
        for i in (0..lanes).filter(|i| active(*i)) {
            // SAFETY: `map` checked that `element` bytes at the address are writable.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.0[i * element..].as_ptr(),
                    host_addresses[i],
                    element,
                )
            };
        }
    }
    Ok(())
}

/// Interprets one vector op, `reg` are the scalar registers of the interpreter
///
/// On error nothing (vector registers, scalar registers or memory) has been modified, except by
/// the accesses of an `ldn` or `stn` before the failing one.
pub(crate) fn execute<C: ContextObject>(
    vm: &mut EbpfVm<C>,
    reg: &mut [u64; 12],
    insn: &ebpf::Insn,
) -> Result<(), EbpfError> {
    let v = VectorInsn::decode(insn).map_err(|_| EbpfError::UnsupportedInstruction)?;
    let cu = v.info.cu(v.imm8);
    if cu > 1 && vm.loader.get_config().enable_instruction_meter {
        // The op either gets all its CUs or fails before any effect
        if vm.due_insn_count + (cu - 1) > vm.previous_instruction_meter {
            vm.due_insn_count = vm.previous_instruction_meter;
            return Err(EbpfError::ExceededMaxInstructions);
        }
        vm.due_insn_count += cu - 1;
    }
    if let Some(r0) = execute_decoded(vm, &v, reg[1], reg[2])? {
        reg[0] = r0;
    }
    Ok(())
}

/// The effect of a vector op, without its CUs. Returns the new value of `r0` if the op writes it.
///
/// Also the slow path of the JIT for gathers and scatters. On error nothing has been modified,
/// except by the accesses of an `ldn` or `stn` before the failing one.
pub(crate) fn execute_decoded<C: ContextObject>(
    vm: &mut EbpfVm<C>,
    v: &VectorInsn,
    r1: u64,
    r2: u64,
) -> Result<Option<u64>, EbpfError> {
    let (info, op, imm8, mode) = (v.info, v.info.op, v.imm8, v.mode);
    let (out, in0, in1) = (v.out as usize, v.in0 as usize, v.in1 as usize);
    let vm_addr = (r1 as i64).wrapping_add(v.off as i64) as u64;
    let regs = &vm.vector_registers;
    let (c, a, b) = (&regs[out], &regs[in0], &regs[in1]);
    if let Some(result) = execute_mask_op(op, imm8, mode, c, a, b, r2) {
        vm.vector_registers[out] = result;
    } else if let Some(result) = execute_register_op(op, imm8, c, a, b) {
        vm.vector_registers[out] = apply_mask(info, imm8, mode, c, result, r2);
    } else if let Some(result) = execute_scalar_output_op(op, imm8, a, b, r2) {
        return Ok(Some(result));
    } else if let Some(result) = execute_scalar_input_op(op, imm8, c, r1) {
        vm.vector_registers[out] = apply_mask(info, imm8, mode, c, result, r2);
    } else if matches!(info.kind, Kind::Ga | Kind::Sc) {
        gather_scatter(vm, v, vm_addr, r2)?;
    } else {
        let count = v.register_count() as usize;
        let len = match op {
            op::LDBC_D => 4,
            op::LDBC_Q => 8,
            op::LDBC_B => 1,
            op::LDBC_W => 2,
            op::LD_X | op::ST_X | op::LDBC_X => 16,
            op::LD_Y | op::ST_Y | op::LDBC_Y => 32,
            _ => VECTOR_SIZE,
        };
        // `ldn` and `stn` are `count` accesses of one register each, in ascending order
        let vm_addr = |k: usize| vm_addr.wrapping_add((k * VECTOR_SIZE) as u64);
        if matches!(info.kind, Kind::St | Kind::Sti) {
            if op == op::ST_M {
                let host_addr = translate(vm, AccessType::Store, vm_addr(0), len as u64)?;
                let bytes = info.mask_lane_bytes(imm8);
                let value = vm.vector_registers[in0];
                for i in (0..VECTOR_SIZE / bytes).filter(|i| (r2 >> i) & 1 == 1) {
                    // SAFETY: `translate` checked that `len` bytes at `host_addr` are mapped writable.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            value.0[i * bytes..].as_ptr(),
                            host_addr.add(i * bytes),
                            bytes,
                        )
                    };
                }
            } else {
                for k in 0..count {
                    let host_addr = translate(vm, AccessType::Store, vm_addr(k), len as u64)?;
                    let value = vm.vector_registers[in0 + k];
                    // SAFETY: `translate` checked that `len` bytes at `host_addr` are mapped writable.
                    unsafe { std::ptr::copy_nonoverlapping(value.0.as_ptr(), host_addr, len) };
                }
            }
        } else {
            for k in 0..count {
                let host_addr = translate(vm, AccessType::Load, vm_addr(k), len as u64)?;
                let mut bytes = [0u8; VECTOR_SIZE];
                // SAFETY: `translate` checked that `len` bytes at `host_addr` are mapped readable.
                unsafe { std::ptr::copy_nonoverlapping(host_addr, bytes.as_mut_ptr(), len) };
                let element = VectorRegister(bytes);
                let result = match op {
                    op::LDBC_D => map1::<4>(&V::ZERO, |_| element.lane::<4>(0)),
                    op::LDBC_Q => map1::<8>(&V::ZERO, |_| element.lane::<8>(0)),
                    op::LDBC_B => map1::<1>(&V::ZERO, |_| element.lane::<1>(0)),
                    op::LDBC_W => map1::<2>(&V::ZERO, |_| element.lane::<2>(0)),
                    op::LDBC_X => VectorRegister(std::array::from_fn(|i| element.0[i % 16])),
                    op::LDBC_Y => VectorRegister(std::array::from_fn(|i| element.0[i % 32])),
                    _ => element,
                };
                let old = &vm.vector_registers[out + k];
                vm.vector_registers[out + k] = apply_mask(info, imm8, mode, old, result, r2);
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_table() {
        let mut seen = std::collections::HashSet::new();
        for info in OPS {
            assert!(seen.insert(info.op), "duplicate op {:#x}", info.op);
            assert_eq!(op_info(info.op).unwrap().op, info.op);
            assert!(!info.name.contains('_'), "{}", info.name);
        }
        assert_eq!(OPS.len(), 249);
        assert!(op_info(0xf9).is_none() && op_info(0xff).is_none());
    }

    #[test]
    fn test_decode() {
        let word = encode(op::ADD_D, 0, 29, 1, 2);
        assert_eq!(word & 0xff, 0x85);
        assert_eq!((word >> 8) & 0xff, 0x20);
        let insn = ebpf::get_insn(&word.to_le_bytes(), 0);
        let v = VectorInsn::decode(&insn).unwrap();
        assert_eq!((v.info.op, v.out, v.in0, v.in1), (op::ADD_D, 29, 1, 2));
        assert_eq!(v.registers_used(), 1 << 29 | 1 << 1 | 1 << 2);
        assert_eq!(v.to_string(), "vec.add.d v29, v1, v2");
        let insn = ebpf::get_insn(&encode(op::EXTRACT_Q, 7, 0, 3, 0).to_le_bytes(), 0);
        assert_eq!(
            VectorInsn::decode(&insn).unwrap().to_string(),
            "vec.extract.q r0, v3, 0x7"
        );
        let decode = |word: u64| VectorInsn::decode(&ebpf::get_insn(&word.to_le_bytes(), 0));
        let v = decode(encode_with_offset(op::ST, 0, 0, 5, 0, -64)).unwrap();
        assert_eq!((v.off, v.to_string().as_str()), (-64, "vec.st v5, [r1-64]"));
        let v = decode(encode_with_offset(op::LD, 0, 2, 0, 0, 0x100)).unwrap();
        assert_eq!(v.to_string(), "vec.ld v2, [r1+256]");
        assert_eq!(
            decode(encode_with_offset(op::SPLAT_D, 0, 2, 0, 0, 8)),
            Err("off must be 0")
        );
    }
}
