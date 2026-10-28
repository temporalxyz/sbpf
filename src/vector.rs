#![allow(clippy::arithmetic_side_effects)]
//! 512-bit vector ops
//!
//! A vector op is a `CALL_IMM` with `src = 2` and `dst = 0`. `off` is a signed displacement added
//! to `r1` for `ld`, `st`, `ldbc.d` and `ldbc.q`, and must be 0 for every other op. The
//! immediate holds:
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
}

impl Kind {
    /// Whether `out` names a register (read or written)
    pub const fn uses_out(self) -> bool {
        !matches!(self, Kind::St | Kind::Sr | Kind::Sri)
    }
    /// Whether `out` is also read
    pub const fn reads_out(self) -> bool {
        matches!(self, Kind::A3 | Kind::A3i | Kind::Ai)
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
        )
    }
    /// Whether `in1` is read
    pub const fn uses_in1(self) -> bool {
        matches!(self, Kind::R3 | Kind::A3 | Kind::A3i | Kind::R3i)
    }
    /// Whether imm8 is used
    pub const fn uses_imm8(self) -> bool {
        matches!(
            self,
            Kind::A3i | Kind::R3i | Kind::R2i | Kind::Sri | Kind::Ai
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
    imm: Imm,
}

impl OpInfo {
    /// Whether `imm8` is valid for this op
    pub const fn imm8_is_valid(&self, imm8: u8) -> bool {
        match self.imm {
            Imm::Zero => imm8 == 0,
            Imm::Any => true,
            Imm::Below(n) => imm8 < n,
            Imm::Clmul => matches!(imm8, 0x00 | 0x01 | 0x10 | 0x11),
        }
    }

    /// Whether the op accesses memory at `r1 + off` (`ld`, `st`, `ldbc.d`, `ldbc.q`)
    pub const fn accesses_memory(&self) -> bool {
        matches!(self.op, op::LD | op::ST | op::LDBC_D | op::LDBC_Q)
    }
}

macro_rules! ops {
    ($($code:literal $name:ident $mnemonic:literal $kind:ident $imm:expr;)*) => {
        /// Op codes
        #[allow(missing_docs)]
        pub mod op {
            $(pub const $name: u8 = $code;)*
        }
        /// All defined ops
        pub const OPS: &[OpInfo] = &[
            $(OpInfo { op: $code, name: $mnemonic, kind: Kind::$kind, imm: $imm },)*
        ];
    };
}

use Imm::{Any, Below, Clmul, Zero};
ops! {
    0x00 LD "ld" Ls Zero;
    0x01 ST "st" St Zero;
    0x02 LDBC_D "ldbc.d" Ls Zero;
    0x03 LDBC_Q "ldbc.q" Ls Zero;
    0x04 SPLAT_B "splat.b" Ls Zero;
    0x05 SPLAT_W "splat.w" Ls Zero;
    0x06 SPLAT_D "splat.d" Ls Zero;
    0x07 SPLAT_Q "splat.q" Ls Zero;
    0x08 ZERO "zero" R0 Zero;
    0x09 MOV "mov" R2 Zero;
    0x10 ADD_B "add.b" R3 Zero;
    0x11 ADD_W "add.w" R3 Zero;
    0x12 ADD_D "add.d" R3 Zero;
    0x13 ADD_Q "add.q" R3 Zero;
    0x14 SUB_B "sub.b" R3 Zero;
    0x15 SUB_W "sub.w" R3 Zero;
    0x16 SUB_D "sub.d" R3 Zero;
    0x17 SUB_Q "sub.q" R3 Zero;
    0x18 MULLO_W "mullo.w" R3 Zero;
    0x19 MULLO_D "mullo.d" R3 Zero;
    0x1a MULLO_Q "mullo.q" R3 Zero;
    0x1b MULWU "mulwu" R3 Zero;
    0x1c MULWS "mulws" R3 Zero;
    0x1d MULHI_W "mulhi.w" R3 Zero;
    0x1e MULHIU_W "mulhiu.w" R3 Zero;
    0x20 MAX_D "max.d" R3 Zero;
    0x21 MIN_D "min.d" R3 Zero;
    0x22 MAXU_D "maxu.d" R3 Zero;
    0x23 MINU_D "minu.d" R3 Zero;
    0x24 MAX_W "max.w" R3 Zero;
    0x25 MIN_W "min.w" R3 Zero;
    0x26 MAXU_Q "maxu.q" R3 Zero;
    0x27 MINU_Q "minu.q" R3 Zero;
    0x28 AND "and" R3 Zero;
    0x29 OR "or" R3 Zero;
    0x2a XOR "xor" R3 Zero;
    0x2b ANDN "andn" R3 Zero;
    0x2c TERNLOG "ternlog" A3i Any;
    0x2e MAXU_W "maxu.w" R3 Zero;
    0x2f MINU_W "minu.w" R3 Zero;
    0x30 SHLI_W "shli.w" R2i Below(16);
    0x31 SHLI_D "shli.d" R2i Below(32);
    0x32 SHLI_Q "shli.q" R2i Below(64);
    0x33 SHRI_W "shri.w" R2i Below(16);
    0x34 SHRI_D "shri.d" R2i Below(32);
    0x35 SHRI_Q "shri.q" R2i Below(64);
    0x36 SARI_W "sari.w" R2i Below(16);
    0x37 SARI_D "sari.d" R2i Below(32);
    0x38 SARI_Q "sari.q" R2i Below(64);
    0x39 ROLI_D "roli.d" R2i Below(32);
    0x3a ROLI_Q "roli.q" R2i Below(64);
    0x3b SHLV_D "shlv.d" R3 Zero;
    0x3c SHLV_Q "shlv.q" R3 Zero;
    0x3d SHRV_D "shrv.d" R3 Zero;
    0x3e SHRV_Q "shrv.q" R3 Zero;
    0x3f ROLV_D "rolv.d" R3 Zero;
    0x40 ROLV_Q "rolv.q" R3 Zero;
    0x48 CMPEQ_B "cmpeq.b" R3 Zero;
    0x49 CMPEQ_D "cmpeq.d" R3 Zero;
    0x4a CMPEQ_Q "cmpeq.q" R3 Zero;
    0x4b CMPGT_D "cmpgt.d" R3 Zero;
    0x4c CMPGT_Q "cmpgt.q" R3 Zero;
    0x4d CMPGTU_Q "cmpgtu.q" R3 Zero;
    0x50 DPBUSD "dpbusd" A3 Zero;
    0x51 DPWSSD "dpwssd" A3 Zero;
    0x52 MADDUBS "maddubs" R3 Zero;
    0x53 MADDWD "maddwd" R3 Zero;
    0x54 MADD52LO "madd52lo" A3 Zero;
    0x55 MADD52HI "madd52hi" A3 Zero;
    0x58 PACKS_D "packs.d" R3 Zero;
    0x59 PACKS_W "packs.w" R3 Zero;
    0x5a PACKUS_W "packus.w" R3 Zero;
    0x5b PERMD "permd" R3 Zero;
    0x5c PERMQ "permq" R3 Zero;
    0x5d SHUFB "shufb" R3 Zero;
    0x5e PERM2_W "perm2.w" A3 Zero;
    0x60 NARROWUS_D_B "narrowus.d.b" R2 Zero;
    0x61 NARROWS_D_B "narrows.d.b" R2 Zero;
    0x62 NARROWS_D_W "narrows.d.w" R2 Zero;
    0x63 NARROW_D_B "narrow.d.b" R2 Zero;
    0x64 ZX_B_D "zx.b.d" R2 Zero;
    0x65 SX_B_D "sx.b.d" R2 Zero;
    0x66 ZX_W_D "zx.w.d" R2 Zero;
    0x67 SX_W_D "sx.w.d" R2 Zero;
    0x68 ZX_D_Q "zx.d.q" R2 Zero;
    0x69 SX_D_Q "sx.d.q" R2 Zero;
    0x6a POPCNT_B "popcnt.b" R2 Zero;
    0x6b POPCNT_Q "popcnt.q" R2 Zero;
    0x70 HSUM_D "hsum.d" Sr Zero;
    0x71 HSUM_Q "hsum.q" Sr Zero;
    0x72 EXTRACT_Q "extract.q" Sri Below(8);
    0x73 EXTRACT_D "extract.d" Sri Below(16);
    0x74 INSERT_Q "insert.q" Ai Below(8);
    0x75 MOVEMASK_B "movemask.b" Sr Zero;
    0x78 CLMUL "clmul" R3i Clmul;
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
    let imm = (op as u32) << 24
        | (imm8 as u32) << 16
        | (out as u32 & 0x1f) << 10
        | (in0 as u32 & 0x1f) << 5
        | (in1 as u32 & 0x1f);
    ebpf::CALL_IMM as u64
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
        if insn.dst != 0 {
            return Err("dst must be 0");
        }
        if imm & (1 << 15) != 0 {
            return Err("imm bit 15 must be 0");
        }
        let info = op_info((imm >> 24) as u8).ok_or("unknown op")?;
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
        Ok(Self {
            info,
            imm8,
            out,
            in0,
            in1,
            off: insn.off,
        })
    }

    /// Bit set of the vector registers this op reads or writes
    pub fn registers_used(&self) -> u32 {
        let kind = self.info.kind;
        (kind.uses_out() as u32) << self.out
            | (kind.uses_in0() as u32) << self.in0
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
        if kind.uses_out() {
            operand(f, format!("v{}", self.out))?;
        }
        if matches!(kind, Kind::Sr | Kind::Sri) {
            operand(f, "r0".to_string())?;
        }
        if kind.uses_in0() {
            operand(f, format!("v{}", self.in0))?;
        }
        if kind.uses_in1() {
            operand(f, format!("v{}", self.in1))?;
        }
        if self.info.accesses_memory() {
            operand(f, format!("[r1{:+}]", self.off))?;
        } else if matches!(kind, Kind::Ls | Kind::Ai) {
            operand(f, "r1".to_string())?;
        }
        if kind.uses_imm8() {
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

/// Bit set of the vector registers used by any vector op in `prog`, 0 if there are none
pub(crate) fn registers_used_by_program(prog: &[u8], sbpf_version: SBPFVersion) -> u32 {
    let mut mask = 0;
    let mut insn_ptr: usize = 0;
    while (insn_ptr + 1) * ebpf::INSN_SIZE <= prog.len() {
        let insn = ebpf::get_insn_unchecked(prog, insn_ptr);
        if insn.opc == ebpf::LD_DW_IMM && !sbpf_version.disable_lddw() {
            insn_ptr += 1;
        } else if is_vector_op(&insn, true, sbpf_version) {
            if let Ok(vector_insn) = VectorInsn::decode(&insn) {
                mask |= vector_insn.registers_used();
            }
        }
        insn_ptr += 1;
    }
    mask
}

// Portable reference semantics

type V = VectorRegister;

fn map1<const W: usize>(a: &V, f: impl Fn(u64) -> u64) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / W {
        r.set_lane::<W>(i, f(a.lane::<W>(i)));
    }
    r
}

fn map2<const W: usize>(a: &V, b: &V, f: impl Fn(u64, u64) -> u64) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / W {
        r.set_lane::<W>(i, f(a.lane::<W>(i), b.lane::<W>(i)));
    }
    r
}

fn map3<const W: usize>(c: &V, a: &V, b: &V, f: impl Fn(u64, u64, u64) -> u64) -> V {
    let mut r = V::ZERO;
    for i in 0..VECTOR_SIZE / W {
        r.set_lane::<W>(i, f(c.lane::<W>(i), a.lane::<W>(i), b.lane::<W>(i)));
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
        _ => return None,
    })
}

/// Reference semantics of the ops which write the scalar register `r0`
pub fn execute_scalar_output_op(op: u8, imm8: u8, a: &V) -> Option<u64> {
    Some(match op {
        op::HSUM_D => (0..16).fold(0u32, |sum, k| sum.wrapping_add(a.lane::<4>(k) as u32)) as u64,
        op::HSUM_Q => (0..8).fold(0u64, |sum, k| sum.wrapping_add(a.lane::<8>(k))),
        op::EXTRACT_Q => a.lane::<8>(imm8 as usize),
        op::EXTRACT_D => a.lane::<4>(imm8 as usize),
        op::MOVEMASK_B => (0..64).fold(0u64, |mask, k| mask | ((a.0[k] >> 7) as u64) << k),
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

/// Interprets one vector op, `reg` are the scalar registers of the interpreter
///
/// On error nothing (vector registers, scalar registers or memory) has been modified.
pub(crate) fn execute<C: ContextObject>(
    vm: &mut EbpfVm<C>,
    reg: &mut [u64; 12],
    insn: &ebpf::Insn,
) -> Result<(), EbpfError> {
    let v = VectorInsn::decode(insn).map_err(|_| EbpfError::UnsupportedInstruction)?;
    let (op, imm8) = (v.info.op, v.imm8);
    let (out, in0, in1) = (v.out as usize, v.in0 as usize, v.in1 as usize);
    let vm_addr = (reg[1] as i64).wrapping_add(v.off as i64) as u64;
    let regs = &vm.vector_registers;
    if let Some(result) = execute_register_op(op, imm8, &regs[out], &regs[in0], &regs[in1]) {
        vm.vector_registers[out] = result;
    } else if let Some(result) = execute_scalar_output_op(op, imm8, &regs[in0]) {
        reg[0] = result;
    } else if let Some(result) = execute_scalar_input_op(op, imm8, &regs[out], reg[1]) {
        vm.vector_registers[out] = result;
    } else {
        let len = match op {
            op::LDBC_D => 4,
            op::LDBC_Q => 8,
            _ => VECTOR_SIZE,
        };
        if op == op::ST {
            let host_addr = translate(vm, AccessType::Store, vm_addr, len as u64)?;
            let value = vm.vector_registers[in0];
            // SAFETY: `translate` checked that `len` bytes at `host_addr` are mapped writable.
            unsafe { std::ptr::copy_nonoverlapping(value.0.as_ptr(), host_addr, len) };
        } else {
            let host_addr = translate(vm, AccessType::Load, vm_addr, len as u64)?;
            let mut bytes = [0u8; VECTOR_SIZE];
            // SAFETY: `translate` checked that `len` bytes at `host_addr` are mapped readable.
            unsafe { std::ptr::copy_nonoverlapping(host_addr, bytes.as_mut_ptr(), len) };
            let element = VectorRegister(bytes);
            vm.vector_registers[out] = match op {
                op::LD => element,
                op::LDBC_D => map1::<4>(&V::ZERO, |_| element.lane::<4>(0)),
                _ => map1::<8>(&V::ZERO, |_| element.lane::<8>(0)),
            };
        }
    }
    Ok(())
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
        assert_eq!(OPS.len(), 95);
        assert!(op_info(0x0a).is_none() && op_info(0xff).is_none());
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
