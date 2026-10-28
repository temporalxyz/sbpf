//! Minimal EVEX (AVX-512) machinecode encoder for the vector op lowering of the JIT
//!
//! Covers exactly the forms used in `vector_jit.rs`: register operands (zmm/ymm/xmm 0-31, k0-7
//! and general purpose registers in the ModRM.rm field), memory operands of the form
//! `[base + disp32]` and VSIB operands, with opmask and zeroing. There is no broadcast or disp8*N
//! compression. VEX encoded opmask instructions and legacy encoded instructions (SHA-NI, scalar
//! bit counts) are at the end.

#![allow(clippy::arithmetic_side_effects)]

use crate::x86::X86Register;

/// Mandatory prefix (EVEX.pp)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prefix {
    /// No prefix (vshufps)
    None = 0,
    /// 0x66
    P66 = 1,
    /// 0xF3
    PF3 = 2,
    /// 0xF2 (vpshuflw, vmovdqu8/16)
    PF2 = 3,
}

/// Opcode map (EVEX.mmm)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Map {
    /// 0x0F
    M0F = 1,
    /// 0x0F 0x38
    M0F38 = 2,
    /// 0x0F 0x3A
    M0F3A = 3,
}

/// Vector length (EVEX.L'L)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Length {
    /// xmm
    L128 = 0,
    /// ymm
    L256 = 1,
    /// zmm
    L512 = 2,
}

/// The ModRM.rm operand
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rm {
    /// A vector (0-31), mask (0-7) or general purpose (0-15) register
    Reg(u8),
    /// `[base + disp]`
    Mem(X86Register, i32),
    /// `[base + index * 2^scale + disp]` with a vector index register (0-31)
    Vsib {
        base: X86Register,
        index: u8,
        scale: u8,
        disp: i32,
    },
}

/// One EVEX encoded instruction
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvexInstruction {
    pub prefix: Prefix,
    pub map: Map,
    pub w: bool,
    pub length: Length,
    pub opcode: u8,
    /// ModRM.reg: a register (0-31) or an opcode extension (0-7)
    pub reg: u8,
    /// EVEX.vvvv: second source register (0-31), 0 if unused
    pub vvvv: u8,
    pub rm: Rm,
    pub imm8: Option<u8>,
    /// Opmask register (EVEX.aaa), k0 = unmasked
    pub mask: u8,
    /// Masked-off lanes are zeroed instead of merged (EVEX.z)
    pub zeroing: bool,
}

/// Maximum length of an encoded instruction (EVEX + opcode + ModRM + SIB + disp32 + imm8)
pub(crate) const MAX_EVEX_INSTRUCTION_LENGTH: usize = 12;

impl EvexInstruction {
    /// 512-bit `opcode reg, vvvv, rm` (Intel operand order)
    pub(crate) const fn rrr(
        prefix: Prefix,
        map: Map,
        w: bool,
        opcode: u8,
        reg: u8,
        vvvv: u8,
        rm: u8,
    ) -> Self {
        Self {
            prefix,
            map,
            w,
            length: Length::L512,
            opcode,
            reg,
            vvvv,
            rm: Rm::Reg(rm),
            imm8: None,
            mask: 0,
            zeroing: false,
        }
    }

    /// 512-bit `opcode reg, rm` (no EVEX.vvvv operand)
    pub(crate) const fn rr(prefix: Prefix, map: Map, w: bool, opcode: u8, reg: u8, rm: u8) -> Self {
        Self::rrr(prefix, map, w, opcode, reg, 0, rm)
    }

    /// Sets the vector length
    pub(crate) const fn length(mut self, length: Length) -> Self {
        self.length = length;
        self
    }

    /// Appends an immediate byte
    pub(crate) const fn imm(mut self, imm8: u8) -> Self {
        self.imm8 = Some(imm8);
        self
    }

    /// Replaces the ModRM.rm operand with a memory operand
    pub(crate) const fn mem(mut self, base: X86Register, disp: i32) -> Self {
        self.rm = Rm::Mem(base, disp);
        self
    }

    /// Replaces the ModRM.rm operand with a VSIB memory operand
    pub(crate) const fn vsib(mut self, base: X86Register, index: u8, scale: u8, disp: i32) -> Self {
        self.rm = Rm::Vsib {
            base,
            index,
            scale,
            disp,
        };
        self
    }

    /// Sets the opmask register (k1-k7) and whether masked-off lanes are zeroed
    pub(crate) const fn mask(mut self, mask: u8, zeroing: bool) -> Self {
        self.mask = mask;
        self.zeroing = zeroing;
        self
    }

    /// Encodes the instruction, returns the buffer and the number of valid bytes in it
    pub(crate) fn encode(&self) -> ([u8; MAX_EVEX_INSTRUCTION_LENGTH], usize) {
        debug_assert!(self.reg < 32 && self.vvvv < 32);
        debug_assert!(self.mask < 8);
        let mut bytes = [0u8; MAX_EVEX_INSTRUCTION_LENGTH];
        // Register extension bits as stored (inverted) in P0
        let (x, b) = match self.rm {
            Rm::Reg(rm) => {
                debug_assert!(rm < 32);
                ((rm >> 4) & 1, (rm >> 3) & 1)
            }
            Rm::Mem(base, _) => (0, (base as u8 >> 3) & 1),
            // X and V' extend the index register
            Rm::Vsib { base, index, .. } => {
                debug_assert!(index < 32);
                ((index >> 3) & 1, (base as u8 >> 3) & 1)
            }
        };
        let r = (self.reg >> 3) & 1;
        let r_prime = (self.reg >> 4) & 1;
        let v_prime = match self.rm {
            Rm::Vsib { index, .. } => (index >> 4) & 1,
            _ => (self.vvvv >> 4) & 1,
        };
        bytes[0] = 0x62;
        bytes[1] = (r ^ 1) << 7 | (x ^ 1) << 6 | (b ^ 1) << 5 | (r_prime ^ 1) << 4 | self.map as u8;
        bytes[2] = (self.w as u8) << 7 | (!self.vvvv & 0xf) << 3 | 1 << 2 | self.prefix as u8;
        bytes[3] =
            (self.zeroing as u8) << 7 | (self.length as u8) << 5 | (v_prime ^ 1) << 3 | self.mask;
        bytes[4] = self.opcode;
        let mut len = 5;
        match self.rm {
            Rm::Reg(rm) => {
                bytes[len] = 0xc0 | (self.reg & 7) << 3 | (rm & 7);
                len += 1;
            }
            Rm::Mem(base, disp) => {
                let base = base as u8 & 7;
                // mod = 00 has no displacement, except for rbp / r13 which then mean rip / disp32
                let mode = if disp == 0 && base != 5 { 0 } else { 2 };
                bytes[len] = mode << 6 | (self.reg & 7) << 3 | base;
                len += 1;
                if base == 4 {
                    // SIB: no index, base = rsp / r12
                    bytes[len] = 0x24;
                    len += 1;
                }
                if mode == 2 {
                    bytes[len..len + 4].copy_from_slice(&disp.to_le_bytes());
                    len += 4;
                }
            }
            Rm::Vsib {
                base,
                index,
                scale,
                disp,
            } => {
                let base = base as u8 & 7;
                let mode = if disp == 0 && base != 5 { 0 } else { 2 };
                bytes[len] = mode << 6 | (self.reg & 7) << 3 | 0b100;
                len += 1;
                bytes[len] = (scale & 3) << 6 | (index & 7) << 3 | base;
                len += 1;
                if mode == 2 {
                    bytes[len..len + 4].copy_from_slice(&disp.to_le_bytes());
                    len += 4;
                }
            }
        }
        if let Some(imm8) = self.imm8 {
            bytes[len] = imm8;
            len += 1;
        }
        (bytes, len)
    }
}

/// Three byte VEX encoded instruction of map 0F with register operands
pub(crate) const fn vex(
    prefix: u8,
    w: bool,
    l: bool,
    opcode: u8,
    reg: u8,
    vvvv: u8,
    rm: u8,
) -> [u8; 5] {
    [
        0xc4,
        (((reg >> 3) & 1) ^ 1) << 7 | 1 << 6 | (((rm >> 3) & 1) ^ 1) << 5 | 0x01,
        (w as u8) << 7 | (!vvvv & 0xf) << 3 | (l as u8) << 2 | prefix,
        opcode,
        0xc0 | (reg & 7) << 3 | (rm & 7),
    ]
}

/// `kmovq r64, k` (VEX.L0.F2.0F.W1 93 /r)
pub(crate) const fn kmovq_to_gpr(destination: X86Register, mask: u8) -> [u8; 5] {
    vex(3, true, false, 0x93, destination as u8, 0, mask)
}

/// `kmovq k, r64` (VEX.L0.F2.0F.W1 92 /r)
pub(crate) const fn kmovq_from_gpr(mask: u8, source: X86Register) -> [u8; 5] {
    vex(3, true, false, 0x92, mask, 0, source as u8)
}

/// `kxnorq k, k, k` (VEX.L1.0F.W1 46 /r)
pub(crate) const fn kxnorq(destination: u8, a: u8, b: u8) -> [u8; 5] {
    vex(0, true, true, 0x46, destination, a, b)
}

/// `kortestb k, k` (VEX.L0.66.0F.W0 98 /r) or, for more than 8 lanes, `kortestw` (VEX.L0.0F.W0 98 /r)
pub(crate) const fn kortest(lanes: usize, a: u8, b: u8) -> [u8; 5] {
    vex(if lanes <= 8 { 1 } else { 0 }, false, false, 0x98, a, 0, b)
}

/// Legacy (non-VEX) encoded instruction with register operands: SHA-NI and the scalar bit counts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LegacyInstruction {
    /// Mandatory prefix byte, if any
    pub prefix: Option<u8>,
    pub rex_w: bool,
    pub map: Map,
    pub opcode: u8,
    /// ModRM.reg (xmm0-15 or general purpose register)
    pub reg: u8,
    /// ModRM.rm (xmm0-15 or general purpose register)
    pub rm: u8,
    pub imm8: Option<u8>,
}

impl LegacyInstruction {
    /// Encodes the instruction, returns the buffer and the number of valid bytes in it
    pub(crate) fn encode(&self) -> ([u8; 9], usize) {
        debug_assert!(self.reg < 16 && self.rm < 16);
        let mut bytes = [0u8; 9];
        let mut len = 0;
        let mut push = |byte: u8| {
            bytes[len] = byte;
            len += 1;
        };
        if let Some(prefix) = self.prefix {
            push(prefix);
        }
        let rex = (self.rex_w as u8) << 3 | ((self.reg >> 3) & 1) << 2 | ((self.rm >> 3) & 1);
        if rex != 0 {
            push(0x40 | rex);
        }
        push(0x0f);
        match self.map {
            Map::M0F => {}
            Map::M0F38 => push(0x38),
            Map::M0F3A => push(0x3a),
        }
        push(self.opcode);
        push(0xc0 | (self.reg & 7) << 3 | (self.rm & 7));
        if let Some(imm8) = self.imm8 {
            push(imm8);
        }
        (bytes, len)
    }
}

/// `vzeroupper` (VEX.128.0F.WIG 77)
pub(crate) const VZEROUPPER: [u8; 3] = [0xc5, 0xf8, 0x77];
