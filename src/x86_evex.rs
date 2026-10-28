//! Minimal EVEX (AVX-512) machinecode encoder for the vector op lowering of the JIT
//!
//! Covers exactly the forms used in `vector_jit.rs`: register operands (zmm/ymm/xmm 0-31, k0-7
//! and general purpose registers in the ModRM.rm field) and memory operands of the form
//! `[base + disp32]` without index. There is no masking, broadcast or disp8*N compression.

#![allow(clippy::arithmetic_side_effects)]

use crate::x86::X86Register;

/// Mandatory prefix (EVEX.pp)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prefix {
    /// 0x66
    P66 = 1,
    /// 0xF3
    PF3 = 2,
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

    /// Encodes the instruction, returns the buffer and the number of valid bytes in it
    pub(crate) fn encode(&self) -> ([u8; MAX_EVEX_INSTRUCTION_LENGTH], usize) {
        debug_assert!(self.reg < 32 && self.vvvv < 32);
        let mut bytes = [0u8; MAX_EVEX_INSTRUCTION_LENGTH];
        // Register extension bits as stored (inverted) in P0
        let (x, b) = match self.rm {
            Rm::Reg(rm) => {
                debug_assert!(rm < 32);
                ((rm >> 4) & 1, (rm >> 3) & 1)
            }
            Rm::Mem(base, _) => (0, (base as u8 >> 3) & 1),
        };
        let r = (self.reg >> 3) & 1;
        let r_prime = (self.reg >> 4) & 1;
        bytes[0] = 0x62;
        bytes[1] = (r ^ 1) << 7 | (x ^ 1) << 6 | (b ^ 1) << 5 | (r_prime ^ 1) << 4 | self.map as u8;
        bytes[2] = (self.w as u8) << 7 | (!self.vvvv & 0xf) << 3 | 1 << 2 | self.prefix as u8;
        bytes[3] = (self.length as u8) << 5 | (((self.vvvv >> 4) & 1) ^ 1) << 3;
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
        }
        if let Some(imm8) = self.imm8 {
            bytes[len] = imm8;
            len += 1;
        }
        (bytes, len)
    }
}

/// `kmovq r64, k` (VEX.L0.F2.0F.W1 93 /r)
pub(crate) const fn kmovq_to_gpr(destination: X86Register, mask: u8) -> [u8; 5] {
    let destination = destination as u8;
    [
        0xc4,
        (((destination >> 3) & 1) ^ 1) << 7 | 0x60 | 0x01,
        0xfb,
        0x93,
        0xc0 | (destination & 7) << 3 | (mask & 7),
    ]
}

/// `vzeroupper` (VEX.128.0F.WIG 77)
pub(crate) const VZEROUPPER: [u8; 3] = [0xc5, 0xf8, 0x77];
