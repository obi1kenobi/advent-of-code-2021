use std::fmt::Display;

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1, line_ending, one_of, space1},
    combinator::{map, map_res, opt, recognize},
    multi::many1,
    sequence::tuple,
    IResult,
};

#[derive(Debug, Clone, Copy)]
pub struct Register(pub usize);

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let letter = match self.0 {
            0 => "w",
            1 => "x",
            2 => "y",
            3 => "z",
            _ => unreachable!("{:?}", self),
        };

        write!(f, "{}", letter)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Operand {
    Literal(i64),
    Register(Register),
}

impl Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Literal(l) => write!(f, "{}", l),
            Operand::Register(r) => write!(f, "{}", *r),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    Input(Register),
    Add(Register, Operand),
    Mul(Register, Operand),
    Div(Register, Operand),
    Mod(Register, Operand),
    Equal(Register, Operand),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Input(r) => write!(f, "inp {}", *r),
            Instruction::Add(r, o) => write!(f, "add {} {}", *r, *o),
            Instruction::Mul(r, o) => write!(f, "mul {} {}", *r, *o),
            Instruction::Div(r, o) => write!(f, "div {} {}", *r, *o),
            Instruction::Mod(r, o) => write!(f, "mod {} {}", *r, *o),
            Instruction::Equal(r, o) => write!(f, "eql {} {}", *r, *o),
        }
    }
}

pub struct InstructionStream<'a>(pub &'a [Instruction]);

impl<'a> Display for InstructionStream<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for instr in self.0.iter() {
            writeln!(f, "{}", instr)?;
        }
        Ok(())
    }
}

impl Instruction {
    #[allow(dead_code)]
    #[inline]
    pub fn destination(&self) -> Register {
        *match self {
            Instruction::Input(r) => r,
            Instruction::Add(r, _) => r,
            Instruction::Mul(r, _) => r,
            Instruction::Div(r, _) => r,
            Instruction::Mod(r, _) => r,
            Instruction::Equal(r, _) => r,
        }
    }

    #[inline]
    pub fn operand(&self) -> Option<Operand> {
        match self {
            Instruction::Input(_) => None,
            Instruction::Add(_, o) => Some(*o),
            Instruction::Mul(_, o) => Some(*o),
            Instruction::Div(_, o) => Some(*o),
            Instruction::Mod(_, o) => Some(*o),
            Instruction::Equal(_, o) => Some(*o),
        }
    }
}

fn register(input: &str) -> IResult<&str, Register> {
    let (remainder, matched_char) = one_of("wxyz")(input)?;
    let register_id = match matched_char {
        'w' => 0,
        'x' => 1,
        'y' => 2,
        'z' => 3,
        _ => unreachable!("{}", matched_char),
    };

    Ok((remainder, Register(register_id)))
}

fn text_signed_int(input: &str) -> IResult<&str, i64> {
    map_res(recognize(tuple((opt(char('-')), digit1))), |value: &str| {
        value.parse()
    })(input)
}

fn operand(input: &str) -> IResult<&str, Operand> {
    if let Ok((remainder, register)) = register(input) {
        Ok((remainder, Operand::Register(register)))
    } else {
        map(text_signed_int, Operand::Literal)(input)
    }
}

fn input_instruction(input: &str) -> IResult<&str, Instruction> {
    map(
        tuple((tag("inp"), space1, register, opt(line_ending))),
        |(_, _, reg, _)| Instruction::Input(reg),
    )(input)
}

fn binary_instruction(input: &str) -> IResult<&str, Instruction> {
    map(
        tuple((
            alt((tag("add"), tag("mul"), tag("div"), tag("mod"), tag("eql"))),
            space1,
            register,
            space1,
            operand,
            opt(line_ending),
        )),
        |(instr, _, reg, _, val, _)| match instr {
            "add" => Instruction::Add(reg, val),
            "mul" => Instruction::Mul(reg, val),
            "div" => Instruction::Div(reg, val),
            "mod" => Instruction::Mod(reg, val),
            "eql" => Instruction::Equal(reg, val),
            _ => unreachable!("{}", instr),
        },
    )(input)
}

fn instruction(input: &str) -> IResult<&str, Instruction> {
    alt((input_instruction, binary_instruction))(input)
}

pub fn parse_program(input: &str) -> Vec<Instruction> {
    let (remainder, program) = many1(instruction)(input).unwrap();
    assert!(remainder.is_empty());
    program
}
