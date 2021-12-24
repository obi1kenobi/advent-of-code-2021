use std::{env, fs};

#[allow(unused_imports)]
use itertools::Itertools;
use parser::{parse_program, Instruction, Operand, Register};

mod parser;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let input_data: Vec<Instruction> = parse_program(content.as_str());

    match part {
        "1" => {
            let result = solve_part1(&input_data);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(&input_data);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

#[derive(Debug, Clone)]
struct ProvidedInputs {
    values: Vec<i64>,
}

impl ProvidedInputs {
    const MAX_COUNT: usize = 14;

    fn new() -> Self {
        Self {
            values: Vec::with_capacity(ProvidedInputs::MAX_COUNT),
        }
    }

    fn generate_next(self) -> impl Iterator<Item = (i64, Self)> {
        (1..=9).rev().map(move |value| {
            let mut next = self.clone();
            next.values.push(value);
            assert!(next.values.len() <= ProvidedInputs::MAX_COUNT);
            (value, next)
        })
    }
}

#[derive(Debug, Clone)]
struct Simulation<'a> {
    registers: [i64; 4],
    remaining: &'a [Instruction],
    inputs: ProvidedInputs,
}

impl<'a> Simulation<'a> {
    fn new(instructions: &'a [Instruction]) -> Self {
        Self {
            registers: [0; 4],
            remaining: instructions,
            inputs: ProvidedInputs::new(),
        }
    }

    fn is_valid(&self) -> Option<bool> {
        if self.remaining.is_empty() {
            Some(*self.registers.last().unwrap() == 0)
        } else {
            None
        }
    }

    fn advance_until_input(mut self) -> Option<Self> {
        while let Some(instr) = self.remaining.first() {
            if matches!(instr, Instruction::Input(_)) {
                break;
            }

            self.remaining = &self.remaining[1..];
            match instr {
                Instruction::Add(Register(idx), operand) => {
                    self.registers[*idx] += match operand {
                        Operand::Literal(l) => *l,
                        Operand::Register(Register(r)) => self.registers[*r],
                    }
                }
                Instruction::Mul(Register(idx), operand) => {
                    self.registers[*idx] *= match operand {
                        Operand::Literal(l) => *l,
                        Operand::Register(Register(r)) => self.registers[*r],
                    }
                }
                Instruction::Div(Register(idx), operand) => {
                    let value = match operand {
                        Operand::Literal(l) => *l,
                        Operand::Register(Register(r)) => self.registers[*r],
                    };

                    if value == 0 {
                        // Error! This is not a legal execution stream.
                        return None;
                    }

                    self.registers[*idx] /= value;
                }
                Instruction::Mod(Register(idx), operand) => {
                    let left = self.registers[*idx];
                    let right = match operand {
                        Operand::Literal(l) => *l,
                        Operand::Register(Register(r)) => self.registers[*r],
                    };

                    if left < 0 || right <= 0 {
                        // Error! This is not a legal execution stream.
                        return None;
                    }

                    self.registers[*idx] = left % right;
                }
                Instruction::Equal(Register(idx), operand) => {
                    let left = self.registers[*idx];
                    let right = match operand {
                        Operand::Literal(l) => *l,
                        Operand::Register(Register(r)) => self.registers[*r],
                    };
                    self.registers[*idx] = if left == right { 1 } else { 0 };
                }
                Instruction::Input(_) => unreachable!(),
            }
        }

        Some(self)
    }

    fn advance_through_inputs(self) -> impl Iterator<Item = Simulation<'a>> {
        let input_register = match self.remaining.first().unwrap() {
            Instruction::Input(Register(reg)) => *reg,
            _ => unreachable!("{:?}", self.remaining.first().unwrap()),
        };
        let remaining_instrs = &self.remaining[1..];

        self.inputs.generate_next().map(move |(value, inputs)| {
            let mut next_registers = self.registers;
            next_registers[input_register] = value;

            Simulation {
                registers: next_registers,
                remaining: remaining_instrs,
                inputs,
            }
        })
    }
}

fn process<'a>(start: Simulation<'a>) -> Box<dyn Iterator<Item = Simulation<'a>> + 'a> {
    match start.is_valid() {
        None => {
            // The computation isn't done yet. Continue onward.
            Box::new(
                start
                    .advance_through_inputs()
                    .filter_map(Simulation::advance_until_input)
                    .flat_map(process),
            )
        }
        Some(false) => {
            // The computation is done but didn't produce a valid MONAD number.
            Box::new([].into_iter())
        }
        Some(true) => {
            // We found a valid MONAD number!
            Box::new([start].into_iter())
        }
    }
}

fn solve_part1(data: &[Instruction]) -> u64 {
    let simulation = Simulation::new(data).advance_until_input().unwrap();

    let best_simulation = process(simulation).next().unwrap();
    println!("{:?}", best_simulation);

    let mut result = 0u64;
    for digit in best_simulation.inputs.values {
        assert!((1..=9).contains(&digit));
        result *= 10;
        result += digit as u64;
    }
    result
}

#[allow(unused_variables)]
fn solve_part2(data: &[Instruction]) -> usize {
    todo!()
}
