use std::collections::BTreeMap;

#[allow(unused_imports)]
use itertools::Itertools;

use crate::{
    analysis::{
        values::{ValueRange, Vid},
        Analysis, AnnotatedInstr,
    },
    parser::{Instruction, Operand, Register},
};

pub fn find_extremal_input_that_matches_analysis(
    analysis: Analysis,
    input_ranges: Vec<ValueRange>,
    expected_values: BTreeMap<Vid, ValueRange>,
    get_maximal_input: bool,
) -> u64 {
    let instructions = analysis.annotated.values().cloned().collect_vec();
    let simulation = Simulation::new(&instructions, &input_ranges, &expected_values, get_maximal_input)
        .advance_until_input()
        .unwrap();

    let best_simulation = process(simulation).next().unwrap();

    best_simulation.inputs.provided_input_as_number()
}

#[derive(Debug, Clone)]
struct ProvidedInputs<'a> {
    input_ranges: &'a [ValueRange],
    values: Vec<i64>,
    get_maximal_input: bool,
}

impl<'a> ProvidedInputs<'a> {
    const MAX_COUNT: usize = 14;

    fn new(input_ranges: &'a [ValueRange], get_maximal_input: bool) -> Self {
        Self {
            input_ranges,
            get_maximal_input,
            values: Vec::with_capacity(ProvidedInputs::MAX_COUNT),
        }
    }

    fn provided_input_as_number(&self) -> u64 {
        let mut result = 0u64;
        for digit in self.values.iter() {
            assert!((1..=9).contains(digit));
            result *= 10;
            result += *digit as u64;
        }
        result
    }

    fn generate_next(mut self) -> impl Iterator<Item = (i64, ProvidedInputs<'a>)> {
        let next_range = self.input_ranges.first().unwrap();
        self.input_ranges = &self.input_ranges[1..];

        let mut range = (next_range.start()..=next_range.end()).collect_vec();
        if self.get_maximal_input {
            range.reverse();
        }

        range.into_iter().map(move |value| {
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
    remaining: &'a [AnnotatedInstr],
    expected_values: &'a BTreeMap<Vid, ValueRange>,
    inputs: ProvidedInputs<'a>,
}

impl<'a> Simulation<'a> {
    fn new(
        instructions: &'a [AnnotatedInstr],
        input_ranges: &'a [ValueRange],
        expected_values: &'a BTreeMap<Vid, ValueRange>,
        get_maximal_input: bool,
    ) -> Self {
        Self {
            registers: [0; 4],
            remaining: instructions,
            expected_values,
            inputs: ProvidedInputs::new(input_ranges, get_maximal_input),
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
        while let Some(ann_instr) = self.remaining.first() {
            if matches!(ann_instr.instr, Instruction::Input(_)) {
                break;
            }

            let destination = ann_instr.instr.destination().0;

            self.remaining = &self.remaining[1..];
            match ann_instr.instr {
                Instruction::Add(_, operand) => {
                    self.registers[destination] += match operand {
                        Operand::Literal(l) => l,
                        Operand::Register(Register(r)) => self.registers[r],
                    }
                }
                Instruction::Mul(_, operand) => {
                    self.registers[destination] *= match operand {
                        Operand::Literal(l) => l,
                        Operand::Register(Register(r)) => self.registers[r],
                    }
                }
                Instruction::Div(_, operand) => {
                    let value = match operand {
                        Operand::Literal(l) => l,
                        Operand::Register(Register(r)) => self.registers[r],
                    };

                    if value == 0 {
                        // Error! This is not a legal execution stream.
                        return None;
                    }

                    self.registers[destination] /= value;
                }
                Instruction::Mod(_, operand) => {
                    let left = self.registers[destination];
                    let right = match operand {
                        Operand::Literal(l) => l,
                        Operand::Register(Register(r)) => self.registers[r],
                    };

                    if left < 0 || right <= 0 {
                        // Error! This is not a legal execution stream.
                        return None;
                    }

                    self.registers[destination] = left % right;
                }
                Instruction::Equal(_, operand) => {
                    let left = self.registers[destination];
                    let right = match operand {
                        Operand::Literal(l) => l,
                        Operand::Register(Register(r)) => self.registers[r],
                    };
                    self.registers[destination] = if left == right { 1 } else { 0 };
                }
                Instruction::Input(_) => unreachable!(),
            }

            let expected_register_range = &self.expected_values[&ann_instr.result];
            if !expected_register_range.contains(self.registers[destination]) {
                // The computation has diverged from the expected trajectory, and will not produce
                // the desired Z=0 result. Stop the search down this path.
                return None;
            }
        }

        Some(self)
    }

    fn advance_through_inputs(self) -> impl Iterator<Item = Simulation<'a>> {
        let input_register = match self.remaining.first().unwrap().instr {
            Instruction::Input(Register(reg)) => reg,
            _ => unreachable!("{:?}", self.remaining.first().unwrap()),
        };
        let remaining_instrs = &self.remaining[1..];

        self.inputs.generate_next().map(move |(value, inputs)| {
            let mut next_registers = self.registers;
            next_registers[input_register] = value;

            Simulation {
                registers: next_registers,
                remaining: remaining_instrs,
                expected_values: self.expected_values,
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
