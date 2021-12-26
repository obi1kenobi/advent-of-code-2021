#![feature(map_try_insert)]

use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    env, fs,
};

#[allow(unused_imports)]
use itertools::Itertools;
use itertools::{max, min};
use parser::{parse_program, Instruction, Operand, Register};

#[allow(unused_imports)]
use crate::parser::InstructionStream;

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

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Simulation<'a> {
    registers: [i64; 4],
    remaining: &'a [Instruction],
    inputs: ProvidedInputs,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegisterState {
    Exact(i64),
    Input(usize),
    Unknown(usize),
    Undefined, // Result of undefined behavior.
}

#[allow(clippy::type_complexity)]
fn perform_constant_propagation_and_value_numbering(
    data: &[Instruction],
) -> (
    Vec<[RegisterState; 4]>,
    BTreeMap<usize, ([RegisterState; 4], &Instruction)>,
) {
    let mut state = [RegisterState::Exact(0); 4];
    let mut inputs = 0usize;

    // Perform value-numbering and remember how each unknown value came to be:
    // (registers prior to instruction, instruction)
    let mut value_definitions: BTreeMap<usize, ([RegisterState; 4], &Instruction)> =
        Default::default();

    let state_ref = &mut state;
    let value_definitions_ref = &mut value_definitions;
    let inputs_ref = &mut inputs;
    let register_states = data
        .iter()
        .map(move |instr| {
            let mut next_state = *state_ref;

            let destination = instr.destination().0;
            let source_value = state_ref[destination];
            let operand_value = match instr.operand() {
                Some(Operand::Literal(l)) => RegisterState::Exact(l),
                Some(Operand::Register(r)) => state_ref[r.0],
                _ => RegisterState::Undefined,
            };

            match *instr {
                Instruction::Input(Register(reg)) => {
                    let input_number = *inputs_ref;
                    *inputs_ref += 1;
                    next_state[reg] = RegisterState::Input(input_number);
                }
                Instruction::Add(_, _) => {
                    next_state[destination] = match (source_value, operand_value) {
                        (RegisterState::Undefined, _) | (_, RegisterState::Undefined) => {
                            unreachable!()
                        }
                        (RegisterState::Exact(a), RegisterState::Exact(b)) => {
                            RegisterState::Exact(a + b)
                        }
                        (RegisterState::Exact(0), _) => operand_value,
                        (_, RegisterState::Exact(0)) => source_value,
                        (RegisterState::Unknown(_), _)
                        | (_, RegisterState::Unknown(_))
                        | (RegisterState::Input(_), RegisterState::Input(_))
                        | (RegisterState::Exact(_), RegisterState::Input(_))
                        | (RegisterState::Input(_), RegisterState::Exact(_)) => {
                            let value_number = value_definitions_ref.len();
                            value_definitions_ref
                                .try_insert(value_number, (*state_ref, instr))
                                .unwrap();
                            RegisterState::Unknown(value_number)
                        }
                    };
                }
                Instruction::Mul(_, _) => {
                    next_state[destination] = match (source_value, operand_value) {
                        (RegisterState::Undefined, _) | (_, RegisterState::Undefined) => {
                            unreachable!()
                        }
                        (RegisterState::Exact(a), RegisterState::Exact(b)) => {
                            RegisterState::Exact(a * b)
                        }
                        (RegisterState::Exact(1), _) => operand_value,
                        (_, RegisterState::Exact(1)) => source_value,
                        (RegisterState::Exact(0), _) | (_, RegisterState::Exact(0)) => {
                            RegisterState::Exact(0)
                        }
                        (RegisterState::Unknown(_), _)
                        | (_, RegisterState::Unknown(_))
                        | (RegisterState::Input(_), RegisterState::Input(_))
                        | (RegisterState::Exact(_), RegisterState::Input(_))
                        | (RegisterState::Input(_), RegisterState::Exact(_)) => {
                            let value_number = value_definitions_ref.len();
                            value_definitions_ref
                                .try_insert(value_number, (*state_ref, instr))
                                .unwrap();
                            RegisterState::Unknown(value_number)
                        }
                    };
                }
                Instruction::Div(_, _) => {
                    next_state[destination] = match (source_value, operand_value) {
                        (RegisterState::Undefined, _) | (_, RegisterState::Undefined) => {
                            unreachable!()
                        }
                        (_, RegisterState::Exact(0)) => {
                            panic!("dividing by zero: {} {:?}", *instr, state_ref)
                        }
                        (RegisterState::Exact(a), RegisterState::Exact(b)) => {
                            RegisterState::Exact(a / b)
                        }
                        (RegisterState::Exact(0), _) | (_, RegisterState::Exact(1)) => source_value,
                        (RegisterState::Unknown(a), RegisterState::Unknown(b)) if a == b => {
                            // Both values are the same number, so they must be equal.
                            RegisterState::Exact(1)
                        }
                        (RegisterState::Unknown(_), _)
                        | (_, RegisterState::Unknown(_))
                        | (RegisterState::Input(_), RegisterState::Input(_))
                        | (RegisterState::Exact(_), RegisterState::Input(_))
                        | (RegisterState::Input(_), RegisterState::Exact(_)) => {
                            let value_number = value_definitions_ref.len();
                            value_definitions_ref
                                .try_insert(value_number, (*state_ref, instr))
                                .unwrap();
                            RegisterState::Unknown(value_number)
                        }
                    };
                }
                Instruction::Mod(_, _) => {
                    next_state[destination] = match (source_value, operand_value) {
                        (RegisterState::Undefined, _) | (_, RegisterState::Undefined) => {
                            unreachable!()
                        }
                        (RegisterState::Exact(a), RegisterState::Exact(b)) => {
                            if a < 0 || b <= 0 {
                                panic!(
                                    "illegal mod operands: {} {} {} {:?}",
                                    a, b, *instr, state_ref
                                );
                            } else {
                                RegisterState::Exact(a % b)
                            }
                        }
                        (_, RegisterState::Exact(1)) => source_value,
                        (RegisterState::Unknown(a), RegisterState::Unknown(b)) if a == b => {
                            // Both values are the same number, so they must be equal.
                            // The modulus operation has the property that for all x, x % x == 0.
                            RegisterState::Exact(0)
                        }
                        (RegisterState::Unknown(_), _)
                        | (_, RegisterState::Unknown(_))
                        | (RegisterState::Input(_), RegisterState::Input(_))
                        | (RegisterState::Exact(_), RegisterState::Input(_))
                        | (RegisterState::Input(_), RegisterState::Exact(_)) => {
                            let value_number = value_definitions_ref.len();
                            value_definitions_ref
                                .try_insert(value_number, (*state_ref, instr))
                                .unwrap();
                            RegisterState::Unknown(value_number)
                        }
                    };
                }
                Instruction::Equal(_, _) => {
                    next_state[destination] = match (source_value, operand_value) {
                        (RegisterState::Undefined, _) | (_, RegisterState::Undefined) => {
                            unreachable!()
                        }
                        (RegisterState::Exact(a), RegisterState::Exact(b)) => {
                            if a == b {
                                RegisterState::Exact(1)
                            } else {
                                RegisterState::Exact(0)
                            }
                        }
                        (RegisterState::Input(a), RegisterState::Input(b)) if a == b => {
                            // These two registers' values are loaded from the same input step.
                            // It doesn't matter what their exact values are, they are always equal.
                            RegisterState::Exact(1)
                        }
                        (RegisterState::Input(_), RegisterState::Exact(x))
                        | (RegisterState::Exact(x), RegisterState::Input(_))
                            if !(1..=9).contains(&x) =>
                        {
                            // One of the values is exactly equal to an input, whereas the other
                            // is outside of the range 1..=9 where all input values are guaranteed
                            // to fall. It's impossible for this equality to hold.
                            RegisterState::Exact(0)
                        }
                        (RegisterState::Unknown(a), RegisterState::Unknown(b)) if a == b => {
                            // Both values are the same number, so they must be equal.
                            RegisterState::Exact(1)
                        }
                        (RegisterState::Unknown(_), _)
                        | (_, RegisterState::Unknown(_))
                        | (RegisterState::Input(_), RegisterState::Input(_))
                        | (RegisterState::Exact(_), RegisterState::Input(_))
                        | (RegisterState::Input(_), RegisterState::Exact(_)) => {
                            let value_number = value_definitions_ref.len();
                            value_definitions_ref
                                .try_insert(value_number, (*state_ref, instr))
                                .unwrap();
                            RegisterState::Unknown(value_number)
                        }
                    }
                }
            }

            *state_ref = next_state;

            next_state
        })
        .collect_vec();

    (register_states, value_definitions)
}

fn find_input_and_value_dependencies_recursively(
    value_definitions: &BTreeMap<usize, ([RegisterState; 4], &Instruction)>,
    target_unknown_value: usize,
    input_dependencies: &mut BTreeSet<usize>,
    value_dependencies: &mut BTreeSet<usize>,
) {
    let (registers_before_instr, instr) = value_definitions[&target_unknown_value];

    match instr {
        Instruction::Input(_) => {
            // RegisterState::Unknown values cannot directly originate from an Input instruction.
            unreachable!();
        }
        Instruction::Add(_, _)
        | Instruction::Mul(_, _)
        | Instruction::Div(_, _)
        | Instruction::Mod(_, _)
        | Instruction::Equal(_, _) => {
            let operand_state = match instr.operand().unwrap() {
                Operand::Literal(_) => None,
                Operand::Register(Register(reg)) => Some(registers_before_instr[reg]),
            };
            let states = [
                Some(registers_before_instr[instr.destination().0]),
                operand_state,
            ];

            for state in states.into_iter().flatten() {
                match state {
                    RegisterState::Exact(_) => {}
                    RegisterState::Input(n) => {
                        input_dependencies.insert(n);
                    }
                    RegisterState::Unknown(n) => {
                        if value_dependencies.insert(n) {
                            // This is a newly-discovered dependency,
                            // recurse into it to see if there are more
                            // dependencies to be discovered.
                            find_input_and_value_dependencies_recursively(
                                value_definitions,
                                n,
                                input_dependencies,
                                value_dependencies,
                            );
                        }
                    }
                    RegisterState::Undefined => unreachable!(),
                }
            }
        }
    }
}

fn find_input_and_value_dependencies(
    value_definitions: &BTreeMap<usize, ([RegisterState; 4], &Instruction)>,
    target_unknown_value: usize,
) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let mut input_dependencies = BTreeSet::new();
    let mut value_dependencies = BTreeSet::new();

    find_input_and_value_dependencies_recursively(
        value_definitions,
        target_unknown_value,
        &mut input_dependencies,
        &mut value_dependencies,
    );

    (input_dependencies, value_dependencies)
}

const INPUT_VALUE_RANGE: (i64, i64) = (1, 9);
const MAX_VALUE_RANGE: (i64, i64) = (i64::MIN, i64::MAX);

fn get_value_range(
    prior_value_ranges: &BTreeMap<usize, (i64, i64)>,
    register_state: RegisterState,
) -> (i64, i64) {
    match register_state {
        RegisterState::Exact(n) => (n, n),
        RegisterState::Input(_) => INPUT_VALUE_RANGE,
        RegisterState::Unknown(n) => prior_value_ranges[&n],
        RegisterState::Undefined => unreachable!(),
    }
}

fn get_binary_instr_ranges(instr: &Instruction) -> ((i64, i64), (i64, i64)) {
    match instr {
        Instruction::Input(_) => unreachable!(),
        Instruction::Add(_, _)
        | Instruction::Mul(_, _)
        | Instruction::Div(_, _)  // We can't exclude zero since range "holes" are non-representable
        | Instruction::Equal(_, _) => (MAX_VALUE_RANGE, MAX_VALUE_RANGE),
        Instruction::Mod(_, _) => {
            // The source register's value must be >= 0, or else it's UB.
            // If the operand is a register, we know it must be > 0 or else it's UB.
            ((0, MAX_VALUE_RANGE.1), (1, MAX_VALUE_RANGE.1))
        }
    }
}

fn intersect_value_ranges(a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    let lower = std::cmp::max(a.0, b.0);
    let higher = std::cmp::min(a.1, b.1);

    if higher < lower {
        None
    } else {
        Some((lower, higher))
    }
}

fn div_range_analysis(source: (i64, i64), divisor: (i64, i64)) -> (i64, i64) {
    let (source_low, source_high) = source;
    let (mut divisor_low, mut divisor_high) = divisor;

    // If the divisor range is exactly 0, something has gone horribly wrong.
    assert!(divisor != (0, 0));

    // Otherwise, if either divisor range endpoint is on 0, we can nudge the range to exclude 0
    // since we know 0 isn't a legal divisor value.
    if divisor_low == 0 {
        divisor_low += 1;
    }
    if divisor_high == 0 {
        divisor_high -= 1;
    }

    let (result_low, result_high) = if divisor_low < 0 && divisor_high > 0 {
        // straddling 0, both -1 and 1 are possible
        let extreme_values = [-source_low, source_low, -source_high, source_high];
        (min(extreme_values).unwrap(), max(extreme_values).unwrap())
    } else if divisor_high < 0 {
        // negative divisor range
        if source_low <= 0 && source_high >= 0 {
            // any divided by negative
            (
                source_high.saturating_div(divisor_high),
                source_low.saturating_div(divisor_high),
            )
        } else if source_high < 0 {
            // both operands are negative, so the result is positive
            (
                source_high.saturating_div(divisor_low),
                source_low.saturating_div(divisor_high),
            )
        } else if source_low > 0 {
            // positive divided by negative, the result is negative
            (
                source_high.saturating_div(divisor_high),
                source_low.saturating_div(divisor_low),
            )
        } else {
            unreachable!()
        }
    } else if divisor_low > 0 {
        // positive divisor range
        if source_low <= 0 && source_high >= 0 {
            // any divided by positive
            (
                source_low.saturating_div(divisor_low),
                source_high.saturating_div(divisor_low),
            )
        } else if source_high < 0 {
            // negative divided by positive, the result is negative
            (
                source_low.saturating_div(divisor_low),
                source_high.saturating_div(divisor_high),
            )
        } else if source_low > 0 {
            // both operands are positive, the result is positive
            (
                source_low.saturating_div(divisor_high),
                source_high.saturating_div(divisor_low),
            )
        } else {
            unreachable!()
        }
    } else {
        unreachable!()
    };

    assert!(
        result_low <= result_high,
        "{:?} {:?} -> ({}, {})",
        source,
        divisor,
        result_low,
        result_high
    );
    (result_low, result_high)
}

fn mul_range_analysis(source: (i64, i64), multiplier: (i64, i64)) -> (i64, i64) {
    let (result_low, result_high) = {
        let (source_low, source_high) = source;
        let (multiplier_low, multiplier_high) = multiplier;

        // careful with negative numbers!
        let extreme_values = [
            source_low.saturating_mul(multiplier_low),
            source_low.saturating_mul(multiplier_high),
            source_high.saturating_mul(multiplier_low),
            source_high.saturating_mul(multiplier_high),
        ];
        (min(extreme_values).unwrap(), max(extreme_values).unwrap())
    };

    assert!(
        result_low <= result_high,
        "{:?} {:?} -> ({}, {})",
        source,
        multiplier,
        result_low,
        result_high
    );
    (result_low, result_high)
}

fn add_range_analysis(source: (i64, i64), operand: (i64, i64)) -> (i64, i64) {
    let (result_low, result_high) = {
        let (source_low, source_high) = source;
        let (operand_low, operand_high) = operand;

        (
            source_low.saturating_add(operand_low),
            source_high.saturating_add(operand_high),
        )
    };

    assert!(
        result_low <= result_high,
        "{:?} {:?} -> ({}, {})",
        source,
        operand,
        result_low,
        result_high
    );
    (result_low, result_high)
}

fn sub_range_analysis(source: (i64, i64), operand: (i64, i64)) -> (i64, i64) {
    let (result_low, result_high) = {
        let (source_low, source_high) = source;
        let (operand_low, operand_high) = operand;

        (
            source_low.saturating_sub(operand_high),
            source_high.saturating_sub(operand_low),
        )
    };

    assert!(
        result_low <= result_high,
        "{:?} {:?} -> ({}, {})",
        source,
        operand,
        result_low,
        result_high
    );
    (result_low, result_high)
}

fn equal_range_analysis(source: (i64, i64), operand: (i64, i64)) -> (i64, i64) {
    let (result_low, result_high) = {
        let (source_low, source_high) = source;
        let (operand_low, operand_high) = operand;

        let operand_exact = operand_low == operand_high;
        let source_exact = source_low == source_high;
        let values_match = source_low == operand_low;

        if operand_exact && source_exact {
            // both values are known exactly
            if values_match {
                (1, 1)
            } else {
                (0, 0)
            }
        } else if operand_low > source_high || operand_high < source_low {
            // no overlap in the ranges, no match
            (0, 0)
        } else {
            // ranges overlap, could be equal or not
            (0, 1)
        }
    };

    assert!(
        result_low <= result_high,
        "{:?} {:?} -> ({}, {})",
        source,
        operand,
        result_low,
        result_high
    );
    (result_low, result_high)
}

fn mod_range_analysis(source: (i64, i64), operand: (i64, i64)) -> (i64, i64) {
    let (result_low, result_high) = {
        let (source_low, source_high) = source;
        let (operand_low, operand_high) = operand;

        let source_low = std::cmp::max(0, source_low);
        let operand_low = std::cmp::max(1, operand_low);

        // Ensure valid values are available in the range.
        assert!(source_high >= source_low);
        assert!(operand_high >= operand_low);

        if source_high < operand_low {
            // The source range is under the lowest possible mod operand value.
            // The mod is a no-op for all source and operand values.
            // TODO: If the target was previously an Unknown() value,
            //       replace it with the source value!
            (source_low, source_high)
        } else if source_low == 0 && source_high < operand_high {
            // The values X at the top end of the mod range are unreachable:
            // - to get a value X, the operand must be X + 1 or greater,
            // - but source_high < X.
            (0, source_high)
        } else {
            // The full mod range is represented.
            (0, operand_high - 1)
        }
    };

    assert!(
        result_low <= result_high,
        "{:?} {:?} -> ({}, {})",
        source,
        operand,
        result_low,
        result_high
    );
    (result_low, result_high)
}

fn inv_mul_range_analysis(result: (i64, i64), multiplier: (i64, i64)) -> (i64, i64) {
    // source * multiplier = result, solve for source given ranges for result and multiplier
    // This is similar to div_range_analysis but accounts for the truncation in division.
    let (result_low, result_high) = result;
    let (multiplier_low, multiplier_high) = multiplier;

    let result_interval = result_low..=result_high;

    // Ensure valid values are available in the range.
    assert!(result_high >= result_low);
    assert!(multiplier_high >= multiplier_low);

    // If the multiplier range is exactly 0 but the result can't be zero,
    // something has gone horribly wrong.
    if multiplier == (0, 0) {
        // The result interval must be (0, 0), and the source can be any number.
        assert!(result_interval.contains(&0));
        return MAX_VALUE_RANGE;
    }

    let (mut source_low, mut source_high) = div_range_analysis(result, multiplier);

    // Try shrinking the range by 1 from the bottom and top, to account for truncation.
    if !result_interval.contains(&(source_low * multiplier_low))
        && !result_interval.contains(&(source_low * multiplier_high))
    {
        source_low += 1;
    }
    if !result_interval.contains(&(source_high * multiplier_low))
        && !result_interval.contains(&(source_high * multiplier_high))
    {
        source_high -= 1;
    }

    assert!(
        result_interval.contains(&(source_low * multiplier_low))
            || result_interval.contains(&(source_low * multiplier_high))
    );
    assert!(
        result_interval.contains(&(source_high * multiplier_low))
            || result_interval.contains(&(source_high * multiplier_high))
    );

    (source_low, source_high)
}

fn inv_div_range_analysis(result: (i64, i64), mut divisor: (i64, i64)) -> (i64, i64) {
    // source / divisor = result, solve for source given ranges for result and divisor
    // This is similar to mul_range_analysis but accounts for the truncation in division.
    let (result_low, result_high) = result;
    let (mut divisor_low, mut divisor_high) = divisor;

    // Ensure valid values are available in the range.
    assert!(result_high >= result_low);
    assert!(divisor_high >= divisor_low);

    // If the divisor range is exactly 0, something has gone horribly wrong.
    assert!(divisor != (0, 0));

    // Otherwise, if either divisor range endpoint is on 0, we can nudge the range to exclude 0
    // since we know 0 isn't a legal divisor value.
    if divisor_low == 0 {
        divisor_low += 1;
        divisor = (divisor_low, divisor_high);
    }
    if divisor_high == 0 {
        divisor_high -= 1;
        divisor = (divisor_low, divisor_high);
    }

    let (source_low, source_high) = {
        let mut source_high = i64::MIN;
        let mut source_low = i64::MAX;

        let (non_positive_result_extremes, non_negative_result_extremes) = {
            if result_high < 0 {
                // The entire range is negative. There are no non-negative values.
                (Some((result_low, result_high)), None)
            } else if result_low > 0 {
                // The entire range is positive. There are no non-positive values.
                (None, Some((result_low, result_high)))
            } else {
                // The range has both non-positive and non-negative values.
                (Some((result_low, 0)), Some((0, result_high)))
            }
        };
        let (negative_divisor_extremes, positive_divisor_extremes) = {
            if divisor_high < 0 {
                // The entire range is negative. There are no negative values.
                (Some((divisor_low, divisor_high)), None)
            } else if divisor_low > 0 {
                // The entire range is positive. There are no positive values.
                (None, Some((divisor_low, divisor_high)))
            } else {
                // The range has both positive and negative values.
                // We don't consider 0 values since this is the divisor.
                (Some((divisor_low, 1)), Some((1, divisor_high)))
            }
        };

        if let (Some((min_pos_result, max_pos_result)), Some((min_pos_div, max_pos_div))) = (non_negative_result_extremes, positive_divisor_extremes) {
            // Both operands are non-negative, so the result is always non-negative.
            // Truncation toward zero means that only the high value needs a truncation adjustment.
            source_high = std::cmp::max(
                source_high,
                max_pos_result.saturating_mul(max_pos_div).saturating_add(max_pos_div - 1),
            );
            source_low = std::cmp::min(
                source_low,
                min_pos_result.saturating_mul(min_pos_div)
            );
        }
        if let (Some((min_neg_result, max_neg_result)), Some((min_neg_div, max_neg_div))) = (non_positive_result_extremes, negative_divisor_extremes) {
            // Both operands are non-positive, so the result is always non-negative.
            // Truncation toward zero means that only the high value needs a truncation adjustment.
            source_high = std::cmp::max(
                source_high,
                // We do a saturating_sub() of min_neg_div + 1 rather than
                // a saturating_add() of -min_neg_div followed by -1, since
                // -min_neg_div could overflow if min_neg_div is i64::MIN.
                min_neg_result.saturating_mul(min_neg_div).saturating_sub(min_neg_div + 1),
            );
            source_low = std::cmp::min(
                source_low,
                max_neg_result.saturating_mul(max_neg_div)
            );
        }
        if let (Some((min_pos_result, max_pos_result)), Some((min_neg_div, max_neg_div))) = (non_negative_result_extremes, negative_divisor_extremes) {
            // One operand is non-negative and the other is negative,
            // so the result is always non-positive.
            // Truncation toward zero means that only the low value needs a truncation adjustment.
            source_high = std::cmp::max(
                source_high,
                min_pos_result.saturating_mul(max_neg_div),
            );
            source_low = std::cmp::min(
                source_low,
                max_pos_result.saturating_mul(min_neg_div).saturating_add(min_neg_div + 1),
            );
        }
        if let (Some((min_neg_result, max_neg_result)), Some((min_pos_div, max_pos_div))) = (non_positive_result_extremes, positive_divisor_extremes) {
            // One operand is non-positive and the other is positive,
            // so the result is always non-positive.
            // Truncation toward zero means that only the low value needs a truncation adjustment.
            source_high = std::cmp::max(
                source_high,
                max_neg_result.saturating_mul(min_pos_div),
            );
            source_low = std::cmp::min(
                source_low,
                min_neg_result.saturating_mul(max_pos_div).saturating_sub(max_pos_div - 1),
            );
        }

        (source_low, source_high)
    };

    assert!(
        source_low <= source_high,
        "{:?} {:?} -> ({}, {})",
        result,
        divisor,
        source_low,
        source_high
    );
    (source_low, source_high)
}

fn value_range_analysis(
    value_definitions: &BTreeMap<usize, ([RegisterState; 4], &Instruction)>,
) -> BTreeMap<usize, (i64, i64)> {
    // returned ranges are inclusive on both endpoints
    let mut result: BTreeMap<usize, (i64, i64)> = Default::default();

    for (value, (registers, instr)) in value_definitions.iter() {
        let range = if let Instruction::Input(_) = instr {
            INPUT_VALUE_RANGE
        } else {
            let destination = instr.destination().0;
            let source_value = registers[destination];
            let operand_value = match instr.operand().unwrap() {
                Operand::Literal(l) => RegisterState::Exact(l),
                Operand::Register(r) => registers[r.0],
            };

            let (source_instr_range, operand_instr_range) = get_binary_instr_ranges(*instr);
            let (source_low, source_high) =
                intersect_value_ranges(source_instr_range, get_value_range(&result, source_value))
                    .unwrap();
            let (operand_low, operand_high) = intersect_value_ranges(
                operand_instr_range,
                get_value_range(&result, operand_value),
            )
            .unwrap();
            assert!(source_low <= source_high);
            assert!(operand_low <= operand_high);

            match instr {
                Instruction::Input(_) => unreachable!(), // already handled above
                Instruction::Add(_, _) => {
                    add_range_analysis((source_low, source_high), (operand_low, operand_high))
                }
                Instruction::Mul(_, _) => {
                    mul_range_analysis((source_low, source_high), (operand_low, operand_high))
                }
                Instruction::Div(_, _) => {
                    div_range_analysis((source_low, source_high), (operand_low, operand_high))
                }
                Instruction::Mod(_, _) => {
                    mod_range_analysis((source_low, source_high), (operand_low, operand_high))
                }
                Instruction::Equal(_, _) => {
                    equal_range_analysis((source_low, source_high), (operand_low, operand_high))
                }
            }
        };

        let (range_low, range_high) = range;
        assert!(range_low <= range_high);
        result.try_insert(*value, range).unwrap();
    }

    result
}

fn update_range_data(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value: RegisterState,
    new_range: (i64, i64),
) -> (i64, i64) {
    let entry_modifier = |val: &mut (i64, i64)| {
        let final_range = intersect_value_ranges(*val, new_range).unwrap();
        *val = final_range;
    };
    match value {
        RegisterState::Input(n) => match input_ranges.entry(n).and_modify(entry_modifier) {
            Entry::Occupied(occ) => *occ.get(),
            Entry::Vacant(_) => unreachable!(),
        },
        RegisterState::Unknown(n) => match value_ranges.entry(n).and_modify(entry_modifier) {
            Entry::Occupied(occ) => *occ.get(),
            Entry::Vacant(_) => unreachable!(),
        },
        RegisterState::Exact(n) => {
            // Use the intersection function to ensure the ranges overlap. If they don't overlap,
            // we've detected UB (or there's a bug in the range analysis).
            intersect_value_ranges(new_range, (n, n)).unwrap()
        }
        RegisterState::Undefined => unreachable!(),
    }
}

fn update_range_data_if_register(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    operand: Operand,
    registers: &[RegisterState; 4],
    new_range: (i64, i64),
) -> (i64, i64) {
    match operand {
        Operand::Register(Register(r)) => {
            update_range_data(input_ranges, value_ranges, registers[r], new_range)
        }
        Operand::Literal(l) => {
            // Use the intersection function to ensure the ranges overlap. If they don't overlap,
            // we've detected UB (or there's a bug in the range analysis).
            intersect_value_ranges(new_range, (l, l)).unwrap()
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn perform_input_range_analysis(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    result_range: (i64, i64),
    operand_value_range: (i64, i64),
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
    source_range_func: fn((i64, i64), (i64, i64)) -> (i64, i64),
    operand_range_func: fn((i64, i64), (i64, i64)) -> (i64, i64),
    result_range_func: fn((i64, i64), (i64, i64)) -> (i64, i64),
) {
    // It's maybe possible that updating range data for one input may allow
    // the other input's range to also be narrowed. Allow each input's range
    // to be updated after the other input's range, to ensure convergence in any case.
    let source_value_range = update_range_data(
        input_ranges,
        value_ranges,
        source_value,
        source_range_func(result_range, operand_value_range),
    );
    let result_range = intersect_value_ranges(
        result_range,
        result_range_func(source_value_range, operand_value_range),
    )
    .unwrap();
    let operand_value_range = update_range_data_if_register(
        input_ranges,
        value_ranges,
        operand,
        registers,
        operand_range_func(result_range, source_value_range),
    );
    let result_range = intersect_value_ranges(
        result_range,
        result_range_func(source_value_range, operand_value_range),
    )
    .unwrap();
    update_range_data(
        input_ranges,
        value_ranges,
        source_value,
        source_range_func(result_range, operand_value_range),
    );
}

#[allow(clippy::too_many_arguments)]
fn add_input_range_analysis(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    result_range: (i64, i64),
    _source_value_range: (i64, i64),
    operand_value_range: (i64, i64),
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
) {
    perform_input_range_analysis(
        input_ranges,
        value_ranges,
        result_range,
        operand_value_range,
        source_value,
        operand,
        registers,
        sub_range_analysis,
        sub_range_analysis,
        add_range_analysis,
    )
}

fn mul_input_range_analysis_for_zero_result(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    source_value_range: (i64, i64),
    operand_value_range: (i64, i64),
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
) {
    // This is a special case, since we don't want
    // to divide by zero in the range analysis.
    //
    // Approach:
    // - If exactly one of the two mul input ranges contains 0,
    //   that range must be exactly (0, 0).
    // - If both mul input ranges contain 0, then we have no information.
    //   We know that at least one of the inputs is always 0,
    //   but we can't know which one -- and it's possible that both inputs
    //   are zero sometimes. We simply have no way to tell.
    // - If neither input range contains 0, we either have found UB
    //   or a bug in the analysis.
    let source_range = source_value_range.0..=source_value_range.1;
    let operand_range = operand_value_range.0..=operand_value_range.1;
    match (source_range.contains(&0), operand_range.contains(&0)) {
        (true, false) => {
            // The source range must be (0, 0).
            update_range_data(input_ranges, value_ranges, source_value, (0, 0));
        }
        (false, true) => {
            // The operand range must be (0, 0).
            update_range_data_if_register(input_ranges, value_ranges, operand, registers, (0, 0));
        }
        (true, true) => {} // No information, see comment above.
        (false, false) => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn mul_input_range_analysis(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    mut result_range: (i64, i64),
    source_value_range: (i64, i64),
    operand_value_range: (i64, i64),
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
) {
    // Can we prove that the result is always zero?
    if (result_range.0..=result_range.1).contains(&0) && result_range != (0, 0) {
        let expected_source_range = inv_mul_range_analysis(result_range, operand_value_range);
        let expected_operand_range = inv_mul_range_analysis(result_range, source_value_range);

        // If either expected input range has no overlap with its corresponding actual input range,
        // then the only possible solution is a zero result.
        let (source_value_range, operand_value_range) = get_register_and_operand_ranges(
            input_ranges,
            value_ranges,
            source_value,
            operand,
            registers,
        );

        if intersect_value_ranges(source_value_range, expected_source_range).is_none()
            || intersect_value_ranges(operand_value_range, expected_operand_range).is_none()
        {
            // The only possible solution is for the multiplication to be a "multiply by zero."
            result_range = (0, 0);
        }
    }

    if result_range == (0, 0) {
        mul_input_range_analysis_for_zero_result(
            input_ranges,
            value_ranges,
            source_value_range,
            operand_value_range,
            source_value,
            operand,
            registers,
        );
    } else {
        perform_input_range_analysis(
            input_ranges,
            value_ranges,
            result_range,
            operand_value_range,
            source_value,
            operand,
            registers,
            |result, operand| {
                if operand == (0, 0) {
                    MAX_VALUE_RANGE
                } else {
                    inv_mul_range_analysis(result, operand)
                }
            },
            |result, source| {
                if source == (0, 0) {
                    MAX_VALUE_RANGE
                } else {
                    inv_mul_range_analysis(result, source)
                }
            },
            mul_range_analysis,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn div_input_range_analysis(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    result_range: (i64, i64),
    _source_value_range: (i64, i64),
    operand_value_range: (i64, i64),
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
) {
    perform_input_range_analysis(
        input_ranges,
        value_ranges,
        result_range,
        operand_value_range,
        source_value,
        operand,
        registers,
        inv_div_range_analysis,
        |result, source| {
            if result == (0, 0) {
                MAX_VALUE_RANGE
            } else {
                div_range_analysis(source, result)
            }
        },
        div_range_analysis,
    );
}

#[allow(clippy::too_many_arguments)]
fn equal_input_range_analysis(
    input_ranges: &mut BTreeMap<usize, (i64, i64)>,
    value_ranges: &mut BTreeMap<usize, (i64, i64)>,
    result_range: (i64, i64),
    mut source_value_range: (i64, i64),
    mut operand_value_range: (i64, i64),
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
) {
    // There's only stuff to be learned if the instruction's output range is exact.
    if result_range.0 == result_range.1 {
        if result_range.0 == 1 {
            // The inputs are known-equal, their actual ranges are the intersection of
            // their prior ranges.
            let intersection =
                intersect_value_ranges(source_value_range, operand_value_range).unwrap();

            update_range_data(input_ranges, value_ranges, source_value, intersection);
            update_range_data_if_register(
                input_ranges,
                value_ranges,
                operand,
                registers,
                intersection,
            );
        } else if result_range.0 == 0 {
            // The inputs are known-unequal. This isn't much information -- if one
            // of the ranges is exact, and if that value is on the boundary of the other
            // input's range, then we can shrink that input's range by one.
            // Nothing else that can be expressed in our range analysis can be known.
            //
            // This loop ensures that both the check of source_value_range and
            // the check of operand_value_range run at least once before the other.
            // This is because each of them can prune the range and allow the other
            // to prune its range (or detect UB and panic).
            for _ in 0..2 {
                if source_value_range.0 == source_value_range.1 {
                    if source_value_range.0 == operand_value_range.0 {
                        operand_value_range.0 += 1;
                    }
                    if source_value_range.0 == operand_value_range.1 {
                        operand_value_range.1 -= 1;
                    }
                    assert!(operand_value_range.0 <= operand_value_range.1);
                }
                if operand_value_range.0 == operand_value_range.1 {
                    if operand_value_range.0 == source_value_range.0 {
                        source_value_range.0 += 1;
                    }
                    if operand_value_range.0 == source_value_range.1 {
                        source_value_range.1 -= 1;
                    }
                    assert!(source_value_range.0 <= source_value_range.1);
                }
            }
            update_range_data(input_ranges, value_ranges, source_value, source_value_range);
            update_range_data_if_register(
                input_ranges,
                value_ranges,
                operand,
                registers,
                operand_value_range,
            );
        }
    }
}

fn get_register_and_operand_ranges(
    input_ranges: &BTreeMap<usize, (i64, i64)>,
    value_ranges: &BTreeMap<usize, (i64, i64)>,
    source_value: RegisterState,
    operand: Operand,
    registers: &[RegisterState; 4],
) -> ((i64, i64), (i64, i64)) {
    let source_value_range = match source_value {
        RegisterState::Exact(n) => (n, n),
        RegisterState::Input(n) => input_ranges[&n],
        RegisterState::Unknown(n) => value_ranges[&n],
        RegisterState::Undefined => unreachable!(),
    };
    let operand_value_range = match operand {
        Operand::Literal(l) => (l, l),
        Operand::Register(Register(reg)) => {
            let register_value = registers[reg];
            match register_value {
                RegisterState::Exact(n) => (n, n),
                RegisterState::Input(n) => input_ranges[&n],
                RegisterState::Unknown(n) => value_ranges[&n],
                RegisterState::Undefined => unreachable!(),
            }
        }
    };

    (source_value_range, operand_value_range)
}

#[allow(clippy::type_complexity)]
fn backpropagate_range_analysis(
    value_definitions: &BTreeMap<usize, ([RegisterState; 4], &Instruction)>,
    forward_range_analysis: &BTreeMap<usize, (i64, i64)>,
    num_inputs: usize,
    final_value_range: (i64, i64),
) -> (BTreeMap<usize, (i64, i64)>, BTreeMap<usize, (i64, i64)>) {
    // return (input data, value data)
    // where each is a map from item ID -> (min, max)
    // and where (min, max) is a standard range, guaranteed to be equal to or within the input range
    let mut input_ranges: BTreeMap<usize, (i64, i64)> =
        (0..num_inputs).map(|x| (x, INPUT_VALUE_RANGE)).collect();
    let mut value_ranges: BTreeMap<usize, (i64, i64)> = forward_range_analysis.clone();

    // The final value's range is known ahead of time, update its entry.
    let (final_value, forward_range) = forward_range_analysis.iter().last().unwrap();
    let final_value_range = intersect_value_ranges(*forward_range, final_value_range).unwrap();
    value_ranges.insert(*final_value, final_value_range);

    // Begin the analysis pass, going backward from the last instruction toward the first.
    for (&value_id, &forward_range) in forward_range_analysis.iter().rev() {
        let (registers, instr) = value_definitions[&value_id];
        let range = intersect_value_ranges(forward_range, value_ranges[&value_id]).unwrap();

        let destination = instr.destination().0;
        let source_value = registers[destination];
        let operand = instr.operand().unwrap();

        let (source_value_range, operand_value_range) = get_register_and_operand_ranges(
            &input_ranges,
            &value_ranges,
            source_value,
            operand,
            &registers,
        );

        // Let's make sure there's something to learn here:
        // - If the value in the destination register before the operation was Input or Unknown,
        //   we may be able to tighten its bounds.
        // - If we have an operand register, and that register's value is Input or Unknown,
        //   we may be able to tigthen its bounds.
        let worth_exploring = {
            if matches!(
                source_value,
                RegisterState::Input(_) | RegisterState::Unknown(_)
            ) {
                true
            } else {
                match operand {
                    Operand::Literal(_) => false,
                    Operand::Register(Register(reg)) => {
                        let operand_value = registers[reg];
                        matches!(
                            operand_value,
                            RegisterState::Input(_) | RegisterState::Unknown(_)
                        )
                    }
                }
            }
        };

        // Is this true? If the node wasn't worth exploring, why is its value Unknown?
        assert!(worth_exploring);

        println!("*** range analysis ***");
        println!("{}", *instr);
        println!(
            "{:?} {:?} -> {:?}",
            source_value_range, operand_value_range, range
        );

        match *instr {
            Instruction::Input(_) => {
                // RegisterState::Unknown values cannot originate from an Input instruction.
                unreachable!()
            }
            Instruction::Add(_, _) => {
                add_input_range_analysis(
                    &mut input_ranges,
                    &mut value_ranges,
                    range,
                    source_value_range,
                    operand_value_range,
                    source_value,
                    operand,
                    &registers,
                );
            }
            Instruction::Mul(_, _) => {
                mul_input_range_analysis(
                    &mut input_ranges,
                    &mut value_ranges,
                    range,
                    source_value_range,
                    operand_value_range,
                    source_value,
                    operand,
                    &registers,
                );
            }
            Instruction::Div(_, _) => div_input_range_analysis(
                &mut input_ranges,
                &mut value_ranges,
                range,
                source_value_range,
                operand_value_range,
                source_value,
                operand,
                &registers,
            ),
            Instruction::Mod(_, _) => {
                // Our range representation doesn't allow any new information to be learned via
                // backpropagating range analysis through mod instructions. The only information
                // we could learn here is in the case where range analysis determines the mod
                // instruction is a no-op: the source range is completely within the operand range.
                // In that case, we know the result range is equal to the source range, but we
                // also know something much stronger -- the source value EQUALS the result value.
                // So rather than adjusting ranges, we will update the value numbering to reflect
                // this equality. This is strictly superior to backpropagating range analysis
                // through mod instructions.
            }
            Instruction::Equal(_, _) => equal_input_range_analysis(
                &mut input_ranges,
                &mut value_ranges,
                range,
                source_value_range,
                operand_value_range,
                source_value,
                operand,
                &registers,
            ),
        }

        println!("*** end ***\n");
    }

    (input_ranges, value_ranges)
}

fn prune_no_ops(data: &[Instruction]) -> Vec<Instruction> {
    let known_z_value = 0i64;
    let known_z_value_range = (known_z_value, known_z_value);

    let (registers_after_instr, value_definitions) =
        perform_constant_propagation_and_value_numbering(data);

    let num_inputs = registers_after_instr
        .iter()
        .flat_map(|rs| rs.iter())
        .filter_map(|s| match s {
            RegisterState::Input(i) => Some(*i),
            _ => None,
        })
        .max()
        .unwrap()
        + 1;
    let forward_range_analysis = value_range_analysis(&value_definitions);
    let (input_ranges, value_ranges) = backpropagate_range_analysis(
        &value_definitions,
        &forward_range_analysis,
        num_inputs,
        known_z_value_range,
    );

    let final_z_register = registers_after_instr.last().unwrap()[3];
    match final_z_register {
        RegisterState::Undefined => unreachable!(),
        RegisterState::Exact(n) => {
            // The inputs don't matter at all, the final z register value is a constant.
            vec![Instruction::Add(Register(3), Operand::Literal(n))]
        }
        RegisterState::Input(n) => {
            // The final z register value is equal to the n-th input (0-indexed).
            // We have to load the prior (n-1) input values to get to the n-th input,
            // so we emit n+1 "load input into z" instructions.
            vec![Instruction::Input(Register(3)); n + 1]
        }
        RegisterState::Unknown(target_unknown) => {
            // The final z register value is a numbered unknown value. Figure out what
            // inputs and unknown values influenced its value, and resume optimizing from there.
            let (input_dependencies, value_dependencies) =
                find_input_and_value_dependencies(&value_definitions, target_unknown);

            let mut result: Vec<Instruction> = vec![];
            let initial_state = [RegisterState::Exact(0); 4];
            let mut state = initial_state;
            for (registers, instr) in registers_after_instr.iter().zip(data.iter()) {
                println!("{}", instr);
                println!("{:?}", *registers);

                let register_ranges = registers.map(|s| match s {
                    RegisterState::Exact(x) => (x, x),
                    RegisterState::Input(inp) => input_ranges[&inp],
                    RegisterState::Unknown(unk) => value_ranges[&unk],
                    RegisterState::Undefined => unreachable!(),
                });
                println!("{:?}", register_ranges);

                // An operation is known to be a no-op if the state of all registers after the operation
                // is the same as before the instruction.
                // - This is true even if the instruction was an input, since all input instructions produce
                //   a RegisterState::Input value with a new number.
                // - This is true even if some of the non-destination registers are RegisterState::Unknown,
                //   since unknowns are numbered, and a given numbered RegisterState::Unknown value
                //   always represents the same value at all points in the program.
                if state == *registers {
                    println!("  --> PRUNED");
                } else {
                    result.push(instr.clone());
                    state = *registers;
                }
            }

            println!("\ninput deps:\n{:?}\n", input_dependencies);
            println!("\nvalue deps:\n{:?}\n", value_dependencies);

            println!("value {}: {:?}", 135, value_ranges[&135]);
            println!("value {}: {:?}", 134, value_ranges[&134]);
            println!("value {}: {:?}", 132, value_ranges[&132]);

            let dead_values: BTreeSet<usize> = value_definitions
                .keys()
                .copied()
                .filter(|val| *val != target_unknown && !value_dependencies.contains(val))
                .collect();
            println!("\ndead values:\n{:?}\n", dead_values);

            result
        }
    }
}

fn solve_part1(data: &[Instruction]) -> u64 {
    let _pruned = prune_no_ops(data);
    // println!("\n\n{}", InstructionStream(&pruned));

    0
}

#[allow(dead_code)]
fn solve_part1_old(data: &[Instruction]) -> u64 {
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

#[cfg(test)]
mod tests {
    use std::{
        cmp::{max, min},
        collections::BTreeMap,
    };

    use itertools::Itertools;

    use crate::{
        add_input_range_analysis, add_range_analysis, div_input_range_analysis, div_range_analysis,
        equal_range_analysis, inv_div_range_analysis, mod_range_analysis, mul_input_range_analysis,
        mul_range_analysis,
        parser::{Operand, Register},
        sub_range_analysis, RegisterState, MAX_VALUE_RANGE,
    };

    #[test]
    fn test_div_range_analysis() {
        let test_range = -10i64..=10i64;
        let pairs = test_range.clone().cartesian_product(test_range);
        let range_pairs = pairs.clone().cartesian_product(pairs);

        for (source_range, divisor_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let divisor_range = (
                min(divisor_range.0, divisor_range.1),
                max(divisor_range.0, divisor_range.1),
            );

            if divisor_range == (0, 0) {
                continue;
            }
            let expected_range = div_range_analysis(source_range, divisor_range);
            let range = expected_range.0..=expected_range.1;
            for source in source_range.0..=source_range.1 {
                for divisor in divisor_range.0..=divisor_range.1 {
                    if divisor == 0 {
                        continue;
                    }

                    let actual = source / divisor;
                    assert!(
                        range.contains(&actual),
                        "{:?} / {:?} -> {:?} for {} / {}",
                        source_range,
                        divisor_range,
                        expected_range,
                        source,
                        divisor,
                    );
                }
            }
        }
    }

    #[test]
    fn test_mul_range_analysis() {
        let test_range = -10i64..=10i64;
        let pairs = test_range.clone().cartesian_product(test_range);
        let range_pairs = pairs.clone().cartesian_product(pairs);

        for (source_range, multiplier_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let multiplier_range = (
                min(multiplier_range.0, multiplier_range.1),
                max(multiplier_range.0, multiplier_range.1),
            );

            let expected_range = mul_range_analysis(source_range, multiplier_range);
            let range = expected_range.0..=expected_range.1;
            for source in source_range.0..=source_range.1 {
                for multiplier in multiplier_range.0..=multiplier_range.1 {
                    let actual = source * multiplier;
                    assert!(
                        range.contains(&actual),
                        "{:?} * {:?} -> {:?} for {} * {}",
                        source_range,
                        multiplier_range,
                        expected_range,
                        source,
                        multiplier,
                    );
                }
            }
        }
    }

    fn validate_inv_div_source_range(source_range: (i64, i64), operand_range: (i64, i64), result_range: (i64, i64)) {
        // Ensure the source values on either extreme of the range satisfy the operation.
        // N.B.: It is NOT TRUE in general that all points of the range will satisfy
        //       the operation. See test_inv_div_range_with_hole_in_source_range for an example.
        for source in [source_range.0, source_range.1] {
            assert!(
                (operand_range.0..=operand_range.1).any(|operand| {
                    if operand == 0 {
                        false
                    } else {
                        let result = source / operand;
                        (result_range.0..=result_range.1).contains(&result)
                    }
                }),
                "{:?} / {:?} -> {:?} for source {}",
                source_range,
                operand_range,
                result_range,
                source
            );
        }

        // Ensure that values outside the calculated range do not satisfy the operation.
        for source in [
            source_range.0 - 2,
            source_range.0 - 1,
            source_range.1 + 1,
            source_range.1 + 2,
        ] {
            for operand in operand_range.0..=operand_range.1 {
                if operand == 0 {
                    continue;
                }
                let result = source / operand;
                assert!(
                    !(result_range.0..=result_range.1).contains(&result),
                    "{:?} / {:?} -> {:?} for {} / {}",
                    source_range,
                    operand_range,
                    result_range,
                    source,
                    operand,
                );
            }
        }
    }

    #[test]
    fn test_inv_div_range_exact() {
        let result = (-10, -10);
        let operand = (-10, -10);

        let expected_source = (100, 109);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_exact_positive() {
        let result = (10, 10);
        let operand = (10, 10);

        let expected_source = (100, 109);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_with_hole_in_source_range() {
        let result = (-10, -10);
        let operand = (-10, -9);

        // The expected source range is (90, 109).
        // It's easy to see that both endpoints are valid sources:
        //   90  /  -9 = -10
        //   109 / -10 = -10
        // However, it is NOT TRUE that every point in this range is a valid source.
        // For example, consider 99:
        //   99 /  -9  = -11
        //   99 / -10  =  -9
        // Since our range analysis cannot exclude points from the middle of ranges,
        // the (90, 109) range is the correct answer here.
        let expected_source = (90, 109);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_operand_across_zero() {
        let result = (-10, -10);
        let operand = (-10, 5);

        let expected_source = (-54, 109);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_result_positive_operand_across_zero() {
        let result = (10, 10);
        let operand = (-10, 5);

        let expected_source = (-109, 54);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_result_across_zero_operand_positive() {
        let result = (-10, 7);
        let operand = (5, 10);

        let expected_source = (-109, 79);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_result_across_zero_operand_negative() {
        let result = (-10, 7);
        let operand = (-10, -5);

        let expected_source = (-79, 109);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_result_to_zero_operand_across_zero() {
        let result = (-7, 0);
        let operand = (-10, 5);

        let expected_source = (-39, 79);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_result_to_zero() {
        let result = (-10, 0);
        let operand = (-10, -5);

        let expected_source = (-9, 109);
        validate_inv_div_source_range(expected_source, operand, result);

        let source = inv_div_range_analysis(result, operand);

        assert_eq!(source, expected_source);
    }

    #[test]
    fn test_inv_div_range_analysis() {
        let test_range = -10i64..=10i64;
        let pairs = test_range.clone().cartesian_product(test_range);
        let range_pairs = pairs.clone().cartesian_product(pairs);

        for (result_range, operand_range) in range_pairs {
            let result_range = (
                min(result_range.0, result_range.1),
                max(result_range.0, result_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            if operand_range == (0, 0) {
                continue;
            }

            let expected_range = inv_div_range_analysis(result_range, operand_range);

            validate_inv_div_source_range(
                expected_range,
                operand_range,
                result_range,
            );
        }
    }

    #[test]
    fn test_equal_range_analysis() {
        let test_range = -10i64..=10i64;
        let pairs = test_range.clone().cartesian_product(test_range);
        let range_pairs = pairs.clone().cartesian_product(pairs);

        for (source_range, operand_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let expected_range = equal_range_analysis(source_range, operand_range);
            let range = expected_range.0..=expected_range.1;
            for source in source_range.0..=source_range.1 {
                for operand in operand_range.0..=operand_range.1 {
                    let actual = if source == operand { 1 } else { 0 };
                    assert!(
                        range.contains(&actual),
                        "{:?} == {:?} -> {:?} for {} == {}",
                        source_range,
                        operand_range,
                        expected_range,
                        source,
                        operand,
                    );
                }
            }
        }
    }

    #[test]
    fn test_add_range_analysis() {
        let test_range = -10i64..=10i64;
        let pairs = test_range.clone().cartesian_product(test_range);
        let range_pairs = pairs.clone().cartesian_product(pairs);

        for (source_range, operand_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let expected_range = add_range_analysis(source_range, operand_range);
            let range = expected_range.0..=expected_range.1;
            for source in source_range.0..=source_range.1 {
                for operand in operand_range.0..=operand_range.1 {
                    let actual = source + operand;
                    assert!(
                        range.contains(&actual),
                        "{:?} == {:?} -> {:?} for {} == {}",
                        source_range,
                        operand_range,
                        expected_range,
                        source,
                        operand,
                    );
                }
            }
        }
    }

    #[test]
    fn test_sub_range_analysis() {
        let test_range = -10i64..=10i64;
        let pairs = test_range.clone().cartesian_product(test_range);
        let range_pairs = pairs.clone().cartesian_product(pairs);

        for (source_range, operand_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let expected_range = sub_range_analysis(source_range, operand_range);
            let range = expected_range.0..=expected_range.1;
            for source in source_range.0..=source_range.1 {
                for operand in operand_range.0..=operand_range.1 {
                    let actual = source - operand;
                    assert!(
                        range.contains(&actual),
                        "{:?} == {:?} -> {:?} for {} == {}",
                        source_range,
                        operand_range,
                        expected_range,
                        source,
                        operand,
                    );
                }
            }
        }
    }

    #[test]
    fn test_mod_range_analysis() {
        let source_test_range = 0i64..=20i64;
        let source_pairs = source_test_range
            .clone()
            .cartesian_product(source_test_range);

        let operand_test_range = 1i64..=20i64;
        let operand_pairs = operand_test_range
            .clone()
            .cartesian_product(operand_test_range);
        let range_pairs = source_pairs.cartesian_product(operand_pairs);

        for (source_range, operand_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let expected_range = mod_range_analysis(source_range, operand_range);
            let range = expected_range.0..=expected_range.1;
            for source in source_range.0..=source_range.1 {
                for operand in operand_range.0..=operand_range.1 {
                    let actual = source % operand;
                    assert!(
                        range.contains(&actual),
                        "{:?} == {:?} -> {:?} for {} == {}",
                        source_range,
                        operand_range,
                        expected_range,
                        source,
                        operand,
                    );
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn execute_input_range_analysis(
        result_range: (i64, i64),
        source_range: (i64, i64),
        operand_range: (i64, i64),
        input_range_func: fn(
            &mut BTreeMap<usize, (i64, i64)>,
            &mut BTreeMap<usize, (i64, i64)>,
            (i64, i64),
            (i64, i64),
            (i64, i64),
            RegisterState,
            Operand,
            &[RegisterState; 4],
        ),
    ) -> ((i64, i64), (i64, i64)) {
        let mut input_ranges: BTreeMap<usize, (i64, i64)> = Default::default();
        let mut value_ranges: BTreeMap<usize, (i64, i64)> =
            [(0, source_range), (1, operand_range), (2, result_range)]
                .into_iter()
                .collect();

        input_range_func(
            &mut input_ranges,
            &mut value_ranges,
            result_range,
            source_range,
            operand_range,
            RegisterState::Unknown(0),
            Operand::Register(Register(1)),
            &[
                // the register state before the instruction executed
                RegisterState::Unknown(0),
                RegisterState::Unknown(1),
                RegisterState::Exact(0),
                RegisterState::Exact(0),
            ],
        );

        (value_ranges[&0], value_ranges[&1])
    }

    #[test]
    fn test_add_input_range_analysis() {
        let source_test_range = -10i64..=10i64;
        let source_pairs = source_test_range
            .clone()
            .cartesian_product(source_test_range);

        let operand_test_range = -10i64..=10i64;
        let operand_pairs = operand_test_range
            .clone()
            .cartesian_product(operand_test_range);
        let range_pairs = source_pairs.cartesian_product(operand_pairs);

        for (source_range, operand_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let result_range = add_range_analysis(source_range, operand_range);

            let (recovered_source_range, recovered_operand_range) = execute_input_range_analysis(
                result_range,
                MAX_VALUE_RANGE,
                operand_range,
                add_input_range_analysis,
            );
            assert_eq!(recovered_operand_range, operand_range);
            for source in recovered_source_range.0..=recovered_source_range.1 {
                assert!((operand_range.0..=operand_range.1).any(|operand| {
                    (result_range.0..=result_range.1).contains(&(source + operand))
                }));
            }

            let (recovered_source_range, recovered_operand_range) = execute_input_range_analysis(
                result_range,
                source_range,
                MAX_VALUE_RANGE,
                add_input_range_analysis,
            );
            assert_eq!(recovered_source_range, source_range);
            for operand in recovered_operand_range.0..=recovered_operand_range.1 {
                assert!((source_range.0..=source_range.1).any(|source| {
                    (result_range.0..=result_range.1).contains(&(source + operand))
                }));
            }
        }
    }

    #[test]
    fn test_mul_input_range_analysis_mul_by_zero() {
        let source = (23, 25);
        let operand = (0, 1);
        let result = (0, 5);

        // The known result range is no wider than the computed result range.
        let computed_result = mul_range_analysis(source, operand);
        assert!(computed_result.0 <= result.0);
        assert!(computed_result.1 >= result.1);

        let (computed_source, computed_operand) =
            execute_input_range_analysis(result, source, operand, mul_input_range_analysis);
        assert_eq!(computed_source, (23, 25));
        assert_eq!(computed_operand, (0, 0));
    }

    #[test]
    fn test_mul_input_range_analysis_special() {
        let source = (-500, 1000);
        let operand = (-9, -8);
        let result = (80, 90);

        // The known result range is no wider than the computed result range.
        let computed_result = mul_range_analysis(source, operand);
        assert!(computed_result.0 <= result.0);
        assert!(computed_result.1 >= result.1);

        let (computed_source, computed_operand) =
            execute_input_range_analysis(result, source, operand, mul_input_range_analysis);
        assert_eq!(computed_source, (-11, -9));
        assert_eq!(computed_operand, (-9, -8));
    }

    #[test]
    fn test_mul_input_range_analysis() {
        let source_test_range = -10i64..=10i64;
        let source_pairs = source_test_range
            .clone()
            .cartesian_product(source_test_range);

        let operand_test_range = -10i64..=10i64;
        let operand_pairs = operand_test_range
            .clone()
            .cartesian_product(operand_test_range);
        let range_pairs = source_pairs.cartesian_product(operand_pairs);

        for (source_range, operand_range) in range_pairs {
            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let result_range = mul_range_analysis(source_range, operand_range);

            let (recovered_source_range, recovered_operand_range) = execute_input_range_analysis(
                result_range,
                MAX_VALUE_RANGE,
                operand_range,
                mul_input_range_analysis,
            );
            println!(
                "inf {:?} {:?} -> {:?} {:?}",
                operand_range, result_range, recovered_source_range, recovered_operand_range
            );
            assert_eq!(recovered_operand_range, operand_range);
            for source in recovered_source_range.0..=recovered_source_range.1 {
                println!("{} * {:?} => {:?}", source, operand_range, result_range);
                assert!((operand_range.0..=operand_range.1).any(|operand| {
                    (result_range.0..=result_range.1).contains(&(source * operand))
                }));
            }

            let (recovered_source_range, recovered_operand_range) = execute_input_range_analysis(
                result_range,
                source_range,
                MAX_VALUE_RANGE,
                mul_input_range_analysis,
            );
            println!(
                "{:?} inf {:?} -> {:?} {:?}",
                source_range, result_range, recovered_source_range, recovered_operand_range
            );
            assert_eq!(recovered_source_range, source_range);
            for operand in recovered_operand_range.0..=recovered_operand_range.1 {
                println!("{:?} * {} => {:?}", source_range, operand, result_range);
                assert!((source_range.0..=source_range.1).any(|source| {
                    (result_range.0..=result_range.1).contains(&(source * operand))
                }));
            }
        }
    }

    #[test]
    fn test_div_input_range_analysis_special() {
        let source = (-1000, 500);
        let operand = (-10, 0);
        let result = (1, 10);

        // The known result range is no wider than the computed result range.
        let computed_result = div_range_analysis(source, operand);
        assert!(computed_result.0 <= result.0);
        assert!(computed_result.1 >= result.1);

        let (computed_source, computed_operand) =
            execute_input_range_analysis(result, source, operand, div_input_range_analysis);
        assert_eq!(computed_source, (-109, -1));
        assert_eq!(computed_operand, (-10, 0));
    }

    #[test]
    fn test_div_input_range_analysis_special2() {
        let source = (-10, -10);
        let operand = (-500, 300);
        let result = (-10, 10);

        // The known result range is no wider than the computed result range.
        let computed_result = div_range_analysis(source, operand);
        assert!(computed_result.0 <= result.0);
        assert!(computed_result.1 >= result.1);

        let (computed_source, computed_operand) =
            execute_input_range_analysis(result, source, operand, div_input_range_analysis);
        assert_eq!(computed_source, (-10, -10));
        assert_eq!(computed_operand, (-10, 10));
    }

    #[test]
    fn test_div_input_range_analysis() {
        let source_test_range = -10i64..=10i64;
        let source_pairs = source_test_range
            .clone()
            .cartesian_product(source_test_range);

        let operand_test_range = -10i64..=10i64;
        let operand_pairs = operand_test_range
            .clone()
            .cartesian_product(operand_test_range);
        let range_pairs = source_pairs.cartesian_product(operand_pairs);

        for (source_range, operand_range) in range_pairs {
            if operand_range == (0, 0) {
                continue;
            }

            let source_range = (
                min(source_range.0, source_range.1),
                max(source_range.0, source_range.1),
            );
            let operand_range = (
                min(operand_range.0, operand_range.1),
                max(operand_range.0, operand_range.1),
            );

            let result_range = div_range_analysis(source_range, operand_range);

            let (recovered_source_range, recovered_operand_range) = execute_input_range_analysis(
                result_range,
                MAX_VALUE_RANGE,
                operand_range,
                div_input_range_analysis,
            );
            println!(
                "inf {:?} {:?} -> {:?} {:?}",
                operand_range, result_range, recovered_source_range, recovered_operand_range
            );
            assert_eq!(recovered_operand_range, operand_range);
            assert!(recovered_source_range.0.abs() <= 1000);
            assert!(recovered_source_range.1.abs() <= 1000);
            for source in recovered_source_range.0..=recovered_source_range.1 {
                println!("{} / {:?} => {:?}", source, operand_range, result_range);
                assert!((operand_range.0..=operand_range.1).any(|operand| {
                    if operand == 0 {
                        false
                    } else {
                        (result_range.0..=result_range.1).contains(&(source / operand))
                    }
                }));
            }

            let (recovered_source_range, recovered_operand_range) = execute_input_range_analysis(
                result_range,
                source_range,
                MAX_VALUE_RANGE,
                div_input_range_analysis,
            );
            println!(
                "{:?} inf {:?} -> {:?} {:?}",
                source_range, result_range, recovered_source_range, recovered_operand_range
            );
            assert_eq!(recovered_source_range, source_range);
            assert!(recovered_operand_range.0.abs() <= 1000);
            assert!(recovered_operand_range.1.abs() <= 1000);
            for operand in recovered_operand_range.0..=recovered_operand_range.1 {
                if operand == 0 {
                    continue;
                }

                println!("{:?} / {} => {:?}", source_range, operand, result_range);
                assert!((source_range.0..=source_range.1).any(|source| {
                    (result_range.0..=result_range.1).contains(&(source / operand))
                }));
            }
        }
    }
}
