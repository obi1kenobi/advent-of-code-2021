#![feature(map_try_insert)]

use std::{
    env, fs,
};

use analysis::Analysis;
#[allow(unused_imports)]
use itertools::Itertools;

use parser::{parse_program, Instruction};

#[allow(unused_imports)]
use crate::parser::InstructionStream;

mod parser;
mod analysis;

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

fn optimize_program(input_program: &[Instruction]) -> Analysis {
    let analysis: Analysis =  input_program.iter().cloned().collect_vec().into();

    let mut current_analysis = analysis.operation_definedness();

    // Iterate the next few passes until a fixpoint is found,
    // since the passes create opportunities for each other to optimize further.
    let mut value_ranges = current_analysis.values.clone();
    current_analysis = loop {
        current_analysis = current_analysis
            .constant_propagation()
            .known_operation_results()
            .forward_value_range_analysis()
            .matched_mul_and_div_or_mod()
        ;

        if current_analysis.values == value_ranges {
            break current_analysis;
        } else {
            value_ranges = current_analysis.values.clone();
        }
    };

    current_analysis
        // Instruction pruning isn't useful in the early passes, so save it until later.
        .prune_for_no_change_in_registers()

        // Keep this pass near the bottom, since prior analysis passes are not compatible with it.
        .unused_register_elimination()

        // At the moment, this pass doesn't seem to do anything, reconsider and maybe enable
        // if it does something after more analysis passes are implemented.
        .unused_result_elimination()
}

#[allow(unused_variables)]
fn solve_part1(data: &[Instruction]) -> u64 {
    let optimized_program = optimize_program(data);

    println!("{}", optimized_program);

    0
}

#[allow(unused_variables)]
fn solve_part2(data: &[Instruction]) -> usize {
    todo!()
}
