#![feature(map_try_insert)]

use std::{env, fs};

use analysis::Analysis;
#[allow(unused_imports)]
use itertools::Itertools;

use parser::{parse_program, Instruction};

use crate::analysis::values::ValueRange;
#[allow(unused_imports)]
use crate::parser::InstructionStream;

mod analysis;
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
        "optimize" => {
            let optimized_instrs = optimize_program(&input_data);
            println!("{}", InstructionStream(&optimized_instrs));
        }
        "intermediate" => {
            let analysis = analyze_program(&input_data);
            println!("{}", analysis);
        }
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

fn optimize_program(input_program: &[Instruction]) -> Vec<Instruction> {
    let analysis = analyze_program(input_program);

    finalize_optimization(analysis).get_optimized_instructions()
}

fn analyze_program(input_program: &[Instruction]) -> Analysis {
    let analysis: Analysis = input_program.iter().cloned().collect_vec().into();

    let mut current_analysis = analysis.operation_definedness();

    // Iterate the next few passes until a fixpoint is found,
    // since the passes create opportunities for each other to optimize further.
    let mut value_ranges = current_analysis.values.clone();
    loop {
        current_analysis = current_analysis
            .constant_propagation()
            .known_operation_results()
            .forward_value_range_analysis()
            .matched_mul_and_div_or_mod();

        if current_analysis.values == value_ranges {
            break current_analysis;
        } else {
            value_ranges = current_analysis.values.clone();
        }
    }
}

/// These optimization steps make destructive changes to the input program,
/// such that only a limited set of analysis steps are possible from this point onward.
/// TODO: Express these limitations in the type of Analysis.
fn finalize_optimization(analysis: Analysis) -> Analysis {
    analysis
        // Instruction pruning isn't useful in the early passes, so save it until later.
        .prune_for_no_change_in_registers()
        // Keep this pass near the bottom, since prior analysis passes are not compatible with it.
        .unused_register_elimination()
        .unused_result_elimination()
}

#[allow(unused_variables)]
fn solve_part1(data: &[Instruction]) -> u64 {
    let mut current_analysis = analyze_program(data);

    // Update the analysis with the information that the last z register value is 0.
    let last_z_register_id = current_analysis.register_states.values().last().unwrap().registers.last().unwrap();
    current_analysis.values.narrow_value_range(*last_z_register_id, &ValueRange::new_exact(0));

    // Back-propagate the z=0 information up through the rest of the range information.
    current_analysis = current_analysis.backward_value_range_analysis();

    // Iterate the next few passes until a fixpoint is found,
    // since the passes create opportunities for each other to optimize further.
    let mut value_ranges = current_analysis.values.clone();
    current_analysis = loop {
        current_analysis = current_analysis
            .constant_propagation()
            .known_operation_results()
            .forward_value_range_analysis()
            .backward_value_range_analysis()
            .matched_mul_and_div_or_mod();

        if current_analysis.values == value_ranges {
            break current_analysis;
        } else {
            value_ranges = current_analysis.values.clone();
        }
    };

    println!("{}", current_analysis);

    0
}

#[allow(unused_variables)]
fn solve_part2(data: &[Instruction]) -> usize {
    todo!()
}
