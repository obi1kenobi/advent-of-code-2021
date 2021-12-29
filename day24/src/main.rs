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
        "final" => {
            let optimized_instrs = get_optimized_instructions(&input_data);
            println!("{}", InstructionStream(&optimized_instrs));
        }
        "optimize" => {
            let analysis = optimize_program(&input_data);
            println!("{}", analysis);
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

fn get_optimized_instructions(input_program: &[Instruction]) -> Vec<Instruction> {
    optimize_program(input_program).get_optimized_instructions()
}

fn optimize_program(input_program: &[Instruction]) -> Analysis {
    let analysis = analyze_program(input_program);

    finalize_optimization(analysis)
}

fn fixpoint_iteration(mut current_analysis: Analysis) -> Analysis {
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

fn initialize_analysis(input_program: &[Instruction]) -> Analysis {
    let analysis: Analysis = input_program.iter().cloned().collect_vec().into();

    analysis.operation_definedness()
}

fn analyze_program(input_program: &[Instruction]) -> Analysis {
    fixpoint_iteration(initialize_analysis(input_program))
}

/// These optimization steps make destructive changes to the input program,
/// such that only a limited set of analysis steps are possible from this point onward.
///
/// IMPORTANT: These steps are not compatible with backward value range analysis!
///            Running them after backward value range analysis is likely to produce programs
///            that don't actually match the semantics of the original input program.
///
/// TODO: Express these limitations in the type of Analysis.
fn finalize_optimization(analysis: Analysis) -> Analysis {
    analysis
        // Instruction pruning isn't useful in the early passes, so save it until later.
        .prune_for_no_change_in_registers()
        // Keep this pass near the bottom, since prior analysis passes are not compatible with it.
        .unused_register_elimination()
        .unused_result_elimination()
}

fn find_part1_input_ranges(input_program: &[Instruction]) -> Vec<ValueRange> {
    let mut current_analysis = analyze_program(input_program);

    // Update the analysis with the information that the last z register value is 0.
    let last_z_register_id = current_analysis
        .register_states
        .values()
        .last()
        .unwrap()
        .registers
        .last()
        .unwrap();
    current_analysis
        .values
        .narrow_value_range(*last_z_register_id, &ValueRange::new_exact(0));

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

    // Don't trust this analysis' idea of what instructions should be pruned,
    // since with its pruning choices, inputs that don't lead to the last Z value being 0
    // are undefined behavior (UB).
    // ***** THIS IS NOT THE SAME AS INPUTS BEING IN THEIR COMPUTED RANGES! *****
    // For example:
    // - Say the program is such that the last Z value is only 0 if Input_4 == Input_7 + 4.
    // - The computed range for Input_4 is 4..=9, and for Input_7 is 1..=5.
    // - Per the above, choosing Input_4 = Input_7 = 5 is undefined behavior, EVEN THOUGH both
    //   chosen values are within their respective ranges.
    //
    // Instead of trusting the pruning decisions of this analysis, we only output
    // the detected input ranges and then re-run the analysis in only the forward direction.
    // In that case, merely respecting the input values' ranges is sufficient to avoid UB,
    // and those pruning decisions can be trusted.
    current_analysis
        .inputs
        .into_iter()
        .map(|vid| current_analysis.values[&vid].range())
        .collect()
}

#[allow(unused_variables)]
fn solve_part1(data: &[Instruction]) -> u64 {
    // First, optimize the program as much as possible
    // without considering the desired value for the last register.
    let optimized_program = get_optimized_instructions(data);

    // Then, apply the information that the last Z value is 0 and figure out
    // the ranges of the input data necessary for the last Z value to be 0.
    let input_ranges = find_part1_input_ranges(&optimized_program);

    // Then, re-analyze the optimized program, and apply the input ranges from the prior step.
    let mut current_analysis = initialize_analysis(&optimized_program);
    for (index, range) in input_ranges.into_iter().enumerate() {
        current_analysis.values.narrow_value_range(current_analysis.inputs[index], &range);
    }

    // Now finish optimizing the program with the input range bounds applied.
    current_analysis = fixpoint_iteration(current_analysis);

    println!("{}", finalize_optimization(current_analysis));

    0
}

#[allow(unused_variables)]
fn solve_part2(data: &[Instruction]) -> usize {
    todo!()
}
