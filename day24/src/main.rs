#![feature(map_try_insert)]

use std::{
    env, fs,
};

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

#[allow(unused_variables)]
fn solve_part1(data: &[Instruction]) -> u64 {
    0
}

#[allow(unused_variables)]
fn solve_part2(data: &[Instruction]) -> usize {
    todo!()
}
