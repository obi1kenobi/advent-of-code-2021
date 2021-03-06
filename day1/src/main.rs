use std::{env, fs};

use itertools::Itertools;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let input_data: Vec<u64> = content
        .trim_end()
        .split('\n')
        .map(|x| x.parse().unwrap())
        .collect();

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

fn solve_part1(data: &[u64]) -> usize {
    data.iter().tuple_windows().map(|(first, second)| {
        if second > first {
            1
        } else {
            0
        }
    }).sum()
}

fn solve_part2(data: &[u64]) -> usize {
    data.iter().tuple_windows().tuple_windows().map(|((first_a, first_b, first_c), (second_a, second_b, second_c))| {
        let sum_first = first_a + first_b + first_c;
        let sum_second = second_a + second_b + second_c;
        if sum_second > sum_first {
            1
        } else {
            0
        }
    }).sum()
}
