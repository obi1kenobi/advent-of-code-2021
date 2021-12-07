use std::{env, fs};

#[allow(unused_imports)]
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

    let mut input_data: Vec<u64> = content
        .trim_end()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    input_data.sort_unstable();

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

fn solve_part1(positions: &[u64]) -> u64 {
    let mut cost: u64 = positions.iter().map(|pos| *pos - positions[0]).sum();
    let mut lowest_cost = cost;

    let mut left_crabs = 0u64;
    let mut right_crabs = positions.len() as u64;

    for (previous_pos, current_pos) in positions.iter().tuple_windows() {
        left_crabs += 1;
        right_crabs -= 1;

        assert!(left_crabs < positions.len() as u64);

        let distance = *current_pos - *previous_pos;
        cost += distance * left_crabs;
        cost -= distance * right_crabs;

        if cost < lowest_cost {
            lowest_cost = cost;
        }
    }

    lowest_cost
}

fn solve_part2(positions: &[u64]) -> u64 {
    let mut cost: u64 = positions.iter().map(|pos| {
        let delta = *pos - positions[0];
        delta * (delta + 1) / 2
    }).sum();
    let mut lowest_cost = cost;

    let mut left_crabs = 0u64;
    let mut right_crabs = positions.len() as u64;

    let mut delta_positive = 0u64;
    let mut delta_negative: u64 = positions.iter().map(|pos| *pos - positions[0]).sum();

    for (previous_pos, current_pos) in positions.iter().tuple_windows() {
        left_crabs += 1;
        right_crabs -= 1;

        assert!(left_crabs < positions.len() as u64);

        let distance = *current_pos - *previous_pos;
        for _offset in 0..distance {
            delta_positive += left_crabs;

            cost += delta_positive;
            cost -= delta_negative;

            if cost < lowest_cost {
                lowest_cost = cost;
            }

            delta_negative -= right_crabs;
        }
    }

    lowest_cost
}
