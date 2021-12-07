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

    let input_data: Vec<usize> = content
        .trim_end()
        .split(',')
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

fn simulate_fish(data: &[usize], total_days: usize) -> usize {
    let mut today: [usize; 9] = Default::default();
    let mut tomorrow: [usize; 9] = Default::default();

    for fish in data {
        tomorrow[*fish] += 1;
    }

    for _tomorrow_is_after_day in 1..=total_days {
        // Tomorrow becomes today.
        today[0..9].clone_from_slice(&tomorrow[0..9]);

        // Advance the timers.
        tomorrow[0..8].clone_from_slice(&today[1..9usize]);

        tomorrow[8] = today[0];   // Spawn new fish, then
        tomorrow[6] += today[0];  // add the spawning fish back to the queue.
    }

    tomorrow.iter().sum()
}

fn solve_part1(data: &[usize]) -> usize {
    simulate_fish(data, 80)
}

fn solve_part2(data: &[usize]) -> usize {
    simulate_fish(data, 256)
}
