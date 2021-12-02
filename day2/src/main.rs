use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let input_data: Vec<(&str, u64)> = content
        .trim_end()
        .split('\n')
        .map(|x| {
            let (depth, movement) = x.split_once(' ').unwrap();
            (depth, movement.parse().unwrap())
        })
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

fn solve_part1(data: &[(&str, u64)]) -> u64 {
    let mut position = 0;
    let mut depth = 0;

    for (command, distance) in data.iter() {
        match *command {
            "forward" => position += distance,
            "up" => depth -= distance,
            "down" => depth += distance,
            _ => unreachable!(),
        }
    }

    position * depth
}

fn solve_part2(data: &[(&str, u64)]) -> u64 {
    let mut position = 0;
    let mut depth = 0;
    let mut aim = 0;

    for (command, distance) in data.iter() {
        match *command {
            "forward" => {
                position += distance;
                depth += aim * distance;
            }
            "up" => aim -= distance,
            "down" => aim += distance,
            _ => unreachable!(),
        }
    }

    position * depth
}
