use std::{env, fs, cmp::Ordering};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let input_data: Vec<&str> = content
        .trim_end()
        .split('\n')
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

fn solve_part1(data: &[&str]) -> u64 {
    let mut gamma = 0;
    let mut epsilon = 0;

    let mut ones_counter = Vec::new();
    ones_counter.resize(data[0].len(), 0usize);
    let mut zeros_counter = Vec::new();
    zeros_counter.resize(data[0].len(), 0usize);

    for entry in data.iter().copied() {
        for (index, digit) in entry.chars().rev().enumerate() {
            match digit {
                '1' => ones_counter[index] += 1,
                '0' => zeros_counter[index] += 1,
                _ => unreachable!("{}", digit),
            }
        }
    }

    for (index, (ones, zeros)) in ones_counter.iter().zip(zeros_counter.iter()).enumerate() {
        if ones > zeros {
            gamma += 1 << index;
        } else {
            epsilon += 1 << index;
        }
    }

    gamma * epsilon
}

fn solve_part2(data: &[&str]) -> u64 {
    let binary_data: Vec<_> = data.iter().map(|s| {
        let mut value = 0u64;
        for c in s.chars() {
            value <<= 1;
            match c {
                '1' => value += 1,
                '0' => {},
                _ => unreachable!("{}", c),
            }
        }
        value
    }).collect();

    let num_positions = data[0].len();

    let mut oxygen = 0u64;
    let mut co2 = 0u64;

    let mut inv_oxygen_mask = 0u64;
    let mut inv_co2_mask = 0u64;
    for index in (0..num_positions).rev() {
        let mut oxygen_ones = 0usize;
        let mut oxygen_zeros = 0usize;
        let mut co2_ones = 0usize;
        let mut co2_zeros = 0usize;

        for entry in binary_data.iter().copied() {
            let (consider_oxygen, consider_co2) = (
                (entry & oxygen == oxygen) && (!entry & inv_oxygen_mask == inv_oxygen_mask),
                (entry & co2 == co2) && (!entry & inv_co2_mask == inv_co2_mask),
            );

            if (entry & (1 << index)) != 0 {
                if consider_oxygen {
                    oxygen_ones += 1;
                }

                if consider_co2 {
                    co2_ones += 1;
                }
            } else {
                if consider_oxygen {
                    oxygen_zeros += 1;
                }

                if consider_co2 {
                    co2_zeros += 1;
                }
            }
        }

        // Since we pick the greater count, we can never pick positions with zero representatives
        // unless both counts are zero, which should never happen.
        assert!(oxygen_zeros > 0 || oxygen_ones > 0);
        match oxygen_ones.cmp(&oxygen_zeros) {
            Ordering::Less => inv_oxygen_mask += 1 << index,
            Ordering::Equal | Ordering::Greater => oxygen += 1 << index,
        }

        // Don't pick positions with zero representatives.
        if co2_ones == 0 || (co2_zeros > 0 && co2_ones >= co2_zeros) {
            inv_co2_mask += 1 << index;
        } else if co2_zeros == 0 || (co2_ones > 0 && co2_ones < co2_zeros) {
            co2 += 1 << index;
        } else {
            unreachable!("{} {}", co2_ones, co2_zeros);
        }
    }

    oxygen * co2
}
