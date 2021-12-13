use std::{env, fs};

#[allow(unused_imports)]
use itertools::Itertools;
use kth::SliceExtKth;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let input_data: Vec<&str> = content.trim_end().split('\n').collect();

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

fn matching_opening_char(c: char) -> char {
    match c {
        ')' => '(',
        ']' => '[',
        '}' => '{',
        '>' => '<',
        _ => unreachable!("{}", c),
    }
}

fn matching_closing_char(c: char) -> char {
    match c {
        '(' => ')',
        '[' => ']',
        '{' => '}',
        '<' => '>',
        _ => unreachable!("{}", c),
    }
}

// Result <
//  remaining stack if any,
//  (remaining stack, first illegal char, expected char or None for any open chunk char)
// >
#[allow(clippy::type_complexity)]
fn check_content(content: &str) -> Result<Option<Vec<char>>, (Vec<char>, char, Option<char>)> {
    let mut stack: Vec<char> = vec![];
    for c in content.chars() {
        match c {
            '(' | '[' | '{' | '<' => stack.push(c),
            ')' | ']' | '}' | '>' => {
                let opening_char = matching_opening_char(c);
                match stack.pop() {
                    Some(open) if open == opening_char => {}
                    Some(_) => {
                        return Err((stack, c, Some(opening_char)));
                    }
                    None => {
                        return Err((stack, c, None));
                    }
                }
            }
            _ => unreachable!("{}", c),
        }
    }

    if stack.is_empty() {
        Ok(None)
    } else {
        Ok(Some(stack))
    }
}

fn solve_part1(data: &[&str]) -> u64 {
    data.iter()
        .map(|content| match check_content(content) {
            Err((_, error_char, _)) => match error_char {
                ')' => 3,
                ']' => 57,
                '}' => 1197,
                '>' => 25137,
                _ => unreachable!("{}", error_char),
            },
            Ok(_) => 0,
        })
        .sum()
}

fn solve_part2(data: &[&str]) -> u64 {
    let mut scores: Vec<_> = data.iter().filter_map(|content| match check_content(content) {
        Err(_) | Ok(None) => None,
        Ok(Some(remaining_stack)) => {
            let value = remaining_stack
                .iter()
                .rev()
                .copied()
                .map(|c| match matching_closing_char(c) {
                    ')' => 1,
                    ']' => 2,
                    '}' => 3,
                    '>' => 4,
                    _ => unreachable!("{}", c),
                })
                .fold(0, |acc, x| acc * 5 + x);
            Some(value)
        }
    }).collect();

    let median_index = scores.len() / 2;
    scores.partition_by_kth(median_index);
    scores[median_index]
}
