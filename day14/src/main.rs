use std::{collections::HashMap, env, fs};

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

    let (base, rules_component) = content.trim_end().split_once("\n\n").unwrap();

    let rules: Vec<_> = rules_component
        .split('\n')
        .map(|row| {
            let (from, to) = row.split_once(" -> ").unwrap();
            assert_eq!(from.len(), 2);
            assert_eq!(to.len(), 1);

            let (from_first, from_second) = from.chars().collect_tuple().unwrap();
            let inserted = to.chars().collect_tuple::<(char,)>().unwrap().0;

            (from_first, from_second, inserted)
        })
        .collect();

    match part {
        "1" => {
            let result = solve_part1(base, &rules);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(base, &rules);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

fn expand_polymer(
    original: &HashMap<(char, char), usize>,
    rules: &HashMap<(char, char), char>,
) -> HashMap<(char, char), usize> {
    let mut result = HashMap::with_capacity(original.len());

    for (pair, count) in original {
        if let Some(inserted) = rules.get(pair) {
            let (first, second) = *pair;
            *result.entry((first, *inserted)).or_default() += *count;
            *result.entry((*inserted, second)).or_default() += *count;
        } else {
            *result.entry(*pair).or_default() += *count;
        }
    }

    result
}

fn solve_iterations(original: &str, rules: &[(char, char, char)], iterations: usize) -> usize {
    let rules: HashMap<(char, char), char> = rules
        .iter()
        .copied()
        .map(|(a, b, insert)| ((a, b), insert))
        .collect();

    let mut polymer_pairs: HashMap<(char, char), usize> = Default::default();
    for pair in original.chars().tuple_windows() {
        *polymer_pairs.entry(pair).or_default() += 1;
    }
    *polymer_pairs.entry((' ', original.chars().next().unwrap())).or_default() += 1;
    *polymer_pairs.entry((original.chars().next_back().unwrap(), ' ')).or_default() += 1;

    for _ in 0..iterations {
        polymer_pairs = expand_polymer(&polymer_pairs, &rules);
    }

    let mut counts: HashMap<char, usize> = HashMap::new();
    for ((a, b), count) in polymer_pairs {
        *counts.entry(a).or_default() += count;
        *counts.entry(b).or_default() += count;
    }
    counts.remove(&' ').unwrap();

    let max_count = counts.values().max().unwrap() / 2;
    let min_count = counts.values().min().unwrap() / 2;

    max_count - min_count
}

fn solve_part1(original: &str, rules: &[(char, char, char)]) -> usize {
    solve_iterations(original, rules, 10)
}

fn solve_part2(original: &str, rules: &[(char, char, char)]) -> usize {
    solve_iterations(original, rules, 40)
}
