use std::{collections::{HashMap, HashSet}, env, fs};

#[allow(unused_imports)]
use itertools::Itertools;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Cave<'a> {
    name: &'a str,
    large: bool,
    connected_to: Vec<&'a str>,
}

impl<'a> Cave<'a> {
    fn new(name: &'a str) -> Cave<'a> {
        let large = name.chars().all(|c| c.is_uppercase());
        assert!(large || name.chars().all(|c| c.is_lowercase()));
        Cave {
            name,
            large,
            connected_to: Default::default(),
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let edges: Vec<(&str, &str)> = content
        .trim_end()
        .split('\n')
        .map(|x| x.split_once('-').unwrap())
        .collect();
    let mut caves: HashMap<&str, Cave<'_>> = HashMap::with_capacity(edges.len() / 2);
    for (src, dest) in edges {
        let src_cave = caves.entry(src).or_insert_with(|| Cave::new(src));
        src_cave.connected_to.push(dest);
        let src_large = src_cave.large;

        let dest_cave = caves.entry(dest).or_insert_with(|| Cave::new(dest));
        dest_cave.connected_to.push(src);
        let dest_large = dest_cave.large;

        // if two large caves are connected to each other, then there's an infinite number of paths
        // since we can just loop between those two caves forever
        assert!(!src_large || !dest_large);
    }
    assert!(caves.contains_key(START_CAVE));
    assert!(caves.contains_key(END_CAVE));

    match part {
        "1" => {
            let result = solve_part1(&caves);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(&caves);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

const START_CAVE: &str = "start";
const END_CAVE: &str = "end";

#[allow(unused_variables)]
fn solve_part1(caves: &HashMap<&str, Cave<'_>>) -> usize {
    let current_cave = &caves[START_CAVE];

    let mut visit_stack = vec![START_CAVE];
    let mut visited_small_caves = [START_CAVE].iter().copied().collect();

    count_paths(caves, current_cave, &mut visit_stack, &mut visited_small_caves)
}

fn count_paths<'a>(
    caves: &HashMap<&'a str, Cave<'a>>,
    current_cave: &Cave<'a>,
    visit_stack: &mut Vec<&'a str>,
    visited_small_caves: &mut HashSet<&'a str>,
) -> usize {
    assert_ne!(current_cave.name, END_CAVE);

    let mut found_paths = 0usize;
    for next_name in current_cave.connected_to.iter().copied() {
        if next_name == END_CAVE {
            found_paths += 1;
        } else {
            let next_cave = &caves[next_name];
            if !next_cave.large && !visited_small_caves.insert(next_name) {
                // already visited this small cave, skip this iteration
                continue;
            }

            visit_stack.push(next_name);

            found_paths += count_paths(caves, next_cave, visit_stack, visited_small_caves);

            let popped = visit_stack.pop().unwrap();
            assert_eq!(popped, next_name);
            if !next_cave.large {
                let removed = visited_small_caves.remove(next_name);
                assert!(removed);
            }
        }
    }

    found_paths
}

fn count_paths_visiting_small_twice<'a>(
    caves: &HashMap<&'a str, Cave<'a>>,
    current_cave: &Cave<'a>,
    visit_stack: &mut Vec<&'a str>,
    visited_small_caves: &mut HashSet<&'a str>,
) -> usize {
    assert_ne!(current_cave.name, END_CAVE);

    let mut found_paths = 0usize;
    for next_name in current_cave.connected_to.iter().copied() {
        if next_name == END_CAVE {
            found_paths += 1;
        } else if next_name != START_CAVE {
            let next_cave = &caves[next_name];
            if !next_cave.large && !visited_small_caves.insert(next_name) {
                // already visited this small cave, use it as our single re-visit
                visit_stack.push(next_name);

                found_paths += count_paths(caves, next_cave, visit_stack, visited_small_caves);

                let popped = visit_stack.pop().unwrap();
                assert_eq!(popped, next_name);
            } else {
                // visit this cave regularly
                visit_stack.push(next_name);

                found_paths += count_paths_visiting_small_twice(caves, next_cave, visit_stack, visited_small_caves);

                let popped = visit_stack.pop().unwrap();
                assert_eq!(popped, next_name);
                if !next_cave.large {
                    let removed = visited_small_caves.remove(next_name);
                    assert!(removed);
                }
            }
        }
    }

    found_paths
}

#[allow(unused_variables)]
fn solve_part2(caves: &HashMap<&str, Cave<'_>>) -> usize {
    let current_cave = &caves[START_CAVE];

    let mut visit_stack = vec![START_CAVE];
    let mut visited_small_caves = [START_CAVE].iter().copied().collect();

    count_paths_visiting_small_twice(caves, current_cave, &mut visit_stack, &mut visited_small_caves)
}
