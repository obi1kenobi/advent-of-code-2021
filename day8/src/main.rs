use std::{env, fs, collections::{BTreeMap, BTreeSet}};

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

    let input_data: Vec<(Vec<&str>, Vec<&str>)> = content
        .trim_end()
        .split('\n')
        .map(|x| {
            let (data, output) = x.split_once('|').unwrap();
            (data.split_ascii_whitespace().collect(), output.split_ascii_whitespace().collect())
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

fn solve_part1(data: &[(Vec<&str>, Vec<&str>)]) -> usize {
    data.iter().map(|(_, output)| {
        output.iter().filter(|item| {
            let length = item.len();
            length == 2 || length == 3 || length == 4 || length == 7
        }).count()
    }).sum()
}

fn solve_display(clues: &[&str], challenge: &[&str]) -> u64 {
    //   000
    //  1   2
    //  1   2
    //   333
    //  4   5
    //  4   5
    //   666
    let segments: [BTreeSet<usize>; 10] = [
        [0, 1, 2, 4, 5, 6].iter().copied().collect(),    // 0
        [2, 5].iter().copied().collect(),                // 1
        [0, 2, 3, 4, 6].iter().copied().collect(),       // 2
        [0, 2, 3, 5, 6].iter().copied().collect(),       // 3
        [1, 2, 3, 5].iter().copied().collect(),          // 4
        [0, 1, 3, 5, 6].iter().copied().collect(),       // 5
        [0, 1, 3, 4, 5, 6].iter().copied().collect(),    // 6
        [0, 2, 5].iter().copied().collect(),             // 7
        [0, 1, 2, 3, 4, 5, 6].iter().copied().collect(), // 8
        [0, 1, 2, 3, 5, 6].iter().copied().collect(),    // 9
    ];
    let mut solution: BTreeMap<char, usize> = Default::default();
    let mut possible_segments: BTreeMap<char, BTreeSet<usize>> = ('a'..='g')
        .map(|c| (c, [0, 1, 2, 3, 4, 5, 6].iter().copied().collect()))
        .collect();

    let clues_by_length: BTreeMap<usize, Vec<&str>> = clues.iter()
        .map(|c| (c.len(), *c))
        .fold(Default::default(), |mut acc, (cnt, clue)| {
            acc.entry(cnt).or_default().push(clue);
            acc
        });
    for (length, cls) in &clues_by_length {
        let seen_letters: BTreeSet<char> = cls.iter().flat_map(|x| x.chars()).collect();
        let common_letters: BTreeSet<char> = cls.iter().fold(('a'..='g').collect(), |mut acc, elem| {
            acc.retain(|l| elem.contains(*l));
            acc
        });
        let varying_letters: BTreeSet<char> = seen_letters.difference(&common_letters).copied().collect();

        let matching_segments: Vec<&BTreeSet<usize>> = segments.iter().filter(|x| x.len() == *length).collect();
        let seen_segment_ids: BTreeSet<usize> = matching_segments.iter().flat_map(|s| s.iter().copied()).collect();
        let common_segment_ids: BTreeSet<usize> = matching_segments.iter().fold((0..=6).collect(), |mut acc, elem| {
            acc.retain(|x| elem.contains(x));
            acc
        });
        let varying_segment_ids: BTreeSet<usize> = seen_segment_ids.difference(&common_segment_ids).copied().collect();

        for (letter, possibilities) in &mut possible_segments {
            if common_letters.contains(letter) {
                possibilities.retain(|id| common_segment_ids.contains(id));
            } else if varying_letters.contains(letter) {
                possibilities.retain(|id| varying_segment_ids.contains(id));
            }
        }
    }

    for clue in clues {
        let possibilities: BTreeSet<usize> = segments.iter()
            .filter(|seg| clue.len() == seg.len())
            .filter(|seg| {
                let clue_segment_ids = clue.chars()
                    .flat_map(|c| possible_segments[&c].iter().copied())
                    .collect();
                seg.is_subset(&clue_segment_ids)
            })
            .fold(Default::default(), |mut acc, elem| {
                acc.extend(elem);
                acc
            });


        for letter in clue.chars() {
            let s = possible_segments.get_mut(&letter).unwrap();
            s.retain(|elem| possibilities.contains(elem));
        }

        if clue.len() == possibilities.len() {
            // any letter not in 'clue' cannot represent any of the segments in 'possibilities'
            let mut unseen_letters: BTreeSet<char> = ('a'..='g').collect();
            for letter in clue.chars() {
                unseen_letters.remove(&letter);
            }

            for letter in unseen_letters {
                let s = possible_segments.get_mut(&letter).unwrap();
                s.retain(|elem| !possibilities.contains(elem));
            }
        }
    }

    for (letter, possibilities) in &possible_segments {
        if possibilities.len() == 1 && !solution.contains_key(letter) {
            // solved a new letter!
            // remember to remove its segment ID from all other letters' possibilities
            let segment_id = possibilities.iter().next().unwrap();
            solution.insert(*letter, *segment_id);
        }
    }
    for (letter, possibilities) in &mut possible_segments {
        for (solved_letter, segment_id) in &solution {
            if letter != solved_letter {
                possibilities.remove(segment_id);
            }
        }
    }

    challenge.iter().map(|ch| {
        let lit_segments: BTreeSet<usize> = ch.chars().map(|c| solution[&c]).collect();
        segments
            .iter()
            .enumerate()
            .filter_map(|(value, seg)| {
                if seg == &lit_segments {
                    Some(value as u64)
                } else {
                    None
                }
            })
            .exactly_one()
            .unwrap()
    }).fold(0u64, |acc, elem| {
        (acc * 10) + elem
    })
}

fn solve_part2(data: &[(Vec<&str>, Vec<&str>)]) -> u64 {
    data.iter().map(|(clue, challenge)| solve_display(clue, challenge)).sum()
}
