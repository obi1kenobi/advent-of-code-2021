use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, hash_map::Entry},
    env, fs,
};

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

    let input_data: Vec<Vec<u64>> = content
        .trim_end()
        .split('\n')
        .map(|x| x.chars().map(|c| c.to_digit(10).unwrap() as u64).collect())
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

const NEIGHBOR_OFFSETS: [(i64, i64); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

fn neighbors_with_risk(
    risks: &[Vec<u64>],
    location: (usize, usize),
) -> impl Iterator<Item = ((usize, usize), u64)> + '_ {
    let (x, y) = location;
    let max_x = risks.len() as i64;
    let max_y = risks[0].len() as i64;

    NEIGHBOR_OFFSETS.iter().filter_map(move |(dx, dy)| {
        let nx = x as i64 + dx;
        let ny = y as i64 + dy;

        if nx < 0 || ny < 0 || nx >= max_x || ny >= max_y {
            None
        } else {
            let next_x = nx as usize;
            let next_y = ny as usize;
            Some(((next_x, next_y), risks[next_x][next_y]))
        }
    })
}

fn process_location(
    risks: &[Vec<u64>],
    visit_risks: &mut HashMap<(usize, usize), u64>,
    pq: &mut BinaryHeap<Reverse<(u64, (usize, usize))>>,
    location: (usize, usize),
    total_risk: u64,
) {
    match visit_risks.entry(location) {
        Entry::Occupied(e) => {
            assert!(*e.get() <= total_risk);
            return;
        },
        Entry::Vacant(e) => {
            e.insert(total_risk);
        }
    }

    for (neighbor_location, neighbor_risk) in neighbors_with_risk(risks, location) {
        let next_risk = total_risk + neighbor_risk;
        if let Some(known_risk) = visit_risks.get(&neighbor_location) {
            assert!(*known_risk < next_risk);
            continue;
        }

        pq.push(Reverse((next_risk, neighbor_location)));
    }
}

#[allow(unused_variables)]
fn solve_part1(risks: &[Vec<u64>]) -> u64 {
    let mut visit_risks: HashMap<(usize, usize), u64> = Default::default();
    let mut pq: BinaryHeap<Reverse<(u64, (usize, usize))>> = Default::default();

    pq.push(Reverse((0, (0, 0))));

    let target = (risks.len() - 1, risks[0].len() - 1);

    loop {
        let (next_risk, next_location) = pq.pop().unwrap().0;
        if next_location == target {
            break next_risk;
        }

        process_location(risks, &mut visit_risks, &mut pq, next_location, next_risk);
    }
}

fn bias_line(data: &[u64], bias: u64) -> impl Iterator<Item = u64> + '_ {
    data.iter().map(move |value| {
        let next_value = (*value + bias) % 9;
        if next_value == 0 {
            9
        } else {
            next_value
        }
    })
}

fn bias_rows(data: &[Vec<u64>], bias: u64) -> impl Iterator<Item = Vec<u64>> + '_ {
    data.iter().map(move |row| {
        bias_line(row, bias).collect()
    })
}

fn bias_copy_map(data: &[Vec<u64>], copies: usize) -> Vec<Vec<u64>> {
    let duplicated_top: Vec<Vec<u64>> = data.iter().map(|row| {
        (0..copies).flat_map(|bias| bias_line(row, bias as u64)).collect()
    }).collect();

    let mut result = duplicated_top.clone();

    for bias in 1..copies {
        result.extend(bias_rows(&duplicated_top, bias as u64));
    }

    result
}

#[allow(unused_variables)]
fn solve_part2(risks: &[Vec<u64>]) -> u64 {
    let risks = bias_copy_map(risks, 5);

    let mut visit_risks: HashMap<(usize, usize), u64> = Default::default();
    let mut pq: BinaryHeap<Reverse<(u64, (usize, usize))>> = Default::default();

    pq.push(Reverse((0, (0, 0))));

    let target = (risks.len() - 1, risks[0].len() - 1);

    loop {
        let (next_risk, next_location) = pq.pop().unwrap().0;
        if next_location == target {
            break next_risk;
        }

        process_location(&risks, &mut visit_risks, &mut pq, next_location, next_risk);
    }
}
