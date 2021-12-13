use std::{env, fs, collections::HashSet};

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

#[rustfmt::skip]
const NEIGHBOR_OFFSETS: [(isize, isize); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

fn neighbors(
    max_x: usize,
    max_y: usize,
    point: (usize, usize),
) -> impl Iterator<Item = (usize, usize)> {
    let x = point.0 as isize;
    let y = point.1 as isize;

    NEIGHBOR_OFFSETS.iter().filter_map(move |(dx, dy)| {
        let nx = x + *dx;
        let ny = y + *dy;

        if nx >= 0 && ny >= 0 && (nx as usize) < max_x && (ny as usize) < max_y {
            Some(((nx as usize), (ny as usize)))
        } else {
            None
        }
    })
}

fn evaluate_point(
    data: &mut [Vec<u64>],
    flashes: &mut HashSet<(usize, usize)>,
    point: (usize, usize),
) {
    let (x, y) = point;
    if data[x][y] > 9 {
        let is_new = flashes.insert(point);
        if is_new {
            for neighbor in neighbors(data.len(), data[0].len(), point) {
                data[neighbor.0][neighbor.1] += 1;
                evaluate_point(data, flashes, neighbor);
            }
        }
    }
}

#[allow(unused_variables)]
fn solve_part1(data: &[Vec<u64>]) -> usize {
    let mut data: Vec<Vec<u64>> = data.iter().map(Vec::clone).collect();
    let max_x = data.len();
    let max_y = data[0].len();

    let mut flashes: HashSet<(usize, usize)> = HashSet::with_capacity(max_x * max_y);
    let mut total_flashes = 0usize;

    for _ in 1..=100 {
        // increment all by one
        data.iter_mut().for_each(|v| v.iter_mut().for_each(|octopus| *octopus += 1));

        // process all flashes
        for i in 0..max_x {
            for j in 0..max_y {
                evaluate_point(&mut data, &mut flashes, (i, j));
            }
        }
        total_flashes += flashes.len();

        // set all flashed points to zero
        for flash_point in flashes.drain() {
            let (flash_x, flash_y) = flash_point;
            data[flash_x][flash_y] = 0;
        }
    }

    total_flashes
}

#[allow(unused_variables)]
fn solve_part2(data: &[Vec<u64>]) -> usize {
    let mut data: Vec<Vec<u64>> = data.iter().map(Vec::clone).collect();
    let max_x = data.len();
    let max_y = data[0].len();
    let grid_count = max_x * max_y;

    let mut flashes: HashSet<(usize, usize)> = HashSet::with_capacity(grid_count);
    let mut step_count = 0usize;

    loop {
        step_count += 1;

        // increment all by one
        data.iter_mut().for_each(|v| v.iter_mut().for_each(|octopus| *octopus += 1));

        // process all flashes
        for i in 0..max_x {
            for j in 0..max_y {
                evaluate_point(&mut data, &mut flashes, (i, j));
            }
        }

        // if every grid point flashed, we've found the answer
        if flashes.len() == grid_count {
            break step_count;
        }

        // set all flashed points to zero
        for flash_point in flashes.drain() {
            let (flash_x, flash_y) = flash_point;
            data[flash_x][flash_y] = 0;
        }
    }
}
