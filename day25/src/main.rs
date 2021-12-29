use std::{env, fs, mem::swap};

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

    let input_data: Vec<Vec<char>> = content
        .trim_end()
        .lines()
        .map(|x| x.chars().collect_vec())
        .collect();

    match part {
        "1" => {
            let result = solve_part1(input_data);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(input_data);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

fn wrap_coordinates(map: &[Vec<char>], x: usize, y: usize) -> (usize, usize) {
    (x % map.len(), y % map[0].len())
}

fn advance(from: &[Vec<char>], to: &mut Vec<Vec<char>>) {
    reset(to);

    // Move east-facing first.
    for (x, row) in from.iter().enumerate() {
        for (y, pos) in row.iter().enumerate() {
            let move_offset = match *pos {
                '.' | 'v' => None,
                '>' => Some((0, 1)),
                _ => unreachable!(),
            };
            if let Some((dx, dy)) = move_offset {
                let (nx, ny) = wrap_coordinates(from, x + dx, y + dy);
                if from[nx][ny] == '.' {
                    // Free spot! Can move over.
                    to[nx][ny] = *pos;
                } else {
                    // Spot was taken, stay in place.
                    to[x][y] = *pos;
                }
            }
        }
    }

    // Move south-facing next, remembering to consider
    // the post-move locations of the east-facing entities.
    for (x, row) in from.iter().enumerate() {
        for (y, pos) in row.iter().enumerate() {
            let move_offset = match *pos {
                '.' | '>' => None,
                'v' => Some((1, 0)),
                _ => unreachable!(),
            };
            if let Some((dx, dy)) = move_offset {
                let (nx, ny) = wrap_coordinates(from, x + dx, y + dy);
                if from[nx][ny] != 'v' && to[nx][ny] == '.' {
                    // Free spot! Can move over.
                    to[nx][ny] = *pos;
                } else {
                    // Spot was taken, stay in place.
                    to[x][y] = *pos;
                }
            }
        }
    }
}

fn reset(map: &mut Vec<Vec<char>>) {
    map.iter_mut().for_each(|row| row.iter_mut().for_each(|c| *c = '.'));
}

#[allow(unused_variables)]
fn solve_part1(mut start: Vec<Vec<char>>) -> usize {
    let mut next = start.clone();

    let mut from = &mut start;
    let mut to = &mut next;

    let mut count = 0usize;
    loop {
        swap(&mut from, &mut to);

        advance(from, to);
        count += 1;

        if from == to {
            break count;
        }
    }
}

#[allow(unused_variables)]
fn solve_part2(data: Vec<Vec<char>>) -> usize {
    todo!()
}
