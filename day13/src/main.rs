use std::{env, fs, collections::{HashSet}, cmp::{Ordering, max}};

#[allow(unused_imports)]
use itertools::Itertools;

#[derive(Debug, Clone)]
struct Fold {
    fold_along_x: bool,
    coord: i64,
}

impl From<&str> for Fold {
    fn from(row: &str) -> Self {
        let prefix = "fold along ";
        let line = row.strip_prefix(prefix).unwrap();
        let (axis, coordinate) = line.split_once('=').unwrap();
        assert!(axis == "x" || axis == "y");

        let fold_along_x = axis == "x";
        let coord = coordinate.parse().unwrap();

        Fold {
            fold_along_x,
            coord,
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

    let (dots, fold_instructions) = content.trim_end().split_once("\n\n").unwrap();

    let dot_coords: Vec<(i64, i64)> = dots
        .split('\n')
        .map(|row| {
            let coords = row.split_once(',').unwrap();
            (coords.0.parse().unwrap(), coords.1.parse().unwrap())
        })
        .collect();
    let folds: Vec<Fold> = fold_instructions.split('\n').map(Fold::from).collect();

    match part {
        "1" => {
            let result = solve_part1(&dot_coords, &folds);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(&dot_coords, &folds);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

fn apply_fold(current_dots: &mut HashSet<(i64, i64)>, fold: &Fold) {
    let dots_to_reflect: Vec<((i64, i64), (i64, i64))> = current_dots.iter().copied().filter_map(|pt| {
        let (x, y) = pt;
        if fold.fold_along_x {
            let delta_x = x - fold.coord;
            match delta_x.cmp(&0) {
                Ordering::Less => None,  // not over the fold line, no reflection needed
                Ordering::Greater => Some((pt, (fold.coord - delta_x, y))),
                Ordering::Equal => unreachable!(),
            }
        } else {
            let delta_y = y - fold.coord;
            match delta_y.cmp(&0) {
                Ordering::Less => None,  // not over the fold line, no reflection needed
                Ordering::Greater => Some((pt, (x, fold.coord - delta_y))),
                Ordering::Equal => unreachable!(),
            }
        }
    }).collect();

    for (original_dot, new_dot) in dots_to_reflect {
        let removed = current_dots.remove(&original_dot);
        assert!(removed);

        current_dots.insert(new_dot);  // might overlap, so the return may be true or false here
    }
}

fn solve_part1(dots: &[(i64, i64)], folds: &[Fold]) -> usize {
    let first_fold = folds.first().unwrap();
    let mut current_dots: HashSet<(i64, i64)> = dots.iter().copied().collect();

    apply_fold(&mut current_dots, first_fold);

    current_dots.len()
}

fn solve_part2(dots: &[(i64, i64)], folds: &[Fold]) -> String {
    let mut current_dots: HashSet<(i64, i64)> = dots.iter().copied().collect();

    for fold in folds {
        apply_fold(&mut current_dots, fold);
    }

    let mut max_x = 0i64;
    let mut max_y = 0i64;
    for (x, y) in current_dots.iter() {
        assert!(*x >= 0);
        assert!(*y >= 0);

        max_x = max(max_x, *x);
        max_y = max(max_y, *y);
    }

    let row_width = (max_x + 2) as usize;  // 0..=max_x plus a newline char
    let row_count = (max_y + 1) as usize;  // 0..=max_y
    let mut buffer: Vec<u8> = Vec::with_capacity(0);
    buffer.resize(row_count * row_width, b'.');

    for y in 1..=max_y {
        buffer[(y as usize * row_width) - 1] = b'\n';
    }
    *buffer.last_mut().unwrap() = 0;

    for (x, y) in current_dots.iter() {
        buffer[(*y as usize) * row_width + (*x as usize)] = b'#';
    }

    String::from_utf8(buffer).unwrap()
}
