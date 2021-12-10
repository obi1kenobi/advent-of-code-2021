use std::{collections::{HashMap}, env, fs};

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

    let input_data: Vec<Vec<i64>> = content
        .trim_end()
        .split('\n')
        .map(|x| x.chars().map(|c| c.to_digit(10).unwrap() as i64).collect())
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

const NEIGHBOR_OFFSETS: [(i64, i64); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];

fn get_height(data: &[Vec<i64>], x: i64, y: i64) -> Option<i64> {
    let x_limit = data.len() as i64;
    let y_limit = data[0].len() as i64;

    if x >= 0 && x < x_limit && y >= 0 && y < y_limit {
        Some(data[x as usize][y as usize])
    } else {
        None
    }
}

fn neighbors(data: &[Vec<i64>], x: i64, y: i64) -> impl Iterator<Item = (i64, i64)> + '_ {
    assert!(get_height(data, x, y).is_some());

    NEIGHBOR_OFFSETS
        .iter()
        .copied()
        .filter_map(move |(dx, dy)| {
            let new_x = x + dx;
            let new_y = y + dy;

            get_height(data, new_x, new_y).map(|_| (new_x, new_y))
        })
}

fn find_lowest_points(data: &[Vec<i64>]) -> impl Iterator<Item = (i64, i64)> + '_ {
    data.iter().enumerate().flat_map(move |(i, row)| {
        row.iter().enumerate().filter_map(move |(j, height)| {
            let x = i as i64;
            let y = j as i64;

            let lower_neighbor =
                neighbors(data, x, y).find(|(nx, ny)| *height >= data[*nx as usize][*ny as usize]);

            if lower_neighbor.is_none() {
                Some((x, y))
            } else {
                None
            }
        })
    })
}

fn solve_part1(data: &[Vec<i64>]) -> i64 {
    find_lowest_points(data)
        .map(|(x, y)| 1 + data[x as usize][y as usize])
        .sum()
}

fn flood_fill(
    data: &[Vec<i64>],
    belongs_to: &mut HashMap<(i64, i64), (i64, i64)>,
    point: (i64, i64),
    basin: (i64, i64),
) {
    let height = get_height(data, point.0, point.1).unwrap();
    for neighbor in neighbors(data, point.0, point.1) {
        let neighbor_height = get_height(data, neighbor.0, neighbor.1).unwrap();
        if neighbor_height == 9 {
            // height 9 points do not belong to any basin
            continue;
        }

        if neighbor_height > height {
            // all points drain to one basin, and this neighbor *could* drain to this basin
            // so it must be the only basin it will drain to
            let prior_basin = belongs_to.insert(neighbor, basin);
            if let Some(prior) = prior_basin {
                assert_eq!(prior, basin);
            } else {
                flood_fill(data, belongs_to, neighbor, basin);
            }
        }
    }
}

fn solve_part2(data: &[Vec<i64>]) -> usize {
    let mut belongs_to: HashMap<(i64, i64), (i64, i64)> = Default::default();

    for basin in find_lowest_points(data) {
        let prior_basin = belongs_to.insert(basin, basin);
        assert!(prior_basin.is_none());

        flood_fill(data, &mut belongs_to, basin, basin);
    }

    let mut basin_sizes: HashMap<(i64, i64), usize> = Default::default();
    for (_, basin) in belongs_to {
        basin_sizes.entry(basin).and_modify(|x| *x += 1).or_insert(1);
    }

    let mut all_basin_sizes: Vec<_>  = basin_sizes.values().collect();
    let basins_count = all_basin_sizes.len();
    all_basin_sizes.partition_by_kth(basins_count - 3);

    all_basin_sizes[(basins_count - 3)..].iter().copied().copied().product()
}
