#![feature(map_first_last)]
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    ops::RangeInclusive, fmt::Display,
};

#[allow(unused_imports)]
use itertools::Itertools;

fn parse_range(range: &str) -> (i64, i64) {
    let (low, high) = range.split_once("..").unwrap();
    (low.parse().unwrap(), high.parse().unwrap())
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

    let input_data: Vec<RebootStep> = content
        .trim_end()
        .split('\n')
        .map(|x| {
            let (direction, all_coords) = x.split_once(' ').unwrap();
            let switch_on = match direction {
                "on" => true,
                "off" => false,
                _ => unreachable!("{}", direction),
            };

            let (x_range, y_range, z_range) = {
                let (x_comp, (y_comp, z_comp)) = all_coords
                    .split_once(',')
                    .map(|(left, right)| (left, right.split_once(',').unwrap()))
                    .unwrap();
                (
                    parse_range(x_comp.split_once('=').unwrap().1),
                    parse_range(y_comp.split_once('=').unwrap().1),
                    parse_range(z_comp.split_once('=').unwrap().1),
                )
            };

            RebootStep {
                switch_on,
                range: [x_range, y_range, z_range],
            }
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
        "minify" => {
            minify(&input_data);
        }
        _ => unreachable!("{}", part),
    }
}

#[derive(Clone, Debug)]
struct RebootStep {
    switch_on: bool,

    // RangeInclusive has private internal representation including endpoints
    // so it's not suitable to be used here. Instead, we use tuples as inclusive ranges.
    range: [(i64, i64); 3],
}

impl Display for RebootStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = if self.switch_on { "on" } else { "off" };
        write!(
            f, "{} x={}..{},y={}..{},z={}..{}", state,
            self.range[0].0, self.range[0].1,
            self.range[1].0, self.range[1].1,
            self.range[2].0, self.range[2].1,
        )
    }
}

struct RebootStepPrinter<'a>(&'a [RebootStep]);
impl<'a> Display for RebootStepPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for step in self.0.iter() {
            writeln!(f, "{}", step)?;
        }
        Ok(())
    }
}

fn minify(data: &[RebootStep]) {
    let part1 = solve_part1(data);
    let part2 = solve_part2(data);

    assert_ne!(part1, part2);

    let mut minified = data;
    loop {
        let next_minified = &minified[1..];
        if next_minified.is_empty() {
            break;
        }

        let part1 = solve_part1(next_minified);
        let part2 = solve_part2(next_minified);

        if part1 != part2 {
            minified = next_minified;
            println!("eliminated prefix: {}", minified.len());
        } else {
            break;
        }
    }

    loop {
        let next_minified = &minified[..(minified.len() - 1)];
        if next_minified.is_empty() {
            break;
        }

        let part1 = solve_part1(next_minified);
        let part2 = solve_part2(next_minified);

        if part1 != part2 {
            minified = next_minified;
            println!("eliminated suffix: {}", minified.len());
        } else {
            break;
        }
    }

    println!("{}", RebootStepPrinter(minified));
}

fn to_range(range: (i64, i64)) -> RangeInclusive<i64> {
    range.0..=range.1
}

fn solve_part1(data: &[RebootStep]) -> usize {
    let target_range = -50i64..=50;

    target_range
        .clone()
        .cartesian_product(target_range.clone())
        .cartesian_product(target_range)
        .filter(|((x, y), z)| {
            for step in data.iter().rev() {
                let (x_range, y_range, z_range) = step
                    .range
                    .iter()
                    .copied()
                    .map(to_range)
                    .collect_tuple()
                    .unwrap();

                if x_range.contains(x) && y_range.contains(y) && z_range.contains(z) {
                    // this is the most recent step that affected the point (x, y, z)
                    return step.switch_on;
                }
            }

            // this point wasn't affected by any of the reboot steps, it defaults to off
            false
        })
        .count()
}

fn solve_part2(data: &[RebootStep]) -> usize {
    let notable_coordinates: Vec<BTreeSet<i64>> = (0..3).map(|idx| {
        data
            .iter()
            .flat_map(|step| [step.range[idx].0, step.range[idx].1 + 1])
            .collect()
    }).collect_vec();

    let splits = notable_coordinates.iter().map(|coords| coords.iter().copied().collect_vec()).collect_vec();
    let offsets: Vec<BTreeMap<i64, usize>> = splits.iter().map(|axis_splits| axis_splits.iter().enumerate().map(|(idx, val)| (*val, idx))
        .collect()).collect();

    // The input has a bit over ~800 cell divisions per axis, for a total of ~500M cells.
    // We can store a boolean per cell and avoid needing to implement a more complex data structure.
    let mut cell_is_on =
        vec![
            vec![vec![false; notable_coordinates[2].len()]; notable_coordinates[1].len()];
            notable_coordinates[0].len()
        ];

    // We could avoid pre-computing cell volumes and only compute them on demand,
    // but that's more work. This is easy and will fit in ~4GB of memory without issues.
    let mut cell_volumes =
        vec![
            vec![vec![0usize; notable_coordinates[2].len()]; notable_coordinates[1].len()];
            notable_coordinates[0].len()
        ];
    for (x_idx, (x_start, x_end)) in splits[0].iter().tuple_windows().enumerate() {
        for (y_idx, (y_start, y_end)) in splits[1].iter().tuple_windows().enumerate() {
            for (z_idx, (z_start, z_end)) in splits[2].iter().tuple_windows().enumerate() {
                let x_width = (*x_end - *x_start) as usize;
                let y_width = (*y_end - *y_start) as usize;
                let z_width = (*z_end - *z_start) as usize;
                cell_volumes[x_idx + 1][y_idx + 1][z_idx + 1] = x_width * y_width * z_width;
            }
        }
    }

    for step in data {
        let (
            (x_start_cell, x_end_cell),
            (y_start_cell, y_end_cell),
            (z_start_cell, z_end_cell),
        ) = step.range.iter().zip(offsets.iter()).map(|((start, end_incl), offset)| {
            let end = end_incl + 1;
            (offset[start], offset[&end])
        }).collect_tuple().unwrap();

        #[allow(clippy::needless_range_loop)]
        for x in x_start_cell..x_end_cell {
            for y in y_start_cell..y_end_cell {
                for z in z_start_cell..z_end_cell {
                    cell_is_on[x][y][z] = step.switch_on;
                }
            }
        }
    }

    let cell_volumes_ref = &cell_volumes;
    cell_is_on
        .iter()
        .enumerate()
        .flat_map(move |(x_idx, y)| {
            y.iter().enumerate().map::<usize, _>(move |(y_idx, z)| {
                z.iter()
                    .enumerate()
                    .filter_map(|(z_idx, is_on)| {
                        if *is_on {
                            Some(cell_volumes_ref[x_idx + 1][y_idx + 1][z_idx + 1])
                        } else {
                            None
                        }
                    })
                    .sum()
            })
        })
        .sum()
}
