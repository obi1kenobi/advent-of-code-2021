use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs, fmt::Display,
};

#[allow(unused_imports)]
use itertools::Itertools;
use itertools::chain;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    pub x: u64,
    pub y: u64,
}

impl Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.x, self.y)
    }
}

impl From<&str> for Point {
    fn from(value: &str) -> Self {
        let (x, y) = value
            .trim()
            .split_once(',')
            .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
            .unwrap();

        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Line {
    // start is tuple-wise earlier than end
    pub start: Point,
    pub end: Point,
}

impl Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.start, self.end)
    }
}

impl Line {
    fn new(a: Point, b: Point) -> Self {
        if a <= b {
            Self { start: a, end: b }
        } else {
            Self { start: b, end: a }
        }
    }

    fn is_x_aligned(&self) -> bool {
        self.start.y == self.end.y
    }

    fn is_y_aligned(&self) -> bool {
        self.start.x == self.end.x
    }
}

impl From<&str> for Line {
    fn from(value: &str) -> Self {
        let (first, second) = value.split_once("->").unwrap();

        Self::new(first.into(), second.into())
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

    let input_data: Vec<Line> = content.trim_end().split('\n').map(Line::from).collect();

    match part {
        "1" => {
            let result = solve_part1(&input_data);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(&input_data);
            println!("{}", result);
        }
        "diff_1" => {
            let result = solve_part1(&input_data);
            let brute_force = brute_force_part1(&input_data);
            println!("{} vs {}", result, brute_force);
        }
        "minify_1" => {
            minify_part1(&input_data);
        }
        _ => unreachable!("{}", part),
    }
}

fn minify_part1(data: &[Line]) {
    let axis_aligned_lines: Vec<_> = data
        .iter()
        .filter(|line| line.is_x_aligned() || line.is_y_aligned())
        .cloned()
        .collect();

    let mut start_lines = axis_aligned_lines.as_slice();
    let mut next_lines = &start_lines[..start_lines.len() / 2];
    while solve_part1(next_lines) != brute_force_part1(next_lines) {
        eprintln!("Successfully halved, len {}", next_lines.len());

        start_lines = next_lines;
        next_lines = &start_lines[..start_lines.len() / 2];
    }

    eprintln!("Moving to single suffix truncation...");

    next_lines = &start_lines[..start_lines.len() - 1];
    while solve_part1(next_lines) != brute_force_part1(next_lines) {
        eprintln!("Successfully truncated suffix, len {}", next_lines.len());

        start_lines = next_lines;
        next_lines = &start_lines[..start_lines.len() - 1];
    }

    eprintln!("Moving to single prefix truncation...");

    next_lines = &start_lines[1..];
    while solve_part1(next_lines) != brute_force_part1(next_lines) {
        eprintln!("Successfully truncated prefix, len {}", next_lines.len());

        start_lines = next_lines;
        next_lines = &start_lines[1..];
    }

    let known_problematic_lines = vec![*start_lines.first().unwrap(), *start_lines.last().unwrap()];
    if solve_part1(&known_problematic_lines) != brute_force_part1(&known_problematic_lines) {
        eprintln!("Got lucky: found 2-line counterexample!");
        start_lines = known_problematic_lines.as_slice();
    }

    for line in start_lines {
        println!("{}", line);
    }
}

fn solve_part1(data: &[Line]) -> usize {
    let axis_aligned_lines: Vec<_> = data
        .iter()
        .filter(|line| line.is_x_aligned() || line.is_y_aligned())
        .collect();

    let mut all_points: Vec<(Point, bool, usize)> = axis_aligned_lines
        .iter()
        .copied()
        .enumerate()
        .flat_map(|(index, line)| {
            assert!(line.start.y <= line.end.y);

            [(line.start, false, index), (line.end, true, index)]
        })
        .collect();
    all_points.sort_unstable();

    let mut overlaps = 0usize;
    let mut active_points: BTreeMap<u64, usize> = Default::default();
    let mut last_x_coord: Option<u64> = None;

    // Perform a line-sweep along the X-axis, counting overlapping points along the way.
    for (x_coord, point_group) in all_points
        .iter()
        .copied()
        .group_by(|(pt, _, _)| pt.x)
        .into_iter()
    {
        if let Some(last_x) = last_x_coord {
            let delta_x = (x_coord - last_x - 1) as usize;

            // X-parallel lines that have been overlapping for the entire swept segment.
            overlaps += delta_x * active_points
                .iter()
                .filter(|(_, value)| **value > 1)
                .count();
        };
        last_x_coord = Some(x_coord);

        let points: Vec<(Point, bool, usize)> = point_group.collect();

        // For all the X-parallel lines (i.e. perpendicular to our sweep) starting at this point,
        // we add their intersections into the active_points.
        for (point, is_end, index) in points.iter() {
            let line = axis_aligned_lines[*index];
            if line.is_x_aligned() && !is_end {
                active_points
                    .entry(point.y)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        let mut active_points_iter = active_points.iter().peekable();
        let mut y_line_points_iter = points.iter()
            .filter(|(_, _, index)| {
                let line = axis_aligned_lines[*index];

                // line.start == line.end kinds of lines are both X and Y-aligned,
                // so they are accounted for in the X-aligned group.
                line.start != line.end && line.is_y_aligned()
            })
            .peekable();

        if let Some((first_point, _, _)) = y_line_points_iter.peek() {
            let mut last_y = first_point.y;

            // Count X-parallel line overlaps that happen before the first Y-parallel line.
            while let Some((&y, &count)) = active_points_iter.peek() {
                if y < last_y {
                    active_points_iter.next().unwrap();

                    if count > 1 {
                        overlaps += 1;
                    }
                } else {
                    break;
                }
            }

            let y_points_of_interest: BTreeSet<u64> = chain!(
                y_line_points_iter.clone().map(|(pt, _, _)| pt.y),
                active_points_iter.clone().map(|(y_coord, _)| *y_coord)).collect();

            let mut overlapping_ranges = 0usize;
            for y_coord in y_points_of_interest {
                assert!(y_coord >= last_y);

                let mut peeked_y_line = y_line_points_iter.peek();
                let mut peeked_active_point = active_points_iter.peek();

                // (a) If we've been in an overlapping range since the last point of interest,
                //     mark all the skipped Y positions as overlapping.
                // (b) Next, try to add any Y-lines that start at this coordinate.
                // (c) Then, process active points.
                // (d) Finally, process any Y-lines that end at this coordinate.
                // -----
                // (a) If we've been in an overlapping range since the last point of interest,
                //     mark all the skipped Y positions as overlapping.
                if overlapping_ranges > 1 && y_coord > last_y {
                    let skipped_y_positions = y_coord - last_y - 1;
                    overlaps += skipped_y_positions as usize;
                }
                last_y = y_coord;

                // (b) Next, try to add any Y-lines that start at this coordinate.
                while let Some((y_pt, is_end, _)) = peeked_y_line {
                    assert!(y_pt.y >= y_coord);

                    if y_pt.y > y_coord || *is_end {
                        break;
                    }
                    assert_eq!(y_pt.y, y_coord);
                    assert!(!is_end);

                    overlapping_ranges += 1;

                    y_line_points_iter.next().unwrap();
                    peeked_y_line = y_line_points_iter.peek();
                }

                let mut overlap_found = overlapping_ranges > 1;

                // (c) Then, process active points.
                while let Some((active_y, count)) = peeked_active_point {
                    assert!(**active_y >= y_coord);
                    if **active_y > y_coord {
                        break;
                    }

                    // If there's a range overlapping this active point,
                    // or if there's more than one active point at this coordinate,
                    // we've found an overlap.
                    if overlapping_ranges == 1 || **count > 1 {
                        overlap_found = true;
                    }

                    active_points_iter.next().unwrap();
                    peeked_active_point = active_points_iter.peek();
                }

                // (d) Finally, process any Y-lines that end at this coordinate.
                while let Some((y_pt, is_end, _)) = peeked_y_line {
                    assert!(y_pt.y >= y_coord);

                    if y_pt.y > y_coord {
                        break;
                    }
                    assert_eq!(y_pt.y, y_coord);
                    assert!(is_end);

                    overlapping_ranges -= 1;

                    y_line_points_iter.next().unwrap();
                    peeked_y_line = y_line_points_iter.peek();
                }

                if overlap_found {
                    overlaps += 1;
                }
            }
        }

        // Add any remaining X-parallel line overlaps at this X coordinate.
        overlaps += active_points_iter.filter(|(_, count)| **count > 1).count();

        // For all the X-parallel lines (i.e. perpendicular to our sweep) ending at this point,
        // we remove their intersections into the active_points.
        for (point, is_end, index) in points.iter() {
            let line = axis_aligned_lines[*index];
            if line.is_x_aligned() && *is_end {
                let count = *active_points.get(&point.y).unwrap();
                if count == 1 {
                    active_points.remove(&point.y);
                } else {
                    active_points.insert(point.y, count - 1);
                }
            }
        }
    }

    overlaps
}

fn brute_force_part1(data: &[Line]) -> usize {
    let axis_aligned_lines: Vec<_> = data
        .iter()
        .filter(|line| line.is_x_aligned() || line.is_y_aligned())
        .collect();

    let min_x = axis_aligned_lines.iter().flat_map(|line| {
        [
            line.start.x,
            line.end.x,
        ]
    }).min().unwrap();
    let max_x = axis_aligned_lines.iter().flat_map(|line| {
        [
            line.start.x,
            line.end.x,
        ]
    }).max().unwrap();

    let min_y = axis_aligned_lines.iter().flat_map(|line| {
        [
            line.start.y,
            line.end.y,
        ]
    }).min().unwrap();
    let max_y = axis_aligned_lines.iter().flat_map(|line| {
        [
            line.start.y,
            line.end.y,
        ]
    }).max().unwrap();

    let mut count = 0usize;
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            let second_intersection = axis_aligned_lines.iter().filter(|line| {
                (line.is_y_aligned() && line.start.x == x && (
                    line.start.y <= y && y <= line.end.y
                )) ||
                (line.is_x_aligned() && line.start.y == y && (
                    line.start.x <= x && x <= line.end.x
                ))
            }).nth(1);
            if second_intersection.is_some() {
                count += 1;
            }
        }
    }

    count
}

fn solve_part2(data: &[Line]) -> usize {
    todo!()
}
