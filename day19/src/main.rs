#![feature(btree_drain_filter)]
#![feature(map_first_last)]
#![feature(map_try_insert)]

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    env, fs,
    ops::{Neg, Sub},
};

#[allow(unused_imports)]
use itertools::Itertools;

use nom::{
    bytes::complete::tag,
    character::complete::{char, digit1, line_ending},
    combinator::{map, map_res, opt, recognize},
    multi::many1,
    sequence::tuple,
    IResult,
};

use maplit::hashmap;
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let (remainder, input_data) = scanner_list(&content).unwrap();
    assert!(remainder.is_empty());

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

fn text_signed_int(input: &str) -> IResult<&str, i64> {
    map_res(recognize(tuple((opt(char('-')), digit1))), |value: &str| {
        value.parse()
    })(input)
}

fn point(input: &str) -> IResult<&str, Point> {
    map(
        tuple((
            text_signed_int,
            char(','),
            text_signed_int,
            char(','),
            text_signed_int,
            line_ending,
        )),
        |(x, _, y, _, z, _)| Point { x, y, z },
    )(input)
}

fn point_list(input: &str) -> IResult<&str, Vec<Point>> {
    many1(point)(input)
}

fn scanner(input: &str) -> IResult<&str, Scanner> {
    let (remaining, id) = map_res(
        tuple((tag("--- scanner "), digit1, tag(" ---"), line_ending)),
        |(_, scanner_id, _, _): (&str, &str, &str, &str)| scanner_id.parse::<usize>(),
    )(input)?;

    let (rest, (beacons, _)) = tuple((point_list, opt(line_ending)))(remaining)?;
    let beacon_locations: HashMap<Point, usize> = beacons
        .iter()
        .enumerate()
        .map(|(index, point)| (*point, index))
        .collect();

    Ok((
        rest,
        Scanner {
            id,
            beacons,
            beacon_locations,
        },
    ))
}

fn scanner_list(input: &str) -> IResult<&str, Vec<Scanner>> {
    many1(scanner)(input)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Point {
    x: i64,
    y: i64,
    z: i64,
}

impl Point {
    fn reorient(&self, orientation: &Orientation) -> Self {
        let distance_vector = [self.x, self.y, self.z];

        let (new_x, new_y, new_z) = orientation
            .axes
            .iter()
            .map(|row| row.iter().zip(distance_vector).map(|(a, b)| a * b).sum())
            .collect_tuple()
            .unwrap();

        Self {
            x: new_x,
            y: new_y,
            z: new_z,
        }
    }

    fn translate(&self, distance: &Distance) -> Self {
        Self {
            x: self.x + distance.x,
            y: self.y + distance.y,
            z: self.z + distance.z,
        }
    }

    fn from_origin(&self) -> Distance {
        Distance {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl Sub for Point {
    type Output = Distance;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Point {
    type Output = Point;

    fn neg(self) -> Self::Output {
        Self::Output {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Distance {
    x: i64,
    y: i64,
    z: i64,
}

impl Distance {
    fn axis_invariant(self) -> Self {
        let mut coords: [i64; 3] = [self.x.abs(), self.y.abs(), self.z.abs()];
        coords.sort_unstable();

        Self {
            x: coords[0],
            y: coords[1],
            z: coords[2],
        }
    }

    fn axis_ambiguous(&self) -> bool {
        let x = self.x.abs();
        let y = self.y.abs();
        let z = self.z.abs();

        // If two coordinates are equal, we can't distinguish between their axes.
        // If a coordinate is zero, we can't find that axis' orientation.
        x == y || y == z || x == z || x == 0 || y == 0 || z == 0
    }

    fn reorient(&self, orientation: &Orientation) -> Self {
        let distance_vector = [self.x, self.y, self.z];

        let (new_x, new_y, new_z) = orientation
            .axes
            .iter()
            .map(|row| row.iter().zip(distance_vector).map(|(a, b)| a * b).sum())
            .collect_tuple()
            .unwrap();

        Self {
            x: new_x,
            y: new_y,
            z: new_z,
        }
    }
}

#[derive(Clone, Debug)]
struct Scanner {
    #[allow(dead_code)]
    id: usize,
    beacons: Vec<Point>,
    beacon_locations: HashMap<Point, usize>,
}

#[derive(Clone, Debug)]
struct Beacon {
    #[allow(dead_code)]
    location: Point,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Orientation {
    axes: [[i64; 3]; 3],
}

impl Neg for Orientation {
    type Output = Orientation;

    fn neg(mut self) -> Self::Output {
        for row in self.axes.iter_mut() {
            for value in row.iter_mut() {
                *value = -*value;
            }
        }
        self
    }
}

impl Orientation {
    const MULTIPLIERS: [i64; 2] = [1, -1];

    fn all_possible_orientations() -> impl Iterator<Item = Orientation> {
        let multiplier_variations = Orientation::MULTIPLIERS
            .iter()
            .cartesian_product(Orientation::MULTIPLIERS.iter())
            .cartesian_product(Orientation::MULTIPLIERS.iter())
            .map(|((a, b), c)| (*a, *b, *c));

        (0..3usize)
            .permutations(3)
            .cartesian_product(multiplier_variations)
            .map(|(mappings, multipliers)| {
                let mut axes = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

                for (index, one_hot) in mappings.iter().enumerate() {
                    axes[index][*one_hot] = 1;
                }
                let multiplier_values = [multipliers.0, multipliers.1, multipliers.2];
                for (index, mult) in multiplier_values.iter().copied().enumerate() {
                    for value in axes.get_mut(index).unwrap().iter_mut() {
                        *value *= mult;
                    }
                }

                Orientation { axes }
            })
    }
}

#[derive(Clone, Debug)]
struct LocatedScanner {
    scanner: Scanner,
    location: Point,
    orientation: Orientation,
    beacons_in_global_coords: HashMap<Point, usize>,
}

#[derive(Clone, Debug)]
struct Signature {
    pairs: BTreeMap<Distance, Vec<(usize, usize)>>,
}

fn make_signature(points: &[Point]) -> Signature {
    let pairs: BTreeMap<Distance, Vec<(usize, usize)>> = points
        .iter()
        .enumerate()
        .tuple_combinations()
        .map(|((index_a, pt_a), (index_b, pt_b))| {
            let distance = (*pt_a - *pt_b).axis_invariant();
            (distance, (index_a, index_b))
        })
        .fold(Default::default(), |mut acc, (key, indices)| {
            acc.entry(key).or_default().push(indices);
            acc
        });

    Signature { pairs }
}

fn find_orientation(located_distance: &Distance, unlocated_distance: &Distance) -> Orientation {
    for possible_orientation in Orientation::all_possible_orientations() {
        if located_distance == &unlocated_distance.reorient(&possible_orientation) {
            return possible_orientation;
        }
    }

    unreachable!()
}

fn local_point_to_global_point(
    local_point: &Point,
    orientation: &Orientation,
    origin_in_global_coords: &Point,
) -> Point {
    origin_in_global_coords.translate(&local_point.reorient(orientation).from_origin())
}

fn check_beacon_match(
    located_scanner: &LocatedScanner,
    unlocated_scanner: &Scanner,
    orientation: &Orientation,
    location: &Point,
) -> bool {
    let min_matches_threshold = 12usize;

    unlocated_scanner
        .beacons
        .iter()
        .filter(|&beacon_point| {
            let translated_point = local_point_to_global_point(beacon_point, orientation, location);
            located_scanner
                .beacons_in_global_coords
                .contains_key(&translated_point)
        })
        .nth(min_matches_threshold - 1)
        .is_some()
}

fn get_orientation_and_location(
    located_point_a: &Point,
    located_point_b: &Point,
    unlocated_point_a: &Point,
    unlocated_point_b: &Point,
) -> (Orientation, Point) {
    let located_distance = *located_point_a - *located_point_b;
    let unlocated_distance = *unlocated_point_a - *unlocated_point_b;

    assert!(!located_distance.axis_ambiguous());
    assert!(!unlocated_distance.axis_ambiguous());

    let orientation = find_orientation(&located_distance, &unlocated_distance);
    let location =
        located_point_a.translate(&(-*unlocated_point_a).reorient(&orientation).from_origin());

    (orientation, location)
}

fn match_beacons(
    located_scanner: &LocatedScanner,
    located_sig: &Signature,
    unlocated_scanner: &Scanner,
    unlocated_sig: &Signature,
) -> Option<(Orientation, Point)> {
    for (distance, initial_pairs) in &located_sig.pairs {
        if distance.axis_ambiguous() {
            // Find a pairwise distance that we can use to orient the unlocated scanner's axes.
            continue;
        }

        for (initial_pair_a, initial_pair_b) in initial_pairs.iter().copied() {
            let located_point_a = local_point_to_global_point(
                &located_scanner.scanner.beacons[initial_pair_a],
                &located_scanner.orientation,
                &located_scanner.location,
            );
            let located_point_b = local_point_to_global_point(
                &located_scanner.scanner.beacons[initial_pair_b],
                &located_scanner.orientation,
                &located_scanner.location,
            );

            let maybe_matching_pairs = unlocated_sig.pairs.get(distance);
            for (matching_pair_a, matching_pair_b) in
                maybe_matching_pairs.into_iter().flatten().copied()
            {
                // try matching A -> A, B -> B
                let (orientation, location) = get_orientation_and_location(
                    &located_point_a,
                    &located_point_b,
                    &unlocated_scanner.beacons[matching_pair_a],
                    &unlocated_scanner.beacons[matching_pair_b],
                );
                if check_beacon_match(located_scanner, unlocated_scanner, &orientation, &location) {
                    return Some((orientation, location));
                }

                // try matching B -> A, A -> B
                let (orientation, location) = get_orientation_and_location(
                    &located_point_a,
                    &located_point_b,
                    &unlocated_scanner.beacons[matching_pair_b],
                    &unlocated_scanner.beacons[matching_pair_a],
                );

                if check_beacon_match(located_scanner, unlocated_scanner, &orientation, &location) {
                    return Some((orientation, location));
                }
            }
        }
    }

    None
}

fn match_signatures(
    located_scanner: &LocatedScanner,
    located_sig: &Signature,
    unlocated_scanner: &Scanner,
    unlocated_sig: &Signature,
) -> Option<LocatedScanner> {
    match_beacons(
        located_scanner,
        located_sig,
        unlocated_scanner,
        unlocated_sig,
    )
    .map(|(orientation, location)| LocatedScanner {
        scanner: unlocated_scanner.clone(),
        location,
        orientation,
        beacons_in_global_coords: unlocated_scanner
            .beacon_locations
            .iter()
            .map(|(pt, idx)| {
                (
                    local_point_to_global_point(pt, &orientation, &location),
                    *idx,
                )
            })
            .collect(),
    })
}

fn locate_everything(data: &[Scanner]) -> (HashMap<usize, LocatedScanner>, HashSet<Point>) {
    // Without loss of generality, the first scanner is at (0, 0, 0) and has correct axes.
    let origin = Point { x: 0, y: 0, z: 0 };
    let mut all_scanners = hashmap! {
        0 => LocatedScanner {
            scanner: data[0].clone(),
            location: origin,
            orientation: Orientation {
                axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            beacons_in_global_coords: data[0].beacon_locations.clone(),
        }
    };

    let signatures = data
        .iter()
        .map(|s| make_signature(&s.beacons))
        .collect_vec();

    let mut unlocated_scanners: BTreeSet<usize> = (1..data.len()).collect();
    while !unlocated_scanners.is_empty() {
        let located_in_this_iteration = unlocated_scanners
            .drain_filter(|&id| {
                let unlocated_signature = &signatures[id];

                for (&scanner_id, located_scanner) in &all_scanners {
                    let located_signature = &signatures[scanner_id];
                    if let Some(located) = match_signatures(
                        located_scanner,
                        located_signature,
                        &data[id],
                        unlocated_signature,
                    ) {
                        let inserted = all_scanners.insert(id, located);
                        assert!(inserted.is_none());
                        return true;
                    }
                }

                false
            })
            .collect_vec();

        assert!(!located_in_this_iteration.is_empty());
    }

    let all_beacons: HashSet<Point> = all_scanners
        .values()
        .flat_map(|located_scanner| {
            let location = located_scanner.location;
            let orientation = located_scanner.orientation;
            located_scanner
                .scanner
                .beacons
                .iter()
                .map(move |pt| local_point_to_global_point(pt, &orientation, &location))
        })
        .collect();

    (all_scanners, all_beacons)
}

#[allow(unused_variables)]
fn solve_part1(data: &[Scanner]) -> usize {
    let (_, beacons) = locate_everything(data);
    beacons.len()
}

#[allow(unused_variables)]
fn solve_part2(data: &[Scanner]) -> i64 {
    let (scanners, _) = locate_everything(data);

    scanners
        .values()
        .tuple_combinations()
        .map(|(a, b)| {
            let distance = a.location - b.location;
            distance.x.abs() + distance.y.abs() + distance.z.abs()
        })
        .max()
        .unwrap()
}
