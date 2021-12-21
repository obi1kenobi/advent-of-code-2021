use std::{env, fs, ops::Add, fmt::Display, cmp::max};

#[allow(unused_imports)]
use itertools::Itertools;

use nom::{IResult, character::complete::{char, digit1}, combinator::{complete, map_res}, sequence::{delimited, separated_pair}};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();

    let input_data: Vec<SnailfishNumber> = content
        .trim_end()
        .split('\n')
        .map(SnailfishNumber::from)
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
        "normalize" => {
            let start = input_data[0].clone().normalize();
            let result = input_data[1..].iter().fold(start, |acc, elem| {
                (acc + elem.clone()).normalize()
            });
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

#[derive(Clone, Debug)]
enum SnailfishNumber {
    Literal(u64),
    Pair(Box<(SnailfishNumber, SnailfishNumber)>),
}

fn snailfish_number(input: &str) -> IResult<&str, SnailfishNumber> {
    if let Ok((remainder, parsed)) = digit1::<&str, nom::error::Error<&str>>(input) {
        Ok((remainder, SnailfishNumber::Literal(parsed.parse().unwrap())))
    } else {
        pair_number(input)
    }
}

fn pair_number(input: &str) -> IResult<&str, SnailfishNumber> {
    map_res(delimited(
        char('['),
        separated_pair(snailfish_number, char(','), snailfish_number),
        char(']'),
    ), |pair| -> Result<SnailfishNumber, nom::error::Error<&str>> {
        Ok(SnailfishNumber::Pair(Box::new(pair)))
    })(input)
}

impl From<&str> for SnailfishNumber {
    fn from(value: &str) -> SnailfishNumber {
        let (remainder, parsed) = complete(snailfish_number)(value).unwrap();
        assert!(remainder.is_empty());
        parsed
    }
}

impl Display for SnailfishNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnailfishNumber::Literal(l) => l.fmt(f),
            SnailfishNumber::Pair(pair) => {
                write!(f, "[{},{}]", pair.0, pair.1)
            },
        }
    }
}

impl SnailfishNumber {
    fn leftmost_literal(&mut self) -> &mut SnailfishNumber {
        match self {
            SnailfishNumber::Literal(_) => self,
            SnailfishNumber::Pair(pair) => {
                pair.0.leftmost_literal()
            }
        }
    }

    fn rightmost_literal(&mut self) -> &mut SnailfishNumber {
        match self {
            SnailfishNumber::Literal(_) => self,
            SnailfishNumber::Pair(pair) => {
                pair.1.rightmost_literal()
            }
        }
    }

    fn magnitude(&self) -> u64 {
        match self {
            SnailfishNumber::Literal(l) => *l,
            SnailfishNumber::Pair(b) => {
                3 * b.0.magnitude() + 2 * b.1.magnitude()
            }
        }
    }

    fn literal_value(&self) -> u64 {
        match self {
            SnailfishNumber::Literal(l) => *l,
            SnailfishNumber::Pair(_) => unreachable!("{:?}", self),
        }
    }

    fn apply_explode(&mut self) -> bool {
        self.check_explode(0).is_some()
    }

    fn check_explode(&mut self, depth: usize) -> Option<(Option<u64>, Option<u64>, bool)> {
        assert!(depth <= 4);
        match self {
            SnailfishNumber::Literal(_) => None,
            SnailfishNumber::Pair(pair) => {
                if depth == 4 {
                    // BOOM!
                    let left_value = pair.0.literal_value();
                    let right_value = pair.1.literal_value();

                    Some((Some(left_value), Some(right_value), true))
                } else {
                    let next_depth = depth + 1;
                    if let Some((left, right, child_exploded)) = pair.0.check_explode(next_depth) {
                        if let Some(increment) = right {
                            let right_neighbor = pair.1.leftmost_literal();
                            *right_neighbor = SnailfishNumber::Literal(right_neighbor.literal_value() + increment);
                        }
                        if child_exploded {
                            *self = SnailfishNumber::Pair(Box::new((SnailfishNumber::Literal(0), pair.1.clone())));
                        }
                        Some((left, None, false))
                    } else if let Some((left, right, child_exploded)) = pair.1.check_explode(next_depth) {
                        if let Some(increment) = left {
                            let left_neighbor = pair.0.rightmost_literal();
                            *left_neighbor = SnailfishNumber::Literal(left_neighbor.literal_value() + increment);
                        }
                        if child_exploded {
                            *self = SnailfishNumber::Pair(Box::new((pair.0.clone(), SnailfishNumber::Literal(0))));
                        }
                        Some((None, right, false))
                    } else {
                        None
                    }
                }
            }
        }
    }

    fn apply_split(&mut self) -> bool {
        match self {
            SnailfishNumber::Literal(l) => {
                if *l >= 10 {
                    let left = SnailfishNumber::Literal(*l / 2);
                    let right = SnailfishNumber::Literal((*l + 1) / 2);
                    *self = SnailfishNumber::Pair(Box::new((left, right)));
                    true
                } else {
                    false
                }
            }
            SnailfishNumber::Pair(pair) => {
                pair.0.apply_split() || pair.1.apply_split()
            }
        }
    }

    fn normalize(mut self) -> Self {
        loop {
            if !self.apply_explode() && !self.apply_split() {
                break self;
            }
        }
    }
}

impl Add for SnailfishNumber {
    type Output = SnailfishNumber;

    fn add(self, rhs: Self) -> Self::Output {
        SnailfishNumber::Pair(Box::new((self, rhs)))
    }
}

#[allow(unused_variables)]
fn solve_part1(data: &[SnailfishNumber]) -> u64 {
    let start = data[0].clone().normalize();

    let final_number = data[1..].iter().fold(start, |acc, elem| {
        (acc + elem.clone()).normalize()
    });

    final_number.magnitude()
}

#[allow(unused_variables)]
fn solve_part2(data: &[SnailfishNumber]) -> u64 {
    data.iter().tuple_combinations().map(|(left, right)| {
        let left_mag = (left.clone().normalize() + right.clone()).normalize().magnitude();
        let right_mag = (right.clone().normalize() + left.clone()).normalize().magnitude();

        max(left_mag, right_mag)
    }).max().unwrap()
}
