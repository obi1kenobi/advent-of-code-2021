#![feature(map_first_last)]
use std::{
    cmp::{min, Ordering},
    collections::BTreeSet,
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

    let input_data: Vec<Vec<char>> = content
        .trim_end()
        .split('\n')
        .map(|x| x.chars().collect_vec())
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

const COLOR_COUNT: usize = 4;
const STEP_COSTS: [u64; 4] = [1, 10, 100, 1000];

#[derive(Debug, Clone)]
struct Burrow {
    hallway_spots: BTreeSet<(usize, usize)>,
    doorsteps: BTreeSet<(usize, usize)>,
    hallway_x: usize,
    room_ys: Vec<usize>, // at (hallway_x + 1) and (hallway_x + 2) X coordinates
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Migration<const N: usize> {
    cost: u64,
    positions: [[(usize, usize, LocationType); N]; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LocationType {
    StartingRoom,
    Hallway,
    FinalRoom,
}

impl<const N: usize> Migration<N> {
    fn fix_critters_at_final_positions(&mut self, burrow: &Burrow) {
        for color in 0..COLOR_COUNT {
            for num in 0..N {
                let (crit_x, crit_y, _) = self.positions[color][num];

                let correct_room_for_color = crit_y == burrow.room_ys[color];
                let lower_critters_matching_color = ((crit_x + 1)..=(burrow.hallway_x + N))
                    .all(|lower_x| self.is_color_at_position(color, lower_x, crit_y));
                if correct_room_for_color && lower_critters_matching_color {
                    // This critter is the correct color and all the lower critters in the room
                    // are the correct color as well. This is their final room.
                    self.positions[color][num].2 = LocationType::FinalRoom;
                }
            }
        }
    }

    fn attempt_move(&self, x: usize, y: usize) -> bool {
        for color in 0..COLOR_COUNT {
            for num in 0..N {
                let (crit_x, crit_y, _) = self.positions[color][num];
                if crit_x == x && crit_y == y {
                    return false;
                }
            }
        }
        true
    }

    fn is_color_at_position(&self, color: usize, x: usize, y: usize) -> bool {
        for num in 0..N {
            let (crit_x, crit_y, _) = self.positions[color][num];
            if crit_x == x && crit_y == y {
                return true;
            }
        }
        false
    }

    fn move_to_hallway(
        &self,
        burrow: &Burrow,
        critter_id: (usize, usize),
        hallway_y: usize,
    ) -> Option<Migration<N>> {
        let (color, num) = critter_id;
        let (start_x, start_y, location) = self.positions[color][num];

        // Can only move to the hallway if beginning in the starting room.
        // No hallway -> hallway or final room -> hallway moves are allowed.
        if location != LocationType::StartingRoom {
            return None;
        }

        // Cannot move to a room's doorstep.
        if burrow.doorsteps.contains(&(burrow.hallway_x, hallway_y)) {
            return None;
        }

        // Perform the move.
        let mut x = start_x;
        let mut y = start_y;
        let mut steps = 0u64;

        // First, move up and out of the room.
        assert!(x > burrow.hallway_x);
        while x > burrow.hallway_x {
            x -= 1;
            steps += 1;

            if !self.attempt_move(x, y) {
                // Blocked!
                return None;
            }
        }

        // Next, move across the hallway to the desired Y position.
        let delta_y = match hallway_y.cmp(&start_y) {
            Ordering::Less => -1,
            Ordering::Equal => unreachable!(),
            Ordering::Greater => 1,
        };
        while y != hallway_y {
            y = ((y as i64) + delta_y) as usize;
            steps += 1;

            if !self.attempt_move(x, y) {
                // Blocked!
                return None;
            }
        }

        let mut next = self.clone();
        next.cost += steps * STEP_COSTS[color];
        next.positions[color][num] = (x, y, LocationType::Hallway);
        Some(next)
    }

    fn move_to_final_room(
        &self,
        burrow: &Burrow,
        critter_id: (usize, usize),
    ) -> Option<Migration<N>> {
        let (color, num) = critter_id;
        let (start_x, start_y, location) = self.positions[color][num];

        let room_y = burrow.room_ys[color];

        // Can only move to the final room if not already there.
        if location == LocationType::FinalRoom {
            return None;
        }

        // Figure out which direction we need to go in the Y axis.
        let delta_y = match room_y.cmp(&start_y) {
            Ordering::Less => -1,
            Ordering::Equal => {
                // We're already in the right room, but it's not our final position --
                // we'll need to move out of the way first. That will have to be a hallway move.
                return None;
            }
            Ordering::Greater => 1,
        };

        // Perform the move.
        let mut x = start_x;
        let mut y = start_y;
        let mut steps = 0u64;

        if location == LocationType::StartingRoom {
            // First, move up and out of the room.
            assert!(x > burrow.hallway_x);
            while x > burrow.hallway_x {
                x -= 1;
                steps += 1;

                if !self.attempt_move(x, y) {
                    // Blocked!
                    return None;
                }
            }
        }

        // Move across the hallway to the desired Y position.
        while y != room_y {
            y = ((y as i64) + delta_y) as usize;
            steps += 1;

            if !self.attempt_move(x, y) {
                // Blocked!
                return None;
            }
        }

        // Then, move down into the room as far down as possible.
        // If we get blocked before getting to the bottom, make sure all lower spots are taken
        // by critters of the matching color, or else this is an illegal move.
        let mut got_blocked = false;
        for depth in 1..=N {
            if !got_blocked {
                if !self.attempt_move(x + 1, y) {
                    // Blocked!
                    got_blocked = true;
                } else {
                    x += 1;
                    steps += 1;
                }
            }

            if got_blocked && !self.is_color_at_position(color, burrow.hallway_x + depth, y) {
                // Found a critter of a different color below ourselves. Illegal move,
                // because they have to get out first before we move in.
                return None;
            }
        }

        let mut next = self.clone();
        next.cost += steps * STEP_COSTS[color];
        next.positions[color][num] = (x, y, LocationType::FinalRoom);
        Some(next)
    }
}

fn is_solved<const N: usize>(burrow: &Burrow, migration: &Migration<N>) -> Option<u64> {
    for (room_y, critters) in burrow.room_ys.iter().zip(migration.positions.iter()) {
        for &(x, y, location_type) in critters {
            if y != *room_y
                || !((burrow.hallway_x + 1)..=(burrow.hallway_x + N)).contains(&x)
                || location_type != LocationType::FinalRoom
            {
                return None;
            }
        }
    }

    Some(migration.cost)
}

fn dijkstra_search<const N: usize>(burrow: Burrow, start: Migration<N>) -> u64 {
    // Use a BTreeSet instead of a BinaryHeap to get move deduplication for free.
    let mut heap: BTreeSet<Migration<N>> = Default::default();
    heap.insert(start);

    while let Some(migration) = heap.pop_first() {
        if let Some(answer) = is_solved(&burrow, &migration) {
            return answer;
        }

        for critter_color in 0..COLOR_COUNT {
            for critter_num in 0..N {
                let critter_id = (critter_color, critter_num);
                for &(_, hallway_y) in &burrow.hallway_spots {
                    if let Some(next) = migration.move_to_hallway(&burrow, critter_id, hallway_y) {
                        heap.insert(next);
                    }
                }

                if let Some(next) = migration.move_to_final_room(&burrow, critter_id) {
                    heap.insert(next);
                }
            }
        }
    }

    unreachable!("no solution found")
}

fn solve<const N: usize>(data: &[Vec<char>]) -> u64 {
    let hallway_spots: BTreeSet<(usize, usize)> = data
        .iter()
        .enumerate()
        .flat_map(|(x, row)| {
            row.iter().enumerate().filter_map(
                move |(y, c)| {
                    if *c == '.' {
                        Some((x, y))
                    } else {
                        None
                    }
                },
            )
        })
        .collect();

    // Make sure the hallway is horizontal on the map.
    let hallway_x = hallway_spots.iter().next().unwrap().0;
    for &(x, _) in &hallway_spots {
        assert_eq!(hallway_x, x);
    }

    assert_eq!(COLOR_COUNT, 4);
    let mut initial_positions = vec![vec![]; COLOR_COUNT];
    for (x, row) in data.iter().enumerate() {
        for (y, c) in row.iter().enumerate() {
            match c {
                'A' => initial_positions[0].push((x, y)),
                'B' => initial_positions[1].push((x, y)),
                'C' => initial_positions[2].push((x, y)),
                'D' => initial_positions[3].push((x, y)),
                _ => {}
            }
        }
    }

    // Make sure the room positions are the correct number of spots below the hallway,
    // and find their Y coordinates.
    let mut room_y_coordinates: BTreeSet<usize> = Default::default();
    for &(x, y) in initial_positions.iter().flatten() {
        assert!(((x - min(x, N))..x).contains(&hallway_x));

        room_y_coordinates.insert(y);
    }
    let room_ys = room_y_coordinates.iter().copied().collect_vec();

    let doorsteps: BTreeSet<(usize, usize)> =
        room_ys.iter().copied().map(|y| (hallway_x, y)).collect();

    assert!(initial_positions
        .iter()
        .all(|per_color| per_color.len() == N));
    let mut positions: [[(usize, usize, LocationType); N]; COLOR_COUNT] =
        [[(0, 0, LocationType::StartingRoom); N]; COLOR_COUNT];
    for color in 0..COLOR_COUNT {
        for num in 0..N {
            let (x, y) = initial_positions[color][num];
            positions[color][num] = (x, y, LocationType::StartingRoom);
        }
    }

    let burrow = Burrow {
        hallway_spots,
        doorsteps,
        hallway_x,
        room_ys,
    };
    let mut migration = Migration { cost: 0, positions };
    migration.fix_critters_at_final_positions(&burrow);

    dijkstra_search(burrow, migration)
}

fn solve_part1(data: &[Vec<char>]) -> u64 {
    solve::<2>(data)
}

#[allow(unused_variables)]
fn solve_part2(data: &[Vec<char>]) -> u64 {
    assert_eq!(data.len(), 5);
    let amended_data = vec![
        data[0].clone(),
        data[1].clone(),
        data[2].clone(),
        "  #D#C#B#A#  ".chars().collect_vec(),
        "  #D#B#A#C#  ".chars().collect_vec(),
        data[3].clone(),
        data[4].clone(),
    ];
    solve::<4>(&amended_data)
}
