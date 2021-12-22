use std::{cmp::{max, min}, env, fs};

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

    let input_data: (u64, u64) = content
        .trim_end()
        .split_once('\n')
        .map(|(first, second)| {
            (
                first
                    .strip_prefix("Player 1 starting position: ")
                    .unwrap()
                    .parse()
                    .unwrap(),
                second
                    .strip_prefix("Player 2 starting position: ")
                    .unwrap()
                    .parse()
                    .unwrap(),
            )
        })
        .unwrap();

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

fn solve_part1((p1_start, p2_start): (u64, u64)) -> u64 {
    let mut dice_rolls = 0u64;
    let mut next_dice_roll = 1u64;
    let max_dice_roll = 100u64;

    let mut p1_pos = p1_start;
    let mut p2_pos = p2_start;
    let max_pos = 10u64;

    let mut p1_score = 0u64;
    let mut p2_score = 0u64;
    let target_score = 1000u64;

    loop {
        let roll_total = 3 * next_dice_roll + 3;
        next_dice_roll += 3;
        dice_rolls += 3;
        if next_dice_roll > max_dice_roll {
            next_dice_roll -= max_dice_roll;
        }

        // -1 then +1 to get range [1, max_pos] rather than [0, max_pos - 1]
        p1_pos = (p1_pos + roll_total - 1) % max_pos + 1;

        p1_score += p1_pos;
        if p1_score >= target_score {
            break dice_rolls * p2_score;
        }

        let roll_total = 3 * next_dice_roll + 3;
        next_dice_roll += 3;
        dice_rolls += 3;
        if next_dice_roll > max_dice_roll {
            next_dice_roll -= max_dice_roll;
        }

        // -1 then +1 to get range [1, max_pos] rather than [0, max_pos - 1]
        p2_pos = (p2_pos + roll_total - 1) % max_pos + 1;

        p2_score += p2_pos;
        if p2_score >= target_score {
            break dice_rolls * p1_score;
        }
    }
}

fn solve_part2((p1_start, p2_start): (u64, u64)) -> u64 {
    let max_score = 21usize;
    let score_len = 22usize;
    let max_pos = 10usize;
    let max_turns = 40usize;
    let turns_len = max_turns + 1;

    // dp axes in order:
    // - turn number: even means p1 plays next, odd means p2 plays next
    // - p1 position - 1, then p2 position - 1
    // - p1 score w/ max 21, then p2 score w/ max 21
    let mut dp = vec![vec![vec![vec![vec![0u64; score_len]; score_len]; max_pos]; max_pos]; turns_len];

    // 3 rolls in 1..=3 = total interval 3..=9
    let die_total_interval = 3..=9;
    let mut roll_weights = [0u64; 10];
    for die_a in 1..=3 {
        for die_b in 1..=3 {
            for die_c in 1..=3 {
                let sum = die_a + die_b + die_c;
                roll_weights[sum] += 1;
            }
        }
    }

    // start the game with one universe:
    // on turn 0, with the players on their positions, and with 0 score each
    dp[0][p1_start as usize - 1][p2_start as usize - 1][0][0] = 1;

    for turn_number in 0..(max_turns - 1) {
        for p1_pos in 1..=max_pos {
            let p1_pos_index = p1_pos - 1;

            for p2_pos in 1..=max_pos {
                let p2_pos_index = p2_pos - 1;

                for p1_score in 0..max_score {
                    for p2_score in 0..max_score {
                        for rolled_total in die_total_interval.clone() {
                            let roll_weight = roll_weights[rolled_total];

                            match turn_number % 2 {
                                0 => {
                                    // p1's turn
                                    let next_p1_pos = (p1_pos + rolled_total - 1) % max_pos + 1;
                                    let next_p1_pos_index = next_p1_pos - 1;

                                    let next_p1_score = min(p1_score + next_p1_pos, max_score);
                                    dp[turn_number + 1][next_p1_pos_index]
                                        [p2_pos_index][next_p1_score][p2_score] +=
                                        roll_weight * dp[turn_number][p1_pos_index]
                                            [p2_pos_index][p1_score][p2_score];
                                }
                                1 => {
                                    // p2's turn
                                    let next_p2_pos = (p2_pos + rolled_total - 1) % max_pos + 1;
                                    let next_p2_pos_index = next_p2_pos - 1;

                                    let next_p2_score = min(p2_score + next_p2_pos, max_score);
                                    dp[turn_number + 1][p1_pos_index]
                                        [next_p2_pos_index][p1_score][next_p2_score]
                                        += roll_weight * dp[turn_number]
                                        [p1_pos_index][p2_pos_index][p1_score][p2_score];
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
        }
    }

    // count win conditions
    let mut p1_wins = 0u64;
    let mut p2_wins = 0u64;

    #[allow(clippy::needless_range_loop)]
    for turn_number in 1..=max_turns {
        for p1_pos in 1..=max_pos {
            let p1_pos_index = p1_pos - 1;

            for p2_pos in 1..=max_pos {
                let p2_pos_index = p2_pos - 1;

                for losing_score in 0..max_score {
                    match turn_number % 2 {
                        0 => {
                            // p1 would have played next so p2 won
                            p2_wins += dp[turn_number][p1_pos_index][p2_pos_index][losing_score]
                        [max_score];
                        }
                        1 => {
                            // p2 would have played next so p1 won
                            p1_wins += dp[turn_number][p1_pos_index][p2_pos_index][max_score]
                        [losing_score];
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    max(p1_wins, p2_wins)
}
