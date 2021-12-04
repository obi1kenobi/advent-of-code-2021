use std::{env, fs};

struct Board {
    pub(crate) numbers: Vec<Vec<u64>>,
    pub(crate) called: Vec<Vec<bool>>,
}

impl Board {
    pub(crate) fn call_number(&mut self, number: u64) -> bool {
        for (i, row) in self.numbers.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                if *value == number {
                    assert!(!self.called[i][j]);
                    self.called[i][j] = true;

                    let mut x_bingo = true;
                    let mut y_bingo = true;
                    for pt in 0..5usize {
                        x_bingo &= self.called[i][pt];
                        y_bingo &= self.called[pt][j];
                    }
                    if x_bingo || y_bingo {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl From<&str> for Board {
    fn from(data: &str) -> Self {
        let numbers: Vec<Vec<u64>> = data.split('\n').map(|row| {
            row.split_ascii_whitespace()
                .filter(|x| !x.is_empty())
                .map(|number| number.parse().unwrap())
                .collect()
        }).collect();

        assert_eq!(numbers.len(), 5);
        for row in numbers.iter() {
            assert_eq!(row.len(), 5);
        }

        let called = (0..5).map(|_| (0..5).map(|_| false).collect()).collect();
        Self {
            numbers,
            called,
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

    let input_data: Vec<&str> = content
        .trim_end()
        .split("\n\n")
        .collect();
    let draws: Vec<u64> = input_data[0].split(',').map(|x| x.parse().unwrap()).collect();
    let boards: Vec<Board> = input_data[1..].iter().copied().map(Board::from).collect();

    match part {
        "1" => {
            let result = solve_part1(draws, boards);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(draws, boards);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

fn score_winning_board(board: &Board, winning_draw: u64) -> u64 {
    let mut uncalled_sum = 0u64;
    for i in 0..5usize {
        for j in 0..5usize {
            if !board.called[i][j] {
                uncalled_sum += board.numbers[i][j];
            }
        }
    }

    uncalled_sum * winning_draw
}

fn solve_part1(draws: Vec<u64>, mut boards: Vec<Board>) -> u64 {
    for draw in draws {
        for board in boards.iter_mut() {
            if board.call_number(draw) {
                return score_winning_board(board, draw);
            }
        }
    }

    unreachable!()
}

fn solve_part2(draws: Vec<u64>, mut boards: Vec<Board>) -> u64 {
    let mut winner_boards = 0usize;
    let num_boards = boards.len();
    let mut has_won: Vec<_> = boards.iter().map(|_| false).collect();
    for draw in draws {
        for (index, board) in boards.iter_mut().enumerate() {
            if has_won[index] {
                continue;
            }
            if board.call_number(draw) {
                assert!(!has_won[index]);
                has_won[index] = true;
                winner_boards += 1;

                if winner_boards == num_boards {
                    return score_winning_board(board, draw);
                }
            }
        }
    }

    unreachable!()
}
