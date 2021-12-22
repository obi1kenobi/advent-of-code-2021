use std::{collections::HashSet, env, fs};

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

    let (enhancement, input_image) = content
        .trim_end()
        .split_once("\n\n")
        .map(|(enhancement, input_image_data)| {
            let enhancements = enhancement.chars().collect_vec();
            let input_image = input_image_data
                .split('\n')
                .map(|row| row.chars().collect_vec())
                .collect_vec();
            (enhancements, input_image)
        })
        .unwrap();

    match part {
        "1" => {
            let result = solve_part1(&enhancement, &input_image);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(&enhancement, &input_image);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

fn image_lit_pixels(input_image: &[Vec<char>]) -> HashSet<(isize, isize)> {
    input_image
        .iter()
        .enumerate()
        .flat_map(|(idx, row)| {
            row.iter()
                .enumerate()
                .filter_map(move |(idy, pixel)| match *pixel {
                    '#' => Some((idx as isize, idy as isize)),
                    '.' => None,
                    _ => unreachable!("{}", *pixel),
                })
        })
        .collect()
}

fn get_5x5_kernel_value(lit_pixels: &HashSet<(isize, isize)>, pixel: (isize, isize)) -> usize {
    let mut value = 0usize;
    let kernel_offsets = (-2..=2).cartesian_product(-2..=2).collect_vec();
    for (shift, offset) in kernel_offsets.into_iter().rev().enumerate() {
        let (px, py) = pixel;
        let (ox, oy) = offset;
        let kernel_pixel = (px + ox, py + oy);

        if lit_pixels.contains(&kernel_pixel) {
            value |= 1 << shift;
        }
    }

    value
}

fn compact_kernel(enhancement: &[char]) -> HashSet<usize> {
    enhancement.iter().enumerate().filter_map(|(idx, c)| {
        match c {
            '#' => Some(idx),
            '.' => None,
            _ => unreachable!("{}", c),
        }
    }).collect()
}

fn apply_3x3_kernel_to_5x5_bit_image(kernel: &HashSet<usize>, image: usize) -> usize {
    let kernel_offsets = (-1..=1).cartesian_product(-1..=1).collect_vec();
    let starting_width = 5;

    let mut next_image = 0usize;
    for x in 1usize..=3 {
        for y in 1usize..=3 {
            let mut kernel_value = 0usize;
            for (idx, (dx, dy)) in kernel_offsets.iter().rev().copied().enumerate() {
                let composite_x = ((x as isize) - dx) as usize;
                let composite_y = ((y as isize) - dy) as usize;
                let offset = composite_x * starting_width + composite_y;
                if image & (1 << offset) != 0 {
                    kernel_value |= 1 << idx;
                }
            }

            let final_width = starting_width - 2;
            assert!(kernel_value < 512);
            if kernel.contains(&kernel_value) {
                next_image |= 1 << ((x - 1) * final_width + (y - 1));
            }
        }
    }

    next_image
}

fn stack_3x3_kernel(kernel: &HashSet<usize>) -> HashSet<usize> {
    let mut result = HashSet::new();

    let starting_width = 5usize;
    let kernel_offsets = (-1..=1).cartesian_product(-1..=1).collect_vec();

    for image in 0usize..(1 << (starting_width * starting_width)) {
        let next_image = apply_3x3_kernel_to_5x5_bit_image(kernel, image);
        let next_width = starting_width - 2;

        let center_x = 1usize;
        let center_y = 1usize;
        let mut kernel_value = 0usize;
        for (idx, (dx, dy)) in kernel_offsets.iter().rev().copied().enumerate() {
            let composite_x = ((center_x as isize) - dx) as usize;
            let composite_y = ((center_y as isize) - dy) as usize;
            let offset = composite_x * next_width + composite_y;
            if next_image & (1 << offset) != 0 {
                kernel_value |= 1 << idx;
            }
        }

        assert!(kernel_value < 512);
        if kernel.contains(&kernel_value) {
            result.insert(image);
        }
    }

    result
}

fn enhance_image_twice(
    stacked_kernel: &HashSet<usize>,
    lit_pixels: &HashSet<(isize, isize)>,
) -> HashSet<(isize, isize)> {
    let positions_to_consider: HashSet<(isize, isize)> = lit_pixels
        .iter()
        .copied()
        .flat_map(|(px, py)| {
            (-2..=2).cartesian_product(-2..=2)
                .map(move |(ex, ey)| (px + ex, py + ey))
        })
        .collect();

    positions_to_consider
        .iter()
        .copied()
        .filter(|&pixel| {
            let kernel_value = get_5x5_kernel_value(lit_pixels, pixel);
            stacked_kernel.contains(&kernel_value)
        })
        .collect()
}

fn solve_part1(enhancement: &[char], input_image: &[Vec<char>]) -> usize {
    let compacted_kernel = compact_kernel(enhancement);
    let lit_pixels = image_lit_pixels(input_image);

    let stacked_kernel = stack_3x3_kernel(&compacted_kernel);
    let fast_solution = enhance_image_twice(&stacked_kernel, &lit_pixels);

    fast_solution.len()
}

fn solve_part2(enhancement: &[char], input_image: &[Vec<char>]) -> usize {
    let compacted_kernel = compact_kernel(enhancement);
    let mut lit_pixels = image_lit_pixels(input_image);

    let stacked_kernel = stack_3x3_kernel(&compacted_kernel);
    for _ in 0..25 {
        lit_pixels = enhance_image_twice(&stacked_kernel, &lit_pixels);
    }

    lit_pixels.len()
}
