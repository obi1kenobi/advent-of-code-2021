use std::{env, fs};

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

    let components: (&str, &str) = content
        .trim_end()
        .strip_prefix("target area: x=")
        .unwrap()
        .split_once(", y=")
        .unwrap();
    let (lower_x, upper_x) = components.0.split_once("..").unwrap();
    let (lower_y, upper_y) = components.1.split_once("..").unwrap();

    let x_bounds: (i64, i64) = (lower_x.parse().unwrap(), upper_x.parse().unwrap());
    let y_bounds: (i64, i64) = (lower_y.parse().unwrap(), upper_y.parse().unwrap());

    match part {
        "1" => {
            let result = solve_part1(x_bounds, y_bounds);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(x_bounds, y_bounds);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

fn does_shot_hit(x_bounds: (i64, i64), y_bounds: (i64, i64), shot: (i64, i64)) -> bool {
    let (lower_x, upper_x) = x_bounds;
    let (lower_y, upper_y) = y_bounds;
    let (x_shot, y_shot) = shot;

    // firing to the right
    assert!(lower_x >= 0);
    assert!(upper_x >= lower_x);
    assert!(x_shot >= 0);

    // target is on our level or below us
    assert!(upper_y <= 0);
    assert!(lower_y <= upper_y);

    let mut x = 0i64;
    let mut y = 0i64;
    let mut x_vel = x_shot;
    let mut y_vel = y_shot;
    loop {
        x += x_vel;
        y += y_vel;

        assert!(x_vel >= 0);
        if x_vel > 0 {
            x_vel -= 1;
        }
        y_vel -= 1;

        if y < lower_y || x > upper_x {
            // miss!
            break false;
        }
        if x >= lower_x && x <= upper_x && y <= upper_y {
            // hit!
            break true;
        }
    }
}

fn max_height_shot(x_bounds: (i64, i64), y_bounds: (i64, i64)) -> (i64, i64) {
    let (lower_x, upper_x) = x_bounds;
    let (lower_y, upper_y) = y_bounds;

    // firing to the right
    assert!(lower_x >= 0);
    assert!(upper_x >= lower_x);

    // target is on our level or below us
    assert!(upper_y <= 0);
    assert!(lower_y <= upper_y);

    // When firing to the right and aiming below us, we can consider the coordinates separately.
    // Since we are maximizing altitude, we can count on having run out of X velocity well before
    // encountering the target area.
    //
    // This means that any trajectory that is sufficient to reach and stay within the target area's
    // X coordinate space when it runs out of X velocity would work.
    // If we fire at x_spd in the X direction, the total displacement we'd get before running out
    // of X velocity is x_shot * (x_shot + 1) / 2.
    // We can get a good approximate value for x_shot by starting with floor(sqrt(lower_x * 2)) - 1
    // and guess-and-checking up.
    let mut x_shot_guess = (((lower_x * 2) as f64).sqrt().floor()) as i64 - 1;
    let x_shot = loop {
        let distance = x_shot_guess * (x_shot_guess + 1) / 2;
        assert!(distance <= upper_x);
        if distance >= lower_x {
            break x_shot_guess;
        }

        x_shot_guess += 1;
    };

    // However, since we can consider both coordinates separately, we didn't even need the above
    // calculation. We'll only use it to validate our guess at the end.
    //
    // Given that we are firing upward but aiming below us, eventually the probe will cross Y=0.
    // Because of the symmetry of the motion, the probe will actually have a location exactly on Y=0
    // at some point in time. Therefore, given our unlimited max Y velocity at start, we'll have
    // maximized our vertical reach by firing in such a way that the probe covers
    // the entire distance from its Y=0 position to the bottom of the target area lower_y in
    // only a single time step.
    //
    // Therefore, during its fall, the probe was moving at lower_y vertical speed at Y=0.
    // In the prior time step, it must have been moving at vertical speed lower_y + 1.
    // Due to symmetry, and since the firing position is also at Y=0, it must have been the case
    // that y_shot = -lower_y - 1.
    let y_shot = -lower_y - 1;

    (x_shot, y_shot)
}

fn solve_part1(x_bounds: (i64, i64), y_bounds: (i64, i64)) -> i64 {
    let (x_shot, y_shot) = max_height_shot(x_bounds, y_bounds);

    // Since we lose 1 unit of vertical speed per time step all the way down to 0,
    // the max height is just the sum 1 + 2 + ... + y_shot, so Gauss' formula applies:
    let max_height = y_shot * (y_shot + 1) / 2;

    // Finally, simulate our shot and make sure it does hit the target.
    assert!(does_shot_hit(x_bounds, y_bounds, (x_shot, y_shot)));

    max_height
}

fn solve_part2(x_bounds: (i64, i64), y_bounds: (i64, i64)) -> usize {
    let (lower_x, upper_x) = x_bounds;
    let (lower_y, upper_y) = y_bounds;

    // firing to the right
    assert!(lower_x >= 0);
    assert!(upper_x >= lower_x);

    // target is on our level or below us
    assert!(upper_y <= 0);
    assert!(lower_y <= upper_y);

    // We can just parameter-sweep the space, it's not that large.
    //
    // We can't fire slower in X than the max-height shot.
    // We can't fire faster in X than upper_x, or we'd overshoot immediately.
    // We can't fire faster (more positive) in Y than the max-height shot.
    // We can't fire slower (more negative) in Y than lower_y, or we'd overshoot immediately.
    let (max_height_shot_x, max_height_shot_y) = max_height_shot(x_bounds, y_bounds);
    let x_range = max_height_shot_x..=upper_x;
    let y_range = lower_y..=max_height_shot_y;

    x_range
        .cartesian_product(y_range)
        .filter(|shot| does_shot_hit(x_bounds, y_bounds, *shot))
        .count()
}
