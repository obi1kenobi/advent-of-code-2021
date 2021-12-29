use super::values::ValueRange;

pub(super) fn mul_input_range_analysis(
    source: ValueRange,
    operand: ValueRange,
    result: ValueRange,
) -> (ValueRange, ValueRange, ValueRange) {
    let (source_negative, source_zero, source_positive) = split_range(&source);
    let (operand_negative, operand_zero, operand_positive) = split_range(&operand);
    let (result_negative, result_zero, result_positive) = split_range(&result);

    // let mut source_possibilities = vec![];
    // let mut operand_possibilities = vec![];

    todo!()
}

// Solve "left * right = result" for left,
// returning either the solved number or the original left, whichever is farther from zero
fn solve_mul_away_from_zero(left: i64, right: i64, result: i64) -> i64 {
    let estimate_left = result.saturating_div(right);

    // Return either our found value or the original left value,
    // whichever is farther from zero.
    match left.signum() {
        1 => {
            assert!(estimate_left >= 0);
            std::cmp::max(left, estimate_left)
        }
        -1 => {
            assert!(estimate_left <= 0);
            std::cmp::min(left, estimate_left)
        }
        _ => unreachable!(),
    }
}

fn solve_mul_toward_zero(left: i64, right: i64, result: i64) -> i64 {
    let estimate_left = result.saturating_div(right);

    // Return either our found value or the original left value,
    // whichever is closer to zero.
    match left.signum() {
        1 => {
            assert!(estimate_left >= 0);
            std::cmp::min(left, estimate_left)
        }
        -1 => {
            assert!(estimate_left <= 0);
            std::cmp::max(left, estimate_left)
        }
        _ => unreachable!(),
    }
}

fn mul_input_range_opposite_sign_input_ranges(
    source: ValueRange,
    operand: ValueRange,
    result: ValueRange,
) -> Option<(ValueRange, ValueRange, ValueRange)> {
    let (source_low, source_high) = (source.start(), source.end());
    let (operand_low, operand_high) = (operand.start(), operand.end());
    let (result_low, result_high) = (result.start(), result.end());
    // opposite sign inputs -> result is always negative
    assert!(result_high < 0);

    let (estimate_operand_low, estimate_operand_high) = if operand_high < 0 {
        (
            solve_mul_toward_zero(operand_low, source_low, result_low),
            solve_mul_away_from_zero(operand_high, source_high, result_high),
        )
    } else if operand_low > 0 {
        (
            solve_mul_away_from_zero(operand_low, source_low, result_high),
            solve_mul_toward_zero(operand_high, source_high, result_low),
        )
    } else {
        unreachable!()
    };
    if estimate_operand_low > estimate_operand_high {
        // Got empty ranges, no solution here.
        return None;
    }

    let (estimate_source_low, estimate_source_high) = if source_high < 0 {
        (
            solve_mul_toward_zero(source_low, estimate_operand_low, result_low),
            solve_mul_away_from_zero(source_high, estimate_operand_high, result_high),
        )
    } else if source_low > 0 {
        (
            solve_mul_away_from_zero(source_low, estimate_operand_low, result_high),
            solve_mul_toward_zero(source_high, estimate_operand_high, result_low),
        )
    } else {
        unreachable!()
    };
    if estimate_source_low > estimate_source_high {
        // Got empty ranges, no solution here.
        return None;
    }

    let final_source = ValueRange::new(estimate_source_low, estimate_source_high);
    let final_operand = ValueRange::new(estimate_operand_low, estimate_operand_high);
    let multiplied_result = &final_source * &final_operand;

    multiplied_result
        .intersect(&result)
        .map(|final_result| (final_source, final_operand, final_result))
}

fn mul_input_range_same_sign_input_ranges(
    source: ValueRange,
    operand: ValueRange,
    result: ValueRange,
) -> Option<(ValueRange, ValueRange, ValueRange)> {
    let (source_low, source_high) = (source.start(), source.end());
    let (operand_low, operand_high) = (operand.start(), operand.end());
    let (result_low, result_high) = (result.start(), result.end());
    // same sign (nonzero) input ranges --> result is positive
    assert!(result_low > 0);

    let (estimate_operand_low, estimate_operand_high) = if operand_high < 0 {
        (
            solve_mul_toward_zero(operand_low, source_high, result_high),
            solve_mul_away_from_zero(operand_high, source_low, result_low),
        )
    } else if operand_low > 0 {
        (
            solve_mul_away_from_zero(operand_low, source_high, result_low),
            solve_mul_toward_zero(operand_high, source_low, result_high),
        )
    } else {
        unreachable!()
    };
    if estimate_operand_low > estimate_operand_high {
        // Got empty ranges, no solution here.
        return None;
    }

    let (estimate_source_low, estimate_source_high) = if source_high < 0 {
        (
            solve_mul_toward_zero(source_low, estimate_operand_high, result_high),
            solve_mul_away_from_zero(source_high, estimate_operand_low, result_low),
        )
    } else if source_low > 0 {
        (
            solve_mul_away_from_zero(source_low, estimate_operand_high, result_low),
            solve_mul_toward_zero(source_high, estimate_operand_low, result_high),
        )
    } else {
        unreachable!()
    };
    if estimate_source_low > estimate_source_high {
        // Got empty ranges, no solution here.
        return None;
    }

    let final_source = ValueRange::new(estimate_source_low, estimate_source_high);
    let final_operand = ValueRange::new(estimate_operand_low, estimate_operand_high);
    let multiplied_result = &final_source * &final_operand;

    multiplied_result
        .intersect(&result)
        .map(|final_result| (final_source, final_operand, final_result))
}

/// Merge a slice of potential ranges.
fn merge_ranges(ranges: &[Option<ValueRange>]) -> ValueRange {
    let range_min = ranges
        .iter()
        .filter_map(|opt| opt.clone().map(|r| r.start()))
        .min()
        .unwrap();
    let range_max = ranges
        .iter()
        .filter_map(|opt| opt.clone().map(|r| r.end()))
        .max()
        .unwrap();

    ValueRange::new(range_min, range_max)
}

/// Split a range into its negative (if any), zero (if any), and positive (if any) components.
fn split_range(range: &ValueRange) -> (Option<ValueRange>, Option<ValueRange>, Option<ValueRange>) {
    let (mut negative, mut zero, mut positive): (
        Option<ValueRange>,
        Option<ValueRange>,
        Option<ValueRange>,
    ) = Default::default();

    if range.start() < 0 {
        let range_end = if range.end() < 0 { range.end() } else { -1 };
        negative = Some(ValueRange::new(range.start(), range_end));
    }
    if range.contains(0) {
        zero = Some(ValueRange::new_exact(0));
    }
    if range.end() > 0 {
        let range_start = if range.start() > 0 { range.start() } else { 1 };
        positive = Some(ValueRange::new(range_start, range.end()));
    }

    (negative, zero, positive)
}

#[cfg(test)]
mod tests {
    use crate::analysis::{
        range_analysis::*,
        values::ValueRange,
    };

    #[test]
    fn test_mul_input_range_positive_input_ranges() {
        let source = ValueRange::new(4, 100);
        let operand = ValueRange::new(1, 26);
        let result = ValueRange::new(4, 20);

        let expected_source = ValueRange::new(4, 20);
        let expected_operand = ValueRange::new(1, 5);
        let expected_result = ValueRange::new(4, 20);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_same_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_positive_input_ranges2() {
        let source = ValueRange::new(4, 100);
        let operand = ValueRange::new(1, 26);
        let result = ValueRange::new(4, 19);

        let expected_source = ValueRange::new(4, 19);
        let expected_operand = ValueRange::new(1, 4);
        let expected_result = ValueRange::new(4, 19);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_same_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_negative_input_ranges() {
        let source = ValueRange::new(-100, -4);
        let operand = ValueRange::new(-26, -1);
        let result = ValueRange::new(4, 20);

        let expected_source = ValueRange::new(-20, -4);
        let expected_operand = ValueRange::new(-5, -1);
        let expected_result = ValueRange::new(4, 20);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_same_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_negative_input_ranges2() {
        let source = ValueRange::new(-100, -4);
        let operand = ValueRange::new(-26, -1);
        let result = ValueRange::new(4, 19);

        let expected_source = ValueRange::new(-19, -4);
        let expected_operand = ValueRange::new(-4, -1);
        let expected_result = ValueRange::new(4, 19);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_same_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_opposite_sign_input_ranges_source_negative() {
        let source = ValueRange::new(-100, -4);
        let operand = ValueRange::new(1, 26);
        let result = ValueRange::new(-20, -4);

        let expected_source = ValueRange::new(-20, -4);
        let expected_operand = ValueRange::new(1, 5);
        let expected_result = ValueRange::new(-20, -4);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_opposite_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_opposite_sign_input_ranges_source_negative2() {
        let source = ValueRange::new(-100, -4);
        let operand = ValueRange::new(1, 26);
        let result = ValueRange::new(-19, -4);

        let expected_source = ValueRange::new(-19, -4);
        let expected_operand = ValueRange::new(1, 4);
        let expected_result = ValueRange::new(-19, -4);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_opposite_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_opposite_sign_input_ranges_operand_negative() {
        let source = ValueRange::new(4, 100);
        let operand = ValueRange::new(-26, -1);
        let result = ValueRange::new(-20, -4);

        let expected_source = ValueRange::new(4, 20);
        let expected_operand = ValueRange::new(-5, -1);
        let expected_result = ValueRange::new(-20, -4);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_opposite_sign_input_ranges(source, operand, result).unwrap()
        );
    }

    #[test]
    fn test_mul_input_range_opposite_sign_input_ranges_operand_negative2() {
        let source = ValueRange::new(4, 100);
        let operand = ValueRange::new(-26, -1);
        let result = ValueRange::new(-19, -4);

        let expected_source = ValueRange::new(4, 19);
        let expected_operand = ValueRange::new(-4, -1);
        let expected_result = ValueRange::new(-19, -4);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_opposite_sign_input_ranges(source, operand, result).unwrap()
        );
    }
}
