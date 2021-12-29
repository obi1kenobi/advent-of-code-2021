use super::values::ValueRange;

pub(super) fn mul_input_range_analysis(
    source: ValueRange,
    operand: ValueRange,
    result: ValueRange,
) -> (ValueRange, ValueRange, ValueRange) {
    let (source_negative, source_zero, source_positive) = split_range(&source);
    let (operand_negative, operand_zero, operand_positive) = split_range(&operand);
    let (result_negative, result_zero, result_positive) = split_range(&result);

    let mut source_possibilities = vec![];
    let mut operand_possibilities = vec![];
    let mut result_possibilities = vec![];
    let mut add_to_vecs = |maybe_ranges: Option<(ValueRange, ValueRange, ValueRange)>| {
        if let Some((src, op, res)) = maybe_ranges {
            source_possibilities.push(src);
            operand_possibilities.push(op);
            result_possibilities.push(res);
        }
    };

    let same_sign_tuples = [
        (source_negative.clone(), operand_negative.clone(), result_positive.clone()),
        (source_positive.clone(), operand_positive.clone(), result_positive),
    ];
    for tuple in same_sign_tuples {
        if let (Some(source), Some(operand), Some(result)) = tuple {
            add_to_vecs(mul_input_range_same_sign_input_ranges(source, operand, result));
        }
    }

    let opposite_sign_tuples = [
        (source_negative.clone(), operand_positive.clone(), result_negative.clone()),
        (source_positive.clone(), operand_negative.clone(), result_negative),
    ];
    for tuple in opposite_sign_tuples {
        if let (Some(source), Some(operand), Some(result)) = tuple {
            add_to_vecs(mul_input_range_opposite_sign_input_ranges(source, operand, result));
        }
    }

    let zero_tuples = [
        (source_positive, operand_zero.clone(), result_zero.clone()),
        (source_negative, operand_zero.clone(), result_zero.clone()),
        (source_zero.clone(), operand_positive, result_zero.clone()),
        (source_zero.clone(), operand_negative, result_zero.clone()),
        (source_zero, operand_zero, result_zero),
    ];
    for tuple in zero_tuples {
        // If all three ranges are possible, then since multiplying anything by zero gives zero,
        // the entire ranges are possible.
        if let (Some(source), Some(operand), Some(result)) = tuple {
            add_to_vecs(Some((source, operand, result)));
        }
    }

    let final_source = ValueRange::new(
        source_possibilities.iter().map(|r| r.start()).min().unwrap(),
        source_possibilities.iter().map(|r| r.end()).max().unwrap(),
    );
    let final_operand = ValueRange::new(
        operand_possibilities.iter().map(|r| r.start()).min().unwrap(),
        operand_possibilities.iter().map(|r| r.end()).max().unwrap(),
    );
    let final_result = ValueRange::new(
        result_possibilities.iter().map(|r| r.start()).min().unwrap(),
        result_possibilities.iter().map(|r| r.end()).max().unwrap(),
    );
    (final_source, final_operand, final_result)
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

    #[test]
    fn test_mul_input_range_exact_zero_result_source_must_be_zero() {
        let source = ValueRange::new(-70, 100);
        let operand = ValueRange::new(2, 25);
        let result = ValueRange::new_exact(0);

        let expected_source = ValueRange::new_exact(0);
        let expected_operand = ValueRange::new(2, 25);
        let expected_result = ValueRange::new_exact(0);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_exact_zero_result_operand_must_be_zero() {
        let source = ValueRange::new(7, 10);
        let operand = ValueRange::new(-2, 25);
        let result = ValueRange::new_exact(0);

        let expected_source = ValueRange::new(7, 10);
        let expected_operand = ValueRange::new_exact(0);
        let expected_result = ValueRange::new_exact(0);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_exact_zero_result_either_can_be_zero() {
        let source = ValueRange::new(-3, 11);
        let operand = ValueRange::new(-7, 8);
        let result = ValueRange::new_exact(0);

        let expected_source = ValueRange::new(-3, 11);
        let expected_operand = ValueRange::new(-7, 8);
        let expected_result = ValueRange::new_exact(0);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_inputs_all_span_zero() {
        let source = ValueRange::new(-3, 11);
        let operand = ValueRange::new(-7, 8);
        let result = ValueRange::new(-1, 1);

        let expected_source = ValueRange::new(-3, 11);
        let expected_operand = ValueRange::new(-7, 8);
        let expected_result = ValueRange::new(-1, 1);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_inputs_span_zero_output_is_positive() {
        let source = ValueRange::new(-3, 11);
        let operand = ValueRange::new(-7, 8);
        let result = ValueRange::new(3, 16);

        let expected_source = ValueRange::new(-3, 11);
        let expected_operand = ValueRange::new(-7, 8);
        let expected_result = ValueRange::new(3, 16);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_inputs_span_zero_output_is_positive2() {
        let source = ValueRange::new(-3, 11);
        let operand = ValueRange::new(-23, 8);
        let result = ValueRange::new(3, 7);

        let expected_source = ValueRange::new(-3, 7);
        let expected_operand = ValueRange::new(-7, 7);
        let expected_result = ValueRange::new(3, 7);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_inputs_span_zero_output_is_negative() {
        let source = ValueRange::new(-11, 3);
        let operand = ValueRange::new(-8, 7);
        let result = ValueRange::new(-16, -3);

        let expected_source = ValueRange::new(-11, 3);
        let expected_operand = ValueRange::new(-8, 7);
        let expected_result = ValueRange::new(-16, -3);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }

    #[test]
    fn test_mul_input_range_inputs_span_zero_output_is_negative2() {
        let source = ValueRange::new(-11, 3);
        let operand = ValueRange::new(-8, 23);
        let result = ValueRange::new(-7, -3);

        let expected_source = ValueRange::new(-7, 3);
        let expected_operand = ValueRange::new(-7, 7);
        let expected_result = ValueRange::new(-7, -3);

        assert_eq!(
            (expected_source, expected_operand, expected_result),
            mul_input_range_analysis(source, operand, result)
        );
    }
}
