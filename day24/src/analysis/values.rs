use std::{
    fmt::Display,
    ops::{Add, Mul, RangeInclusive, Sub},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vid(pub usize);

impl Vid {
    pub const UNDEFINED: Vid = Vid(0);
}

#[derive(Debug)]
pub struct VidMaker {
    next_id: usize,
}

impl VidMaker {
    pub fn new() -> Self {
        Self { next_id: 1 }
    }
}

impl Iterator for VidMaker {
    type Item = Vid;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_id == usize::MAX {
            None
        } else {
            let next_id = self.next_id;
            self.next_id += 1;
            Some(Vid(next_id))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueRange {
    range: RangeInclusive<i64>,
}

impl ValueRange {
    pub const MAX: ValueRange = ValueRange::new(i64::MIN, i64::MAX);

    #[inline]
    pub const fn new(start: i64, end: i64) -> Self {
        assert!(start <= end);
        Self { range: start..=end }
    }

    #[inline]
    pub const fn new_exact(exact: i64) -> Self {
        Self::new(exact, exact)
    }

    #[inline]
    pub fn is_exact(&self) -> bool {
        self.range.start() == self.range.end()
    }

    #[inline]
    pub fn is_exactly(&self, value: i64) -> bool {
        self.range.start() == &value && self.range.end() == &value
    }

    #[inline]
    pub const fn start(&self) -> i64 {
        *self.range.start()
    }

    #[inline]
    pub const fn end(&self) -> i64 {
        *self.range.end()
    }

    #[inline]
    pub fn contains(&self, value: i64) -> bool {
        self.range.contains(&value)
    }

    #[inline]
    pub fn contains_range(&self, other: &ValueRange) -> bool {
        self.contains(other.start()) && self.contains(other.end())
    }

    #[inline]
    pub fn intersect(&self, other: &ValueRange) -> Option<ValueRange> {
        let lower = std::cmp::max(self.start(), other.start());
        let upper = std::cmp::min(self.end(), other.end());
        if lower <= upper {
            Some(Self::new(lower, upper))
        } else {
            None
        }
    }
}

impl Display for ValueRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self == &ValueRange::MAX {
            write!(f, "max")
        } else if self.start() == ValueRange::MAX.start() {
            write!(f, "-inf..={}", self.end())
        } else if self.end() == ValueRange::MAX.end() {
            write!(f, "{}..=+inf", self.start())
        } else {
            write!(f, "{:?}", self.range)
        }
    }
}

impl From<RangeInclusive<i64>> for ValueRange {
    #[inline]
    fn from(range: RangeInclusive<i64>) -> Self {
        Self { range }
    }
}

impl Add for &ValueRange {
    type Output = ValueRange;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::new(
            self.start().saturating_add(rhs.start()),
            self.end().saturating_add(rhs.end()),
        )
    }
}

impl Add<i64> for &ValueRange {
    type Output = ValueRange;

    #[inline]
    fn add(self, rhs: i64) -> Self::Output {
        Self::Output::new(
            self.start().saturating_add(rhs),
            self.end().saturating_add(rhs),
        )
    }
}

impl Sub for &ValueRange {
    type Output = ValueRange;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(
            self.start().saturating_sub(rhs.end()),
            self.end().saturating_sub(rhs.start()),
        )
    }
}

impl Sub<i64> for &ValueRange {
    type Output = ValueRange;

    #[inline]
    fn sub(self, rhs: i64) -> Self::Output {
        Self::Output::new(
            self.start().saturating_sub(rhs),
            self.end().saturating_sub(rhs),
        )
    }
}

impl Mul<i64> for &ValueRange {
    type Output = ValueRange;

    fn mul(self, rhs: i64) -> Self::Output {
        let endpoint_a = self.start().saturating_mul(rhs);
        let endpoint_b = self.end().saturating_mul(rhs);
        Self::Output::new(
            std::cmp::min(endpoint_a, endpoint_b),
            std::cmp::max(endpoint_a, endpoint_b),
        )
    }
}

impl Mul<&ValueRange> for &ValueRange {
    type Output = ValueRange;

    fn mul(self, rhs: &ValueRange) -> Self::Output {
        if rhs.is_exact() {
            return self * rhs.start();
        }

        let range_a = self * rhs.start();
        let range_b = self * rhs.end();

        let endpoints = [
            range_a.start(),
            range_a.end(),
            range_b.start(),
            range_b.end(),
        ];
        Self::Output::new(
            *endpoints.iter().min().unwrap(),
            *endpoints.iter().max().unwrap(),
        )
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Exact(Vid, i64),
    Input(Vid, usize, ValueRange), // which input number is it
    Unknown(Vid, ValueRange),
    Undefined,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Exact(_, l1), Self::Exact(_, r1)) => {
                // Don't compare value IDs for exact values! Constants are equal by value.
                l1 == r1
            }
            (Self::Input(l0, l1, l2), Self::Input(r0, r1, r2)) => l0 == r0 && l1 == r1 && l2 == r2,
            (Self::Unknown(l0, l1), Self::Unknown(r0, r1)) => l0 == r0 && l1 == r1,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for Value {
    fn assert_receiver_is_total_eq(&self) {}
}

impl Value {
    #[allow(dead_code)]
    #[inline]
    pub fn vid(&self) -> Vid {
        match self {
            Value::Exact(vid, _) | Value::Input(vid, _, _) | Value::Unknown(vid, _) => *vid,
            Value::Undefined => unreachable!(),
        }
    }

    #[inline]
    pub fn range(&self) -> ValueRange {
        match self {
            Value::Exact(_, n) => ValueRange::new(*n, *n),
            Value::Input(_, _, range) => range.clone(),
            Value::Unknown(_, range) => range.clone(),
            Value::Undefined => unreachable!(),
        }
    }

    pub fn narrow_range(self, range: &ValueRange) -> Value {
        let current_range = self.range();
        let final_range = current_range.intersect(range).unwrap();

        match self {
            Value::Exact(_, _) => self,
            Value::Input(vid, inp, _) => Value::Input(vid, inp, final_range),
            Value::Unknown(vid, _) => {
                if final_range.is_exact() {
                    Value::Exact(vid, final_range.start())
                } else {
                    Value::Unknown(vid, final_range)
                }
            }
            Value::Undefined => unreachable!(),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Exact(vid, value) => write!(f, "{}: Exact({})", vid.0, *value),
            Value::Input(vid, inp, range) => write!(f, "{}: Input_{}({})", vid.0, inp, range),
            Value::Unknown(vid, range) => write!(f, "{}: Unknown({})", vid.0, range),
            Value::Undefined => write!(f, "N/A: Undefined"),
        }
    }
}
