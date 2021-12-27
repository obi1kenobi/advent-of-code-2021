use std::{ops::{Add, RangeInclusive, Sub}};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Exact(Vid, i64),
    Input(Vid, usize, ValueRange),  // which input number is it
    Unknown(Vid, ValueRange),
    Undefined,
}

impl Value {
    #[inline]
    pub fn vid(&self) -> Vid {
        match self {
            Value::Exact(vid, _) |
            Value::Input(vid, _, _) |
            Value::Unknown(vid, _) => *vid,
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
            },
            Value::Undefined => unreachable!(),
        }
    }
}
