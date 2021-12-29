use std::{
    cmp::Ordering,
    collections::{btree_map::OccupiedError, BTreeMap},
    fmt::Display,
    ops::Index,
};

#[allow(unused_imports)]
use itertools::Itertools;

use crate::parser::{Instruction, Operand, Register};

use self::{
    range_analysis::mul_input_range_analysis,
    values::{Value, ValueRange, Vid, VidMaker},
};

mod range_analysis;
pub mod values;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstrId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rsid(pub usize);

pub const INITIAL_RSID: Rsid = Rsid(0);

#[derive(Debug)]
pub struct RsidMaker {
    next_id: usize,
}

impl RsidMaker {
    pub fn new() -> Self {
        Self { next_id: 1 }
    }
}

impl Iterator for RsidMaker {
    type Item = Rsid;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_id == usize::MAX {
            None
        } else {
            let next_id = self.next_id;
            self.next_id += 1;
            Some(Rsid(next_id))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterState {
    pub registers: [Vid; 4],
}

impl RegisterState {
    #[inline]
    pub fn new(registers: [Vid; 4]) -> Self {
        Self { registers }
    }
}

#[derive(Debug, Clone)]
pub struct AnnotatedInstr {
    pub id: InstrId,
    pub follows: Option<InstrId>,
    pub instr: Instruction,

    // Input instructions have Vid::UNDEFINED as source and operand.
    pub source: Vid,
    pub operand: Vid,

    pub result: Vid,

    // Register states before and after the instruction.
    pub state_before: Rsid,
    pub state_after: Rsid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrunedReason {
    NoOpInputs,
    NoRegisterChange,
    ResultNeverUsed,
    UnnecessaryClearOnFirstUse,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValueRangeAnalysis(BTreeMap<Vid, Value>);

impl ValueRangeAnalysis {
    #[inline]
    pub fn new() -> Self {
        Self(Default::default())
    }

    #[inline]
    pub fn try_insert(
        &mut self,
        vid: Vid,
        value: Value,
    ) -> Result<&mut Value, OccupiedError<Vid, Value>> {
        self.0.try_insert(vid, value)
    }

    #[inline]
    pub fn into_inner(self) -> BTreeMap<Vid, Value> {
        self.0
    }

    #[allow(dead_code)]
    #[inline]
    pub fn insert(&mut self, vid: Vid, value: Value) -> Option<Value> {
        self.0.insert(vid, value)
    }

    pub fn narrow_value_range(&mut self, vid: Vid, range: &ValueRange) -> ValueRange {
        let owned_value = self.0.remove(&vid).unwrap();
        let final_result = owned_value.narrow_range(range);
        let final_range = final_result.range();
        self.try_insert(vid, final_result).unwrap();
        final_range
    }
}

impl Index<&Vid> for ValueRangeAnalysis {
    type Output = Value;

    fn index(&self, index: &Vid) -> &Self::Output {
        self.0.index(index)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ValueEquivalenceAnalysis(BTreeMap<Vid, Vid>);

impl ValueEquivalenceAnalysis {
    fn are_equivalent(&mut self, values: &mut ValueRangeAnalysis, left: Vid, right: Vid) -> bool {
        let left_equivalent = self.get_equivalent_value(values, left);
        let right_equivalent = self.get_equivalent_value(values, right);

        if left_equivalent == right_equivalent {
            true
        } else {
            // Not all constants are marked equivalent to each other,
            // so these values may still be equal to each other. Check their values for equality.
            values[&left_equivalent] == values[&right_equivalent]
        }
    }

    /// Should only be used for impl Display!
    /// Otherwise, can lead to long chains rather than flat forests of equivalent values.
    fn get_equivalent_value_no_update(&self, vid: Vid) -> Vid {
        *self.0.get(&vid).unwrap_or(&vid)
    }

    fn get_equivalent_value(&mut self, values: &mut ValueRangeAnalysis, vid: Vid) -> Vid {
        // Union-find on the equivalent values graph, always pointing toward lower-numbered Vids.
        let mut current_vid = vid;
        let mut ancestors = vec![];

        let initial_range = &values[&vid].range();
        let mut range = initial_range.clone();
        while let Some(parent) = self.0.get(&current_vid) {
            assert!(*parent < current_vid); // protect against cycles

            let parent_range = &values[parent].range();
            range = range.intersect(parent_range).unwrap();

            ancestors.push(current_vid);
            current_vid = *parent;
        }

        // Not all the values we came across agreed on the range of possible values.
        // Shrink all the ranges to match the common intersection across all of them.
        if initial_range != &range {
            for ancestor in &ancestors {
                values.narrow_value_range(*ancestor, &range);
            }
        }

        // We skip one element from the back since the last element had the correct parent.
        // All other elements need to point to this final equivalent value.
        for ancestor in ancestors.into_iter().rev().skip(1) {
            self.0.insert(ancestor, current_vid).unwrap();
        }

        current_vid
    }

    fn update_equivalent_values(&mut self, values: &mut ValueRangeAnalysis, left: Vid, right: Vid) {
        if left == right {
            // No-op.
            return;
        }

        // Update the ranges of the values to match each other.
        let left_range = &values[&left].range();
        let right_range = &values[&right].range();
        values.narrow_value_range(left, right_range);
        values.narrow_value_range(right, left_range);

        // Union-find on the equivalent values graph, always pointing toward lower-numbered Vids.
        let left_equivalent = self.get_equivalent_value(values, left);
        let right_equivalent = self.get_equivalent_value(values, right);

        if left_equivalent == right_equivalent {
            // These values are already known to be equivalent. Nothing further to be done.
            return;
        }

        let (lower_vid, higher_vid) = match left_equivalent.cmp(&right_equivalent) {
            Ordering::Less => (left_equivalent, right_equivalent),
            Ordering::Greater => (right_equivalent, left_equivalent),
            Ordering::Equal => unreachable!("{:?} {:?}", left_equivalent, right_equivalent),
        };

        self.0.insert(higher_vid, lower_vid);
    }
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub input_program: BTreeMap<InstrId, Instruction>,
    pub annotated: BTreeMap<InstrId, AnnotatedInstr>,
    pub pruned: BTreeMap<InstrId, PrunedReason>,
    pub values: ValueRangeAnalysis,
    pub register_states: BTreeMap<Rsid, RegisterState>,
    pub inputs: Vec<Vid>, // the value IDs of all input instructions
    pub equivalent_values: ValueEquivalenceAnalysis,
}

impl Display for Analysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (instr_id, instr) in &self.input_program {
            if let Some(prune_reason) = self.pruned.get(instr_id) {
                writeln!(f, "{}  *pruned: {:?}", instr, *prune_reason)?;
            } else {
                writeln!(f, "{}", instr)?;
            }

            let annotated = &self.annotated[instr_id];
            let equivalent_registers_after = &self.register_states[&annotated.state_after]
                .registers
                .map(|v| self.equivalent_values.get_equivalent_value_no_update(v));

            writeln!(
                f,
                "[{: ^22} | {: ^22} | {: ^22} | {: ^22}]",
                format!("{}", self.values[&equivalent_registers_after[0]]),
                format!("{}", self.values[&equivalent_registers_after[1]]),
                format!("{}", self.values[&equivalent_registers_after[2]]),
                format!("{}", self.values[&equivalent_registers_after[3]]),
            )?;
        }

        Ok(())
    }
}

impl From<Vec<Instruction>> for Analysis {
    fn from(instrs: Vec<Instruction>) -> Self {
        let input_program: BTreeMap<InstrId, Instruction> = instrs
            .iter()
            .enumerate()
            .map(|(idx, instr)| (InstrId(idx), *instr))
            .collect();

        let mut annotated: BTreeMap<InstrId, AnnotatedInstr> = Default::default();
        let mut values = ValueRangeAnalysis::new();
        let mut register_states: BTreeMap<Rsid, RegisterState> = Default::default();
        let mut inputs = Default::default();

        values.try_insert(Vid::UNDEFINED, Value::Undefined).unwrap();

        perform_value_numbering(
            &input_program,
            &mut annotated,
            &mut values,
            &mut register_states,
            &mut inputs,
        );

        Self {
            input_program,
            annotated,
            pruned: Default::default(),
            values,
            register_states,
            inputs,
            equivalent_values: Default::default(),
        }
    }
}

impl Analysis {
    pub fn constant_propagation(mut self) -> Self {
        for ann_instr in self.annotated.values() {
            let source_vid = ann_instr.source;
            let operand_vid = ann_instr.operand;
            let result_vid = ann_instr.result;
            let source = &self.values[&source_vid];
            let operand = &self.values[&operand_vid];

            if let (&Value::Exact(_, left), &Value::Exact(_, right)) = (source, operand) {
                let result_value = match ann_instr.instr {
                    Instruction::Input(_) => unreachable!(),
                    Instruction::Add(_, _) => left + right,
                    Instruction::Mul(_, _) => left * right,
                    Instruction::Div(_, _) => left / right,
                    Instruction::Mod(_, _) => left % right,
                    Instruction::Equal(_, _) => {
                        if left == right {
                            1
                        } else {
                            0
                        }
                    }
                };
                self.values
                    .narrow_value_range(result_vid, &ValueRange::new_exact(result_value));
            }
        }

        self
    }

    pub fn operation_definedness(mut self) -> Self {
        for ann_instr in self.annotated.values() {
            match ann_instr.instr {
                Instruction::Input(_)
                | Instruction::Add(_, _)
                | Instruction::Mul(_, _)
                | Instruction::Div(_, _) => {
                    // For all these operations except Instruction::Div(),
                    // all input and output values are possible so there are
                    // no definedness invariants to be used.
                    //
                    // For Instruction::Div(), it's not useful to apply its invariant
                    // (divisor is non-zero) during this step, because our chosen
                    // value range representation almost certainly won't be able to use it.
                    // The operation_definedness() pass is designed to only need to be run once,
                    // so the Instruction::Div() invariant here would be wasted.
                    // Instead, it will be applied in the input/output range analysis pass,
                    // which can be run multiple times and will maximize the odds the invariant
                    // is actually useful.
                    continue;
                }
                Instruction::Mod(_, _) => {
                    // The source must be non-negative, and the operand must be strictly positive.
                    // The result must be between zero and the operand value minus one.
                    self.values
                        .narrow_value_range(ann_instr.source, &ValueRange::new(0, i64::MAX));
                    let operand_range = self
                        .values
                        .narrow_value_range(ann_instr.operand, &ValueRange::new(1, i64::MAX));
                    self.values.narrow_value_range(
                        ann_instr.result,
                        &ValueRange::new(0, operand_range.end() - 1),
                    );
                }
                Instruction::Equal(_, _) => {
                    // The result must be either 0 or 1.
                    self.values
                        .narrow_value_range(ann_instr.result, &ValueRange::new(0, 1));
                }
            }
        }

        self
    }

    pub fn prune_for_no_change_in_registers(mut self) -> Self {
        for ann_instr in self.annotated.values() {
            let registers_before = &self.register_states[&ann_instr.state_before].registers;
            let registers_after = &self.register_states[&ann_instr.state_after].registers;

            let mut no_change = true;
            for (r_before, r_after) in registers_before.iter().zip(registers_after) {
                if !self
                    .equivalent_values
                    .are_equivalent(&mut self.values, *r_before, *r_after)
                {
                    no_change = false;
                    break;
                }
            }

            if no_change {
                self.pruned
                    .entry(ann_instr.id)
                    .or_insert(PrunedReason::NoRegisterChange);
            }
        }

        self
    }

    pub fn known_operation_results(mut self) -> Self {
        for ann_instr in self.annotated.values() {
            if matches!(ann_instr.instr, Instruction::Input(_)) {
                continue;
            }

            let source_vid = ann_instr.source;
            let operand_vid = ann_instr.operand;
            let result_vid = ann_instr.result;
            let source_range = &self.values[&source_vid].range();
            let operand_range = &self.values[&operand_vid].range();

            let was_already_no_op =
                self.equivalent_values
                    .are_equivalent(&mut self.values, source_vid, result_vid);

            match ann_instr.instr {
                Instruction::Input(_) => unreachable!(),
                Instruction::Add(_, _) => {
                    // Adding zero is a no-op.
                    if source_range.is_exactly(0) {
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            operand_vid,
                            result_vid,
                        );
                    }
                    if operand_range.is_exactly(0) {
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            source_vid,
                            result_vid,
                        );
                    }
                }
                Instruction::Mul(_, _) => {
                    // Multiplying by one is a no-op.
                    // Multiplying by zero produces zero.
                    if source_range.is_exactly(1) || operand_range.is_exactly(0) {
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            operand_vid,
                            result_vid,
                        );
                    }
                    if operand_range.is_exactly(1) || source_range.is_exactly(0) {
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            source_vid,
                            result_vid,
                        );
                    }
                }
                Instruction::Div(_, _) => {
                    // If the operand's range ends on zero on either side,
                    // we can shrink it to not include zero since division by zero is not allowed.
                    if operand_range.start() == 0 {
                        self.values.narrow_value_range(
                            operand_vid,
                            &ValueRange::new(1, operand_range.end()),
                        );
                    }
                    if operand_range.end() == 0 {
                        self.values.narrow_value_range(
                            operand_vid,
                            &ValueRange::new(operand_range.start(), -1),
                        );
                    }

                    // Dividing zero by anything is a no-op.
                    // Dividing by one is a no-op.
                    if source_range.is_exactly(0) || operand_range.is_exactly(1) {
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            source_vid,
                            result_vid,
                        );
                    }
                }
                Instruction::Mod(_, _) => {
                    // Anything mod 1 will produce 0.
                    if operand_range.is_exactly(1) {
                        self.values
                            .narrow_value_range(result_vid, &ValueRange::new_exact(0));
                    }

                    // If the maximum of the source's range is less than the minimum of
                    // the mod operand's range, the mod is a no-op.
                    // For example, for integers: 4 mod 12 == 4.
                    // For example, for ranges: (4..=12) mod (23..=25) = (4..=12)
                    if source_range.end() < operand_range.start() {
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            source_vid,
                            result_vid,
                        );
                    }
                }
                Instruction::Equal(_, _) => {
                    // If the two inputs have equivalent values, the result is always 1.
                    if self.equivalent_values.are_equivalent(
                        &mut self.values,
                        source_vid,
                        operand_vid,
                    ) {
                        self.values
                            .narrow_value_range(result_vid, &ValueRange::new_exact(1));
                    }

                    // If the two inputs have non-overlapping ranges, the result is always 0.
                    if source_range.intersect(operand_range).is_none() {
                        self.values
                            .narrow_value_range(result_vid, &ValueRange::new_exact(0));
                    }
                }
            }

            let is_now_no_op =
                self.equivalent_values
                    .are_equivalent(&mut self.values, source_vid, result_vid);

            if is_now_no_op && !was_already_no_op {
                self.pruned
                    .try_insert(ann_instr.id, PrunedReason::NoOpInputs)
                    .unwrap();
            }
        }

        self
    }

    pub fn forward_value_range_analysis(mut self) -> Self {
        for ann_instr in self.annotated.values() {
            if matches!(ann_instr.instr, Instruction::Input(_)) {
                continue;
            }

            let source_vid = ann_instr.source;
            let operand_vid = ann_instr.operand;
            let result_vid = ann_instr.result;
            let source_range = &self.values[&source_vid].range();
            let operand_range = &self.values[&operand_vid].range();

            let result_range = match ann_instr.instr {
                Instruction::Input(_) => unreachable!(),
                Instruction::Add(_, _) => Some(source_range + operand_range),
                Instruction::Mul(_, _) => Some(source_range * operand_range),
                Instruction::Div(_, _) => {
                    // The Advent of Code challenge input only contains divisions
                    // by a literal value, and never divides one register by another.
                    //
                    // We can implement the easy literal division range propagation,
                    // and let the more complex register case claim "no information" about
                    // the range of the result.
                    if operand_range.is_exact() {
                        let operand_value = operand_range.start();
                        let result_a = source_range.start().saturating_div(operand_value);
                        let result_b = source_range.end().saturating_div(operand_value);
                        Some(ValueRange::new(
                            std::cmp::min(result_a, result_b),
                            std::cmp::max(result_a, result_b),
                        ))
                    } else {
                        None
                    }
                }
                Instruction::Mod(_, _) | Instruction::Equal(_, _) => {
                    // There isn't any direct forward range analysis that can be done here
                    // that isn't already covered in the known operation results pass.
                    None
                }
            };
            if let Some(result_range) = result_range {
                self.values.narrow_value_range(result_vid, &result_range);
            }
        }

        self
    }

    pub fn backward_value_range_analysis(mut self) -> Self {
        for ann_instr in self.annotated.values().rev() {
            if matches!(ann_instr.instr, Instruction::Input(_)) {
                continue;
            }

            let source_vid = ann_instr.source;
            let operand_vid = ann_instr.operand;
            let result_vid = ann_instr.result;

            let mut source_range = self.values[&source_vid].range();
            let mut operand_range = self.values[&operand_vid].range();
            let mut result_range = self.values[&result_vid].range();

            match ann_instr.instr {
                Instruction::Input(_) => unreachable!(),
                Instruction::Add(_, _) => {
                    loop {
                        let next_source = self
                            .values
                            .narrow_value_range(source_vid, &(&result_range - &operand_range));
                        let next_operand = self
                            .values
                            .narrow_value_range(operand_vid, &(&result_range - &next_source));
                        let next_result = self
                            .values
                            .narrow_value_range(result_vid, &(&next_source + &next_operand));

                        if source_range == next_source
                            && operand_range == next_operand
                            && result_range == next_result
                        {
                            // Fixpoint found.
                            break;
                        }

                        source_range = self.values[&source_vid].range();
                        operand_range = self.values[&operand_vid].range();
                        result_range = self.values[&result_vid].range();
                    }
                }
                Instruction::Mul(_, _) => {
                    let (final_source, final_operand, final_result) =
                        mul_input_range_analysis(source_range, operand_range, result_range);
                    self.values.narrow_value_range(source_vid, &final_source);
                    self.values.narrow_value_range(operand_vid, &final_operand);
                    self.values.narrow_value_range(result_vid, &final_result);
                }
                Instruction::Div(_, _) => {
                    // The Advent of Code challenge input only contains divisions
                    // by a literal value, and never divides one register by another.
                    //
                    // We can implement the easy "exact divisor" division range propagation,
                    // and let the more complex "divisor range" case claim "no information" about
                    // the range of the inputs.
                    if operand_range.is_exact() {
                        let result_range = &self.values[&result_vid].range();
                        let divisor = operand_range.start();
                        if divisor == i64::MIN {
                            // The complicated overflow situations here, and the unlikelihood
                            // of this case being useful at all make it not worth implementing.
                            continue;
                        } else {
                            let multiplied_range = result_range * divisor;

                            // We are safe from under/overflows since:
                            // - only i64::MIN would overflow at abs()
                            // - the abs() result is positive so the -1 won't underflow.
                            let max_non_truncated_value = divisor.abs() - 1;

                            // Account for possible truncation on either side.
                            let final_range = ValueRange::new(
                                multiplied_range
                                    .start()
                                    .saturating_sub(max_non_truncated_value),
                                multiplied_range
                                    .end()
                                    .saturating_add(max_non_truncated_value),
                            );
                            self.values.narrow_value_range(source_vid, &final_range);
                        }
                    }
                }
                Instruction::Mod(_, _) => {
                    // The only case where we can backpropagate information through a mod
                    // is if the source range is known to always be within (0, operand - 1)
                    // for the lowest possible value of the mod operand.
                    //
                    // However, in that case we know the mod is a complete no-op, and that
                    // is already handled by the known operation results pass.
                }
                Instruction::Equal(_, _) => {
                    if result_range.is_exactly(0) {
                        // The inputs are known non-equal. If either input value has an exact range,
                        // then that specific value may be excluded from the other input's range.
                        // Because of our range representation, we can only exclude values
                        // at the range endpoints.
                        loop {
                            let source_range = self.values[&source_vid].range();
                            let operand_range = self.values[&operand_vid].range();
                            let mut next_operand_range = operand_range.clone();
                            let mut next_source_range = source_range.clone();

                            let tuples = [
                                (
                                    &source_range,
                                    &operand_range,
                                    operand_vid,
                                    &mut next_operand_range,
                                ),
                                (
                                    &operand_range,
                                    &source_range,
                                    source_vid,
                                    &mut next_source_range,
                                ),
                            ];
                            for (op_range, target_range, target_vid, next_target_range) in tuples {
                                if op_range.is_exact() {
                                    let endpoint = op_range.start();
                                    if target_range.start() == endpoint {
                                        *next_target_range = self.values.narrow_value_range(
                                            target_vid,
                                            &(ValueRange::new(endpoint + 1, target_range.end())),
                                        )
                                    }
                                    if target_range.end() == endpoint {
                                        *next_target_range = self.values.narrow_value_range(
                                            target_vid,
                                            &(ValueRange::new(target_range.start(), endpoint - 1)),
                                        )
                                    }
                                }
                            }
                            if source_range == next_source_range
                                && operand_range == next_operand_range
                            {
                                // Fixpoint reached.
                                break;
                            }
                        }
                    } else if result_range.is_exactly(1) {
                        // The inputs are known to be equal, mark the source and operand values
                        // as equivalent to each other.
                        self.equivalent_values.update_equivalent_values(
                            &mut self.values,
                            source_vid,
                            operand_vid,
                        );
                    } else {
                        // If the inputs aren't known to be equal or non-equal, then
                        // there's no information to be extracted here.
                        continue;
                    }
                }
            }
        }

        self
    }

    /// Look for mul and div/mod no-op instruction groups like:
    /// 1) "mul x 13; add x 10; mul 2; div x 26", in which case:
    ///    - this is essentially "mul x 26; add x 20" followed by the div
    ///    - the div truncates out the addition, and leaves us with the original input to the mul.
    /// 2) "mul x 13; add x 10; mul 2; mod x 26", in which case:
    ///    - this is essentially "mul x 26; add x 20" followed by the mod
    ///    - the mod eliminates all but the added 20
    /// This pass also supports addition operations that refer to a register, so long as
    /// the accumulated remainder always remains positive and smaller than the divisor or modulus.
    ///
    /// We can also mark the source register of the latter instruction as unused, and perhaps
    /// some of the intermediate instructions as well. This may open up more opportunities
    /// for optimization in other passes.
    /// (TODO the above, perhaps best to mutate self to record this info for later, so as not to
    ///  disable other numeric passes due to the Undefined register value).
    pub fn matched_mul_and_div_or_mod(mut self) -> Self {
        // Vid -> known to be multiple i64 -> of vid with (remainder, origin if single)
        #[allow(clippy::type_complexity)]
        let mut known_div_values: BTreeMap<
            Vid,
            BTreeMap<i64, (Vid, (ValueRange, Option<Vid>))>,
        > = Default::default();

        for ann_instr in self.annotated.values() {
            if matches!(
                ann_instr.instr,
                Instruction::Input(_) | Instruction::Equal(..)
            ) {
                continue;
            }

            let source_vid = ann_instr.source;
            let operand_vid = ann_instr.operand;
            let result_vid = ann_instr.result;
            let source_range = &self.values[&source_vid].range();
            let operand_range = &self.values[&operand_vid].range();

            match ann_instr.instr {
                Instruction::Add(_, _) => {
                    let tuples = [
                        (&source_vid, &operand_vid, operand_range),
                        (&operand_vid, &source_vid, source_range),
                    ];

                    for (input_vid, offset_source_vid, offset_range) in tuples {
                        if let Some(multipliers) = known_div_values.get(input_vid) {
                            let offset_multipliers: BTreeMap<_, _> = multipliers
                                .iter()
                                .filter_map(|(multiplier, (vid, (range, origin)))| {
                                    let new_range = range + offset_range;

                                    // If there's a prior origin, we can't capture both of them.
                                    // If there's no prior origin, it might be that
                                    // we've given up as above previously.
                                    // However, if there's no prior origin and the range is zero,
                                    // then this is the first remainder we are tracking.
                                    let new_origin = if origin.is_none() && range.is_exactly(0) {
                                        Some(*offset_source_vid)
                                    } else if offset_range.is_exactly(0) {
                                        // This addition is a no-op, propagate the prior origin.
                                        *origin
                                    } else {
                                        None
                                    };

                                    if new_range.start() < 0 {
                                        // The division will no longer be exact,
                                        // we can't represent this.
                                        None
                                    } else {
                                        Some((*multiplier, (*vid, (new_range, new_origin))))
                                    }
                                })
                                .collect();
                            known_div_values
                                .try_insert(result_vid, offset_multipliers)
                                .unwrap();
                        }
                    }
                }
                Instruction::Mul(_, _) => {
                    if self.values[&result_vid].range().is_exact() {
                        // The result value is known exactly, we won't learn anything new.
                        continue;
                    }
                    let (multiplier, vid_to_record) = if operand_range.is_exact() {
                        (operand_range.start(), source_vid)
                    } else if source_range.is_exact() {
                        (source_range.start(), operand_vid)
                    } else {
                        // Neither bound is known exactly, we can't figure out the multipliers.
                        continue;
                    };

                    if multiplier < 2 {
                        // 1 and 0 aren't worth implementing because other passes handle them.
                        // Haven't yet worked out whether / how this works with negative numbers.
                        continue;
                    }

                    let mut prior_values: BTreeMap<_, _> = known_div_values
                        .get(&vid_to_record)
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(value, (vid, (remainder, origin)))| {
                            let new_multiplier = value * multiplier;
                            let new_remainder = &remainder * multiplier;
                            (new_multiplier, (vid, (new_remainder, origin)))
                        })
                        .collect();
                    prior_values
                        .try_insert(
                            multiplier,
                            (vid_to_record, (ValueRange::new_exact(0), None)),
                        )
                        .unwrap();
                    known_div_values
                        .try_insert(result_vid, prior_values)
                        .unwrap();
                }
                Instruction::Div(_, _) => {
                    if self.values[&result_vid].range().is_exact() {
                        // The result value is known exactly, we won't learn anything new.
                        continue;
                    }
                    let (divisor, divided_vid) = if operand_range.is_exact() {
                        (operand_range.start(), source_vid)
                    } else if source_range.is_exact() {
                        (source_range.start(), operand_vid)
                    } else {
                        // Neither bound is known exactly, we can't match it to a mul operation.
                        continue;
                    };

                    if divisor < 2 {
                        // 1 and 0 aren't worth implementing because other passes handle them.
                        // Haven't yet worked out whether / how this works with negative numbers.
                        continue;
                    }

                    if let Some((equivalent_vid, (remainder, _))) = known_div_values
                        .get(&divided_vid)
                        .map(|multipliers| multipliers.get(&divisor))
                        .flatten()
                    {
                        // Ensure that the non-multiplied portion will get truncated out,
                        // leaving only the original, pre-multiplication value.
                        if remainder.end() < divisor {
                            // Success! We found that divided_vid / divisor == equivalent_vid.
                            self.equivalent_values.update_equivalent_values(
                                &mut self.values,
                                *equivalent_vid,
                                result_vid,
                            );
                        }
                    }
                }
                Instruction::Mod(_, _) => {
                    if self.values[&result_vid].range().is_exact() {
                        // The result value is known exactly, we won't learn anything new.
                        continue;
                    }
                    let (divisor, divided_vid) = if operand_range.is_exact() {
                        (operand_range.start(), source_vid)
                    } else if source_range.is_exact() {
                        (source_range.start(), operand_vid)
                    } else {
                        // Neither bound is known exactly, we can't match it to a mul operation.
                        continue;
                    };

                    if divisor < 2 {
                        // 1 isn't worth implementing because other passes handle it.
                        // The divisor (modulus) should never be 0 or lower, because that's UB.
                        continue;
                    }

                    // Ensure that we've been able to track where this remainder came from.
                    if let Some((_, (remainder, Some(remainder_origin)))) = known_div_values
                        .get(&divided_vid)
                        .map(|multipliers| multipliers.get(&divisor))
                        .flatten()
                    {
                        // Ensure that the remainder is smaller than the modulus,
                        // making the mod a no-op and leaving the remainder value.
                        if remainder.end() < divisor {
                            // Success! We found that divided_vid / divisor == equivalent_vid.
                            self.equivalent_values.update_equivalent_values(
                                &mut self.values,
                                *remainder_origin,
                                result_vid,
                            );
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        self
    }

    /// Any register whose value gets clobbered without needing to be read first is
    /// functionally "dead" and might as well have an undefined value.
    /// Example instructions that clobber registers without reading them:
    ///   inp x
    ///   mul x 0
    /// Since this pass leaves registers in the undefined value (which has no range info, etc.),
    /// it leaves the Self object in a state where not all other passes can be run anymore.
    /// As a result, this should probably be one of the last passes to be executed.
    ///
    /// It does not make sense to run this pass more than once, since it is idempotent.
    pub fn unused_register_elimination(mut self) -> Self {
        // Everything but the z register is unused after the last instruction executes.
        let mut register_is_unused = [true, true, true, false];

        let undefined_vid = Vid(self.values.0.keys().last().unwrap().0 + 1);
        self.values
            .try_insert(undefined_vid, Value::Undefined)
            .unwrap();

        for ann_instr in self.annotated.values().rev() {
            let state_after = ann_instr.state_after;

            for (is_unused, register_value) in register_is_unused.iter().zip(
                self.register_states
                    .get_mut(&state_after)
                    .unwrap()
                    .registers
                    .iter_mut(),
            ) {
                if *is_unused {
                    *register_value = undefined_vid;
                }
            }

            match ann_instr.instr {
                Instruction::Input(Register(n)) => {
                    register_is_unused[n] = true;
                }
                Instruction::Mul(Register(n), operand) => {
                    match operand {
                        Operand::Literal(_) => {}
                        Operand::Register(Register(r)) => {
                            register_is_unused[r] = false;
                        }
                    }

                    let operand_range = &self.values[&ann_instr.operand].range();
                    if operand_range.is_exactly(0) {
                        // We're multiplying by 0, so the source register's value doesn't matter
                        // since it gets overwritten by 0 regardless.
                        register_is_unused[n] = true;
                    } else {
                        register_is_unused[n] = false;
                    }
                }
                Instruction::Add(Register(n), operand)
                | Instruction::Div(Register(n), operand)
                | Instruction::Mod(Register(n), operand)
                | Instruction::Equal(Register(n), operand) => {
                    match operand {
                        Operand::Literal(_) => {}
                        Operand::Register(Register(r)) => {
                            register_is_unused[r] = false;
                        }
                    }
                    register_is_unused[n] = false;
                }
            }
        }

        self
    }

    /// An analysis pass made to work after unused_register_elimination() and prune results
    /// that never get used.
    ///
    /// Two ways that can happen:
    /// 1. An instruction produces a result that is never used. In that case, this pass:
    ///   - prunes the instruction
    ///   - propagates the source register's Undefined state upward
    ///   - if the operand was a register and was unused after that instruction, it propagates
    ///     that register's Undefined state upward as well
    /// 2. An instruction that already got pruned by a prior pass has a register operand that
    ///    becomes unused immediately after that instruction. Since the instruction got pruned,
    ///    that register operand is unused in that instruction as well. In that case, this pass
    ///    propagates that register's Undefined state upward.
    pub fn unused_result_elimination(mut self) -> Self {
        let mut overwrite_unused: [Option<Vid>; 4] = [None, None, None, None];

        for ann_instr in self.annotated.values().rev() {
            let state_after = self
                .register_states
                .get_mut(&ann_instr.state_after)
                .unwrap();
            for (maybe_vid, register_value) in overwrite_unused
                .iter()
                .zip(state_after.registers.iter_mut())
            {
                if let Some(undefined_vid) = maybe_vid {
                    *register_value = *undefined_vid;
                }
            }

            let registers_after = &state_after.registers;

            // If this instruction has already been pruned by a prior pass,
            // and it was the only reason an operand register was still live,
            // then propagate the operand register's Undefined value.
            if self.pruned.contains_key(&ann_instr.id) {
                if let Some(Operand::Register(Register(r))) = ann_instr.instr.operand() {
                    let operand_vid = registers_after[r];
                    if matches!(self.values[&operand_vid], Value::Undefined) {
                        overwrite_unused[r] = Some(operand_vid);
                    }
                }
            }

            match ann_instr.instr {
                Instruction::Input(_) => {}
                Instruction::Mul(Register(n), operand)
                | Instruction::Add(Register(n), operand)
                | Instruction::Div(Register(n), operand)
                | Instruction::Mod(Register(n), operand)
                | Instruction::Equal(Register(n), operand) => {
                    let result_vid = registers_after[n];
                    if matches!(self.values[&result_vid], Value::Undefined) {
                        // The result of this operation is never used.
                        // We prune this operation, if it wasn't pruned already.
                        self.pruned
                            .entry(ann_instr.id)
                            .or_insert(PrunedReason::ResultNeverUsed);

                        overwrite_unused[n] = Some(result_vid);

                        match operand {
                            Operand::Literal(_) => {}
                            Operand::Register(Register(r)) => {
                                let after_op_operand_vid = registers_after[r];
                                if matches!(self.values[&after_op_operand_vid], Value::Undefined) {
                                    // The operand register of this operation was only defined
                                    // so it could be used in this operation. Since it's not being
                                    // used after all, we make it undefined.
                                    overwrite_unused[r] = Some(after_op_operand_vid);
                                }
                            }
                        }
                    } else if !self.pruned.contains_key(&ann_instr.id) {
                        // Unpruned instructions consume their source and operand values.
                        // Pruned instructions do not consume them since pruned instructions
                        // will be removed.
                        if let Some(Operand::Register(Register(r))) = ann_instr.instr.operand() {
                            overwrite_unused[r] = None;
                        }
                        overwrite_unused[n] = None;
                    }
                }
            }
        }

        self
    }

    /// After pruning instructions, it's possible that we've ended up with the program's first use
    /// of a register being a clearing instruction like "mul x 0". These may have been necessary
    /// prior to pruning (to clear the state left over from now-pruned instructions), but are
    /// no longer necessary since at the time of first use, each register's value is zero.
    pub fn prune_unnecessary_register_clears(mut self) -> Analysis {
        let mut seen_registers = [false; 4];
        for ann_instr in self.annotated.values() {
            let register = ann_instr.instr.destination().0;
            if self.pruned.contains_key(&ann_instr.id) || seen_registers[register] {
                continue;
            }

            match ann_instr.instr {
                Instruction::Mul(_, _) => {
                    let operand_vid = ann_instr.operand;
                    let operand_range = self.values[&operand_vid].range();
                    if operand_range.is_exactly(0) {
                        self.pruned.try_insert(ann_instr.id, PrunedReason::UnnecessaryClearOnFirstUse).unwrap();
                    }
                }
                Instruction::Input(_) |
                Instruction::Add(_, _) |
                Instruction::Div(_, _) |
                Instruction::Mod(_, _) |
                Instruction::Equal(_, _) => {}
            }

            seen_registers[register] = true;
            if seen_registers.iter().all(|x| *x) {
                // We've seen the first use of all registers, we can stop looking
                // for unnecessary initializations to 0.
                break;
            }
        }

        self
    }

    pub fn get_optimized_instructions(self) -> Vec<Instruction> {
        self.annotated
            .into_iter()
            .filter_map(|(key, annotated)| {
                if self.pruned.contains_key(&key) {
                    None
                } else {
                    Some(annotated.instr)
                }
            })
            .collect()
    }
}

fn perform_value_numbering(
    input_program: &BTreeMap<InstrId, Instruction>,
    annotated: &mut BTreeMap<InstrId, AnnotatedInstr>,
    values: &mut ValueRangeAnalysis,
    register_states: &mut BTreeMap<Rsid, RegisterState>,
    inputs: &mut Vec<Vid>,
) {
    let mut vid_maker = VidMaker::new();
    let mut rsid_maker = RsidMaker::new();

    let initial_register_vid = vid_maker.next().unwrap();
    let initial_register_value = Value::Exact(initial_register_vid, 0);
    values
        .try_insert(initial_register_vid, initial_register_value)
        .unwrap();

    let initial_state = RegisterState::new([initial_register_vid; 4]);
    register_states
        .try_insert(INITIAL_RSID, initial_state)
        .unwrap();

    let input_range = ValueRange::new(1, 9);

    let mut prior_state = initial_state;
    let mut prior_rsid = INITIAL_RSID;
    let mut prior_instr_id: Option<InstrId> = None;
    for (instr_id, instr) in input_program {
        let annotated_instr = match instr {
            Instruction::Input(Register(r)) => {
                let current_input = inputs.len();
                let result_vid = vid_maker.next().unwrap();
                let result_value = Value::Input(result_vid, current_input, input_range.clone());
                values.try_insert(result_vid, result_value).unwrap();
                inputs.push(result_vid);

                let next_rsid = rsid_maker.next().unwrap();
                let mut next_state = prior_state;
                next_state.registers[*r] = result_vid;
                register_states.try_insert(next_rsid, next_state).unwrap();

                AnnotatedInstr {
                    id: *instr_id,
                    follows: prior_instr_id,
                    instr: *instr,
                    source: Vid::UNDEFINED,
                    operand: Vid::UNDEFINED,
                    result: result_vid,
                    state_before: prior_rsid,
                    state_after: next_rsid,
                }
            }
            Instruction::Add(Register(r), operand)
            | Instruction::Mul(Register(r), operand)
            | Instruction::Div(Register(r), operand)
            | Instruction::Mod(Register(r), operand)
            | Instruction::Equal(Register(r), operand) => {
                let source_vid = prior_state.registers[*r];
                let operand_vid = match operand {
                    Operand::Literal(l) => {
                        let operand_vid = vid_maker.next().unwrap();
                        let operand_value = Value::Exact(operand_vid, *l);
                        values.try_insert(operand_vid, operand_value).unwrap();
                        operand_vid
                    }
                    Operand::Register(Register(reg)) => prior_state.registers[*reg],
                };

                let result_vid = vid_maker.next().unwrap();
                let result_value = Value::Unknown(result_vid, ValueRange::MAX);
                values.try_insert(result_vid, result_value).unwrap();

                let next_rsid = rsid_maker.next().unwrap();
                let mut next_state = prior_state;
                next_state.registers[*r] = result_vid;
                register_states.try_insert(next_rsid, next_state).unwrap();

                AnnotatedInstr {
                    id: *instr_id,
                    follows: prior_instr_id,
                    instr: *instr,
                    source: source_vid,
                    operand: operand_vid,
                    result: result_vid,
                    state_before: prior_rsid,
                    state_after: next_rsid,
                }
            }
        };

        prior_instr_id = Some(*instr_id);
        prior_rsid = annotated_instr.state_after;
        prior_state = register_states[&prior_rsid];

        annotated.try_insert(*instr_id, annotated_instr).unwrap();
    }
}
