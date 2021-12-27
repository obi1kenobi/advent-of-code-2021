use std::collections::BTreeMap;

use crate::parser::{Instruction, Operand, Register};

use self::values::{Value, ValueRange, Vid, VidMaker};

mod values;

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
struct RegisterState {
    registers: [Vid; 4],
}

impl RegisterState {
    #[inline]
    pub fn new(registers: [Vid; 4]) -> Self {
        Self { registers }
    }
}

#[derive(Debug, Clone)]
struct AnnotatedInstr {
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

#[derive(Debug, Clone)]
struct Analysis {
    pub input_program: BTreeMap<InstrId, Instruction>,
    pub annotated: BTreeMap<InstrId, AnnotatedInstr>,
    pub values: BTreeMap<Vid, Value>,
    pub register_states: BTreeMap<Rsid, RegisterState>,
    pub inputs: Vec<Vid>, // the value IDs of all input instructions
}

impl From<Vec<Instruction>> for Analysis {
    fn from(instrs: Vec<Instruction>) -> Self {
        let input_program: BTreeMap<InstrId, Instruction> = instrs
            .iter()
            .enumerate()
            .map(|(idx, instr)| (InstrId(idx), *instr))
            .collect();

        let mut annotated: BTreeMap<InstrId, AnnotatedInstr> = Default::default();
        let mut values: BTreeMap<Vid, Value> = Default::default();
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
            values,
            register_states,
            inputs,
        }
    }
}

fn narrow_value_range(values: &mut BTreeMap<Vid, Value>, vid: Vid, range: &ValueRange) -> ValueRange {
    let owned_value = values.remove(&vid).unwrap();
    let final_result = owned_value.narrow_range(range);
    let final_range = final_result.range().clone();
    values.try_insert(vid, final_result).unwrap();
    final_range
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
                narrow_value_range(&mut self.values, result_vid, &ValueRange::new_exact(result_value));
            }
        }

        self
    }

    pub fn operation_definedness(mut self) -> Self {
        for ann_instr in self.annotated.values() {
            match ann_instr.instr {
                Instruction::Input(_) |
                Instruction::Add(_, _) |
                Instruction::Mul(_, _) |
                Instruction::Div(_, _) => {
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
                },
                Instruction::Mod(_, _) => {
                    // The source must be non-negative, and the operand must be strictly positive.
                    // The result must be between zero and the operand value minus one.
                    narrow_value_range(&mut self.values, ann_instr.source, &ValueRange::new(0, i64::MAX));
                    let operand_range = narrow_value_range(&mut self.values, ann_instr.operand, &ValueRange::new(1, i64::MAX));
                    narrow_value_range(&mut self.values, ann_instr.result, &ValueRange::new(0, operand_range.end() - 1));
                }
                Instruction::Equal(_, _) => {
                    // The result must be either 0 or 1.
                    narrow_value_range(&mut self.values, ann_instr.result, &ValueRange::new(0, 1));
                }
            }
        }

        self
    }
}

fn perform_value_numbering(
    input_program: &BTreeMap<InstrId, Instruction>,
    annotated: &mut BTreeMap<InstrId, AnnotatedInstr>,
    values: &mut BTreeMap<Vid, Value>,
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
