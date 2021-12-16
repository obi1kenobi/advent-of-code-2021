use std::{env, fs};

#[allow(unused_imports)]
use itertools::Itertools;

use bitvec::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let part = reversed_args.pop().expect("part number");
    let input_file = reversed_args.pop().expect("input file");
    let content = fs::read_to_string(input_file).unwrap();
    let content = content.trim_end();
    assert_eq!(content.len() % 2, 0);

    let input_data: Vec<u8> = content
        .trim_end()
        .chars()
        .map(|x| x.to_digit(16).unwrap() as u8)
        .tuples()
        .map(|(a, b)| a * 16 + b)
        .collect();

    match part {
        "1" => {
            let result = solve_part1(&input_data);
            println!("{}", result);
        }
        "2" => {
            let result = solve_part2(&input_data);
            println!("{}", result);
        }
        _ => unreachable!("{}", part),
    }
}

#[allow(dead_code)]
struct Packet {
    version: u8,
    type_id: u8,

    data: PacketType,
}

impl Packet {
    fn read(bits: &BitSlice<Msb0, u8>) -> (Self, &BitSlice<Msb0, u8>) {
        let version = bits[0..3].load_be::<u8>();
        let type_id = bits[3..6].load_be::<u8>();

        let (data, remainder) = PacketType::read(&bits[3..]);

        (
            Self {
                version,
                type_id,
                data,
            },
            remainder,
        )
    }

    fn read_complete(bits: &BitSlice<Msb0, u8>) -> Self {
        let (packet, remainder) = Packet::read(bits);
        assert_eq!(remainder.count_ones(), 0);
        packet
    }

    fn iter(&self) -> PacketIterator<'_> {
        PacketIterator::new(self)
    }

    fn eval(&self) -> u64 {
        match &self.data {
            PacketType::SumOperator(packets) => {
                packets.iter().map(Packet::eval).sum()
            }
            PacketType::ProductOperator(packets) => {
                packets.iter().map(Packet::eval).product()
            }
            PacketType::MinOperator(packets) => {
                packets.iter().map(Packet::eval).min().unwrap()
            }
            PacketType::MaxOperator(packets) => {
                packets.iter().map(Packet::eval).max().unwrap()
            }
            PacketType::Literal(value) => *value,
            PacketType::GreaterThanOperator(left, right) => {
                if left.eval() > right.eval() {
                    1
                } else {
                    0
                }
            }
            PacketType::LessThanOperator(left, right) => {
                if left.eval() < right.eval() {
                    1
                } else {
                    0
                }
            }
            PacketType::EqualsOperator(left, right) => {
                if left.eval() == right.eval() {
                    1
                } else {
                    0
                }
            }
        }
    }
}

enum PacketType {
    SumOperator(Vec<Packet>),
    ProductOperator(Vec<Packet>),
    MinOperator(Vec<Packet>),
    MaxOperator(Vec<Packet>),
    Literal(u64),
    GreaterThanOperator(Box<Packet>, Box<Packet>),
    LessThanOperator(Box<Packet>, Box<Packet>),
    EqualsOperator(Box<Packet>, Box<Packet>),
}

impl PacketType {
    const SUM_PACKET_TYPE: u8 = 0;
    const PRODUCT_PACKET_TYPE: u8 = 1;
    const MIN_PACKET_TYPE: u8 = 2;
    const MAX_PACKET_TYPE: u8 = 3;
    const LITERAL_PACKET_TYPE: u8 = 4;
    const GREATER_THAN_PACKET_TYPE: u8 = 5;
    const LESS_THAN_PACKET_TYPE: u8 = 6;
    const EQUALS_PACKET_TYPE: u8 = 7;

    fn read(bits: &BitSlice<Msb0, u8>) -> (Self, &BitSlice<Msb0, u8>) {
        let (type_id_bits, bits) = bits.split_at(3);
        let type_id = type_id_bits.load_be::<u8>();

        match type_id {
            Self::SUM_PACKET_TYPE => {
                let (subpackets, remainder) = Self::read_subpackets(bits);
                (Self::SumOperator(subpackets), remainder)
            }
            Self::PRODUCT_PACKET_TYPE => {
                let (subpackets, remainder) = Self::read_subpackets(bits);
                (Self::ProductOperator(subpackets), remainder)
            }
            Self::MIN_PACKET_TYPE => {
                let (subpackets, remainder) = Self::read_subpackets(bits);
                (Self::MinOperator(subpackets), remainder)
            }
            Self::MAX_PACKET_TYPE => {
                let (subpackets, remainder) = Self::read_subpackets(bits);
                (Self::MaxOperator(subpackets), remainder)
            }
            Self::LITERAL_PACKET_TYPE => {
                let (literal, remainder) = Self::read_literal_content(bits);
                (Self::Literal(literal), remainder)
            }
            Self::GREATER_THAN_PACKET_TYPE => {
                let (left, right, remainder) = Self::read_exactly_two_subpackets(bits);
                (Self::GreaterThanOperator(left, right), remainder)
            }
            Self::LESS_THAN_PACKET_TYPE => {
                let (left, right, remainder) = Self::read_exactly_two_subpackets(bits);
                (Self::LessThanOperator(left, right), remainder)
            }
            Self::EQUALS_PACKET_TYPE => {
                let (left, right, remainder) = Self::read_exactly_two_subpackets(bits);
                (Self::EqualsOperator(left, right), remainder)
            }
            _ => unreachable!(),
        }
    }

    fn read_literal_content(bits: &BitSlice<Msb0, u8>) -> (u64, &BitSlice<Msb0, u8>) {
        let mut number = 0u64;
        let mut remaining_bits = bits;

        loop {
            let (block, next_bits) = remaining_bits.split_at(5);
            let (flag, content) = block.split_first().unwrap();

            remaining_bits = next_bits;

            let value: u8 = content.load_be();
            number = (number << 4) + value as u64;

            if !flag {
                break;
            }
        }

        (number, remaining_bits)
    }

    fn read_exactly_two_subpackets(bits: &BitSlice<Msb0, u8>) -> (Box<Packet>, Box<Packet>, &BitSlice<Msb0, u8>) {
        let (mut subpackets, remainder) = Self::read_subpackets(bits);
        assert_eq!(subpackets.len(), 2);
        let mut drain = subpackets.drain(..);
        let left = Box::new(drain.next().unwrap());
        let right = Box::new(drain.next().unwrap());

        (left, right, remainder)
    }

    fn read_subpackets(bits: &BitSlice<Msb0, u8>) -> (Vec<Packet>, &BitSlice<Msb0, u8>) {
        let (length_type_id, bits) = bits.split_first().unwrap();

        let (subpackets, remaining_bits) = if length_type_id == false {
            // length-based subpacket definition, next 15 bits
            let (subpackets_length_bits, remaining_bits) = bits.split_at(15);
            let subpackets_length = subpackets_length_bits.load_be::<usize>();

            let (mut subpacket_bits, remaining_bits) = remaining_bits.split_at(subpackets_length);
            let mut subpackets = Vec::new();
            while !subpacket_bits.is_empty() {
                let (packet, next_packet_bits) = Packet::read(subpacket_bits);
                subpackets.push(packet);
                subpacket_bits = next_packet_bits;
            }

            (subpackets, remaining_bits)
        } else {
            // count-based subpacket definition, next 11 bits
            let (subpackets_count_bits, mut remaining_bits) = bits.split_at(11);
            let subpackets_count = subpackets_count_bits.load_be::<usize>();

            let mut subpackets = Vec::with_capacity(subpackets_count);
            for _ in 0..subpackets_count {
                let (packet, next_packet_bits) = Packet::read(remaining_bits);
                subpackets.push(packet);
                remaining_bits = next_packet_bits;
            }

            (subpackets, remaining_bits)
        };

        (
            subpackets,
            remaining_bits,
        )
    }
}

struct PacketIterator<'a> {
    stack: Vec<&'a Packet>,
}

impl<'a> PacketIterator<'a> {
    fn new(start: &'a Packet) -> Self {
        Self { stack: vec![start] }
    }
}

impl<'a> Iterator for PacketIterator<'a> {
    type Item = &'a Packet;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(packet) = self.stack.pop() {
            match &packet.data {
                PacketType::SumOperator(subpackets)
                | PacketType::ProductOperator(subpackets)
                | PacketType::MinOperator(subpackets)
                | PacketType::MaxOperator(subpackets) => {
                    self.stack.extend(subpackets.iter());
                },
                PacketType::Literal(_) => {},
                PacketType::GreaterThanOperator(left, right)
                | PacketType::LessThanOperator(left, right)
                | PacketType::EqualsOperator(left, right) => {
                    self.stack.push(left);
                    self.stack.push(right);
                }
            }
            Some(packet)
        } else {
            None
        }
    }
}

#[allow(unused_variables)]
fn solve_part1(data: &[u8]) -> u64 {
    let bits = BitSlice::<Msb0, _>::from_slice(data).unwrap();

    let packet = Packet::read_complete(bits);

    packet.iter().map(|p| p.version as u64).sum()
}

#[allow(unused_variables)]
fn solve_part2(data: &[u8]) -> u64 {
    let bits = BitSlice::<Msb0, _>::from_slice(data).unwrap();

    let packet = Packet::read_complete(bits);

    packet.eval()
}
