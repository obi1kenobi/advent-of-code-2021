use std::{env, fs, sync::Arc};

use lazy_static::lazy_static;
use reqwest::{Url, cookie::Jar};


lazy_static! {
    static ref SESSION_ID: &'static str = include_str!("../.session");
    static ref URL: Url = "https://adventofcode.com/2021/".parse::<Url>().unwrap();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut reversed_args: Vec<_> = args.iter().map(|x| x.as_str()).rev().collect();

    reversed_args
        .pop()
        .expect("Expected the executable name to be the first argument, but was missing");

    let jar = Arc::new(Jar::default());
    jar.add_cookie_str(&format!("session={}", *SESSION_ID), &URL);

    let day = reversed_args.pop().expect("day").parse::<usize>().unwrap();
    let command = reversed_args.pop().expect("command");
    match command {
        "in" => {
            let input_file_url = URL.join(&format!("day/{}/input", day)).unwrap();
            let output_location = reversed_args.pop().expect("output file");

            let client = reqwest::blocking::Client::builder()
                .cookie_provider(jar)
                .build()
                .unwrap();
            let mut result = client.get(input_file_url.as_str())
                .send()
                .unwrap();

            assert!(result.status().is_success());
            let mut output = fs::File::create(output_location).unwrap();
            result.copy_to(&mut output).unwrap();
        }
        _ => unreachable!("{}", command),
    }
}
