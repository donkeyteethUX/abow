#![feature(test)]
extern crate test;
use test::Bencher;

use abow::vocab::Vocabulary;
use abow::{opencv_utils, Desc};
use std::path::PathBuf;

/// Benchmark for Vocabulary::transform()
#[bench]
fn transf(b: &mut Bencher) {
    let voc = Vocabulary::load("vocabs/test.voc").unwrap();
    let features =
        opencv_utils::load_img_get_kps(&PathBuf::from("../fbow/data/test/image_00508.jpg"));
    b.iter(|| {
        voc.transform(&features);
    });
}
