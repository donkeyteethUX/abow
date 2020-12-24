#![feature(test)]
extern crate test;
use test::Bencher;

use abow::{load_img_get_kps, Vocabulary};

/// Benchmark for Vocabulary::transform()
#[bench]
fn transf(b: &mut Bencher) {
    let voc = Vocabulary::load("vocabs/test.voc").unwrap();
    let features = load_img_get_kps("../fbow/data/test/image_00508.jpg").unwrap();
    b.iter(|| {
        voc.transform(&features).unwrap();
    });
}
