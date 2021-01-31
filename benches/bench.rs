#![feature(test)]
extern crate test;
use test::Bencher;

use abow::{load_img_get_kps, Vocabulary};

/// Benchmark for Vocabulary::transform()
#[bench]
fn transf(b: &mut Bencher) {
    const LEVELS: usize = 3;

    let voc = Vocabulary::<LEVELS>::load("vocabs/test.voc").unwrap();
    let features = load_img_get_kps("data/test/0.jpg").unwrap();
    b.iter(|| {
        voc.transform(&features).unwrap();
    });
}

/// Benchmark for Vocabulary::transform_with_direct_idx()
#[bench]
fn transf_dir_idx(b: &mut Bencher) {
    const LEVELS: usize = 3;

    let voc = Vocabulary::<LEVELS>::load("vocabs/test.voc").unwrap();
    let features = load_img_get_kps("data/test/0.jpg").unwrap();
    b.iter(|| {
        voc.transform_with_direct_idx(&features).unwrap();
    });
}
