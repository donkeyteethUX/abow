pub mod opencv_utils;
pub mod vocab;

pub type Desc = [u8; 32];

/// Map from word id to total weight of that word in provided features
pub type BoW = Vec<f32>;

pub trait Similarity {
    fn l1(&self, other: &Self) -> f32;
    fn l2(&self, other: &Self) -> f32;
}

impl Similarity for BoW {
    /// L1 norm used in Galvez (Eq 2)
    fn l1(&self, other: &Self) -> f32 {
        1. - 0.5
            * (self
                .iter()
                .zip(other)
                .fold(0., |a, (b, c)| a + (b - c).abs()))
    }

    fn l2(&self, _other: &Self) -> f32 {
        todo!()
    }
}

#[inline]
pub fn hamming(x: &[u8], y: &[u8]) -> u8 {
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones() as u8)
}
