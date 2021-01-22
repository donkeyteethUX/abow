/// Bag-of-Words representation of an image or descriptor set.
///
/// Index: word/leaf id in the vocabulary.
///
/// Value: total weight of that word in provided features.
pub type BoW = Vec<f32>;


pub type DirectIdx = Vec<Vec<usize>>;

/// Provides method(s) for computing the similarity score between bow vectors.
pub trait BoWTrait {
    fn l1(&self, other: &Self) -> f32;
    fn l2(&self, other: &Self) -> f32;
}

impl BoWTrait for BoW {
    /// Compute L1 norm between two bow. (Used in Galvez (Eq 2)).
    fn l1(&self, other: &Self) -> f32 {
        1. - 0.5
            * (self
                .iter()
                .zip(other)
                .fold(0., |a, (b, c)| a + (b - c).abs()))
    }

    /// Not sure if needed
    fn l2(&self, _other: &Self) -> f32 {
        unimplemented!()
    }
}
