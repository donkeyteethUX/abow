/// Index is word/leaf id. Value is total weight of that word in provided features.
pub type BoW = Vec<f32>;

pub trait BoWTrait {
    fn l1(&self, other: &Self) -> f32;
    fn l2(&self, other: &Self) -> f32;
    fn l1_normalize(&mut self);
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

    /// Normalize a BoW vector
    fn l1_normalize(&mut self) {
        let sum: f32 = self.iter().sum();
        if sum > 0. {
            let inv_sum = 1. / sum;
            for w in self.iter_mut() {
                *w *= inv_sum;
            }
        }
    }
}
