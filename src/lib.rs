use smallvec;
use thiserror::Error;

/// Implementation of a visual bag-of-words vocabulary,
/// which provides the main functionality of this create.
pub mod vocab;
pub use vocab::Vocabulary;

/// Utilities for extracting keypoint descriptors using opencv.
pub mod opencv_utils;
#[cfg(feature = "opencv")]
pub use opencv_utils::*;

/// Supported descriptor type is 32-bit binary array.
///
/// This is the most commonly used keypoint descriptor data type.
/// It is used by ORB and BRIEF, for example.
///
/// In the future support can be added for other binary descriptor sizes.
pub type Desc = [u8; 32];

/// Bag-of-Words representation of an image or descriptor set.
///
/// Index: word/leaf id in the vocabulary.
///
/// Value: total weight of that word in provided features.
pub type BoW = Vec<f32>;

/// This type represents a map from features to their corresponding nodes in the Vocabulary tree.
///
/// The direct index for `feature[i]` is `di = DirectIdx[i]` where
/// `di.len() <= l` (number of levels), and `di[j]` is the id of the node matching `feature[i]`
/// at level `j` in the Vocabulary tree.
pub type DirectIdx<const L: usize> = Vec<smallvec::SmallVec<[usize; L]>>;

/// Provides method(s) for computing the similarity score between bow vectors.
pub trait BoWTrait {
    /// Compute L1 norm between two BoW. (Used in Galvez (Eq 2)).
    fn l1(&self, other: &Self) -> f32;
    fn l2(&self, other: &Self) -> f32;
}

impl BoWTrait for BoW {
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

type BowResult<T> = std::result::Result<T, BowErr>;
#[derive(Error, Debug)]
pub enum BowErr {
    #[error("Io Error")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "bincode")]
    #[error("Vocabulary Serialization Error")]
    Bincode(#[from] bincode::Error),
    #[cfg(feature = "opencv")]
    #[error("Opencv Error")]
    OpenCvInternal(#[from] opencv::Error),
    #[cfg(feature = "opencv")]
    #[error("Opencv Descriptor decode error")]
    OpenCvDecode,
}
