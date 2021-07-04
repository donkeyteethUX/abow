use smallvec::SmallVec;
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
pub struct BoW(pub Vec<f32>);

/// A map from features to their corresponding nodes in the Vocabulary tree.
/// Each feature maps to several nodes, up to one for each level of the tree.
///
/// The direct index for `feature[i]` is `di = DirectIdx[i]` where
/// `di.len() <= l` (number of levels).
/// `di[j]` is the id of the node matching `feature[i]`
/// at level `j` in the Vocabulary tree.
pub type DirectIdx = Vec<IdPath>;

/// The path from the root to the leaf for a given feature.
/// Only 5 entries are stack allocated, so level > 5 would have poor performance.
type IdPath = SmallVec<[usize; 5]>;

impl BoW {
    /// Compute L1 norm between two BoW. (Used in Galvez (Eq 2)).
    pub fn l1(&self, other: &Self) -> f32 {
        let values = self.0.iter().zip(&other.0);
        1. - 0.5 * (values.fold(0., |a, (b, c)| a + (b - c).abs()))
    }
}

type BowResult<T> = std::result::Result<T, BowErr>;
#[derive(Error, Debug)]
pub enum BowErr {
    #[error("No Features Provided")]
    NoFeatures,
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

#[cfg(test)]
mod test {
    use super::*;
    use std::path::{Path, PathBuf};
    #[test]
    fn test_recall() {
        // Load existing vocabulary
        let features = all_kps_from_dir("data/train").unwrap();
        println!("Detected {} ORB keypoints.", features.len());

        for &k in &[6_usize, 8_usize, 10_usize] {
            for &l in &[3_usize, 4_usize, 5_usize] {
                for _ in 0..2 {
                    // Create vocabulary from features
                    let voc = Vocabulary::create(&features, k, l);
                    println!("Vocabulary: {:#?}", voc);

                    // Create BoW vectors from the test data. Save file name for demonstration.
                    let mut bows: Vec<(PathBuf, BoW)> = Vec::new();
                    for entry in Path::new("data/test").read_dir().expect("Error").flatten() {
                        let new_feat = load_img_get_kps(&entry.path()).unwrap();
                        bows.push((
                            entry.path(),
                            voc.transform(&new_feat).unwrap(),
                        ));
                    }

                    // sort the files just for nicer output
                    let num = |s: &str| -> usize {
                        let s = s.strip_suffix(".jpg").unwrap();
                        s.parse().unwrap()
                    };
                    bows.sort_by(|a, b| {
                        num(a.0.file_name().unwrap().to_str().unwrap())
                            .partial_cmp(&num(b.0.file_name().unwrap().to_str().unwrap()))
                            .unwrap()
                    });

                    let mut cost = 0;

                    // Compare a few images to the the while colection, using L1 norm
                    for (f1, bow1) in bows.iter().skip(12).take(158) {
                        let mut scores: Vec<(f32, usize, i32)> = Vec::new();
                        let reference = num(f1.file_name().unwrap().to_str().unwrap());

                        for (f2, bow2) in bows.iter() {
                            let d = bow1.l1(bow2);
                            let matched = num(f2.file_name().unwrap().to_str().unwrap());
                            let cost = i32::abs(matched as i32 - reference as i32);
                            scores.push((d, matched, cost));
                        }

                        // Print out the top 5 matches for each image
                        let base_cost = 0 + 1 + 1 + 2 + 2 + 3 + 3 + 4 + 4 + 5 + 5 + 6;

                        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        for m in scores[..12].iter() {
                            cost += m.2;
                        }
                        cost -= base_cost;
                    }

                    println!("k: {}, l: {}. Total Cost: {}", k, l, cost);
                }
            }
        }
    }
}
