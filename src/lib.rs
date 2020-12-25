use thiserror::Error;

pub mod bow;
pub mod opencv_utils;
pub mod vocab;

pub use bow::{BoW, BoWTrait};
#[cfg(feature = "opencv")]
pub use opencv_utils::{all_kps_from_dir, load_img_get_kps};
pub use vocab::Vocabulary;

/// Supported descriptor type is 32-bit binary array.
///
/// This is the most commonly used keypoint descriptor data type.
/// It is used by ORB and BRIEF, for example.
///
/// In the future support can be added for other binary descriptor sizes.
pub type Desc = [u8; 32];

type Result<T> = std::result::Result<T, BowErr>;
#[derive(Error, Debug)]
pub enum BowErr {
    #[error("Io Error")]
    Io(#[from] std::io::Error),
    #[error("Vocabulary Serialization Error")]
    Bincode(#[from] bincode::Error),
    #[cfg(feature = "opencv")]
    #[error("Opencv Error")]
    OpenCvInternal(#[from] opencv::Error),
    #[cfg(feature = "opencv")]
    #[error("Opencv Descriptor decode error")]
    OpenCvDecode,
}
