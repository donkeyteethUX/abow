#![cfg(feature = "useopencv")]
use crate::{BowErr, Desc, Result};
use opencv::{self, core::MatTrait, prelude::Feature2DTrait};
use std::{convert::TryInto, path::Path};

type CvImage = opencv::prelude::Mat;
type CvMat = opencv::core::Mat;

/// Extract orb keypoint descriptors from an image. Mostly for testing & example purposes.
fn orb_from_cvimage(cv_img: &CvImage) -> Result<Vec<Desc>> {
    // Create detector
    let mut orb = opencv::features2d::ORB::default().unwrap();

    // Detect keypoints and compute descriptors
    let mut kps = opencv::types::VectorOfKeyPoint::new();
    let mut desc = CvMat::default().unwrap();
    let mask = CvMat::default().unwrap();
    orb.detect_and_compute(cv_img, &mask, &mut kps, &mut desc, false)?;

    // Copy data from CvMat into descriptor buffer
    std::panic::catch_unwind(|| {
        (0..kps.len())
            .map(|i| {
                (0..32)
                    .map(|j| *desc.at_2d::<u8>(i as i32, j).unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect()
    })
    .map_err(|_| BowErr::OpenCvDecode)
}

/// Use opencv to load an image and extract orb keypoint descriptors.
pub fn load_img_get_kps<P: AsRef<Path>>(path: P) -> Result<Vec<Desc>> {
    let img: CvImage = opencv::imgcodecs::imread(
        &path.as_ref().to_str().unwrap(),
        opencv::imgcodecs::IMREAD_GRAYSCALE,
    )
    .unwrap();
    orb_from_cvimage(&img)
}

/// Extract orb keypoint descriptors from all images in directory using opencv.
pub fn all_kps_from_dir<P: AsRef<Path>>(path: P) -> Result<Vec<Desc>> {
    let mut features: Vec<Desc> = Vec::new();
    for entry in path.as_ref().read_dir()? {
        if let Ok(entry) = entry {
            println!("Extracting keypoint descriptors from {:?}", entry.path());
            features.extend(load_img_get_kps(&entry.path())?);
        }
    }
    Ok(features)
}
