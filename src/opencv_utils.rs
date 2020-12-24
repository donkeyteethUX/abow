#![cfg(feature = "useopencv")]
use io::Result;
use opencv::{self, core::MatTrait, prelude::Feature2DTrait};
use serde::{Deserialize, Serialize};
use std::{convert::TryInto, path::PathBuf};
use std::{
    fs::File,
    io::{self, prelude::*},
    path::Path,
};

use crate::{vocab::*, Desc};

type CvImage = opencv::prelude::Mat;

fn orb_from_cvimage(cv_img: &CvImage) -> Vec<[u8; 32]> {
    // Create detector
    let mut orb = opencv::features2d::ORB::default().unwrap();

    // Detect keypoints and compute descriptors
    let mut kps = opencv::types::VectorOfKeyPoint::new();
    let mut desc = opencv::core::Mat::default().unwrap();
    let mask = opencv::core::Mat::default().unwrap();
    orb.detect_and_compute(cv_img, &mask, &mut kps, &mut desc, false)
        .expect("Error in detection");

    (0..kps.len())
        .map(|i| {
            (0..32)
                .map(|j| *desc.at_2d::<u8>(i as i32, j).unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect()
}

pub fn load_img_get_kps(path: &PathBuf) -> Vec<[u8; 32]> {
    let img: CvImage =
        opencv::imgcodecs::imread(path.to_str().unwrap(), opencv::imgcodecs::IMREAD_GRAYSCALE)
            .unwrap();
    orb_from_cvimage(&img)
}
