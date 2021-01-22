use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use abow::*;

fn main() {
    // Load existing vocabulary
    let voc = Vocabulary::load("vocabs/test.voc").unwrap();
    println!("Vocabulary: {:#?}", voc);

    // Create BoW vectors from the test data. Save file name for demonstration.
    let mut bows: Vec<(PathBuf, BoW)> = Vec::new();
    for entry in Path::new("data/test").read_dir().expect("Error").take(6) {
        if let Ok(entry) = entry {
            let new_feat = load_img_get_kps(&entry.path()).unwrap();
            bows.push((entry.path(), voc.transform_with_direct_idx(&new_feat).unwrap().0));
        }
    }

    // Match every image to every other image
    for (f1, bow1) in bows.iter() {
        let mut scores: Vec<(f32, &OsStr)> = Vec::new();
        for (f2, bow2) in bows.iter() {
            let d = bow1.l1(bow2);
            scores.push((d, f2.file_name().unwrap()));
        }

        // Print out the top 5 matches for each image
        println!("\nTop 5 Matches for {:#?}:", f1.file_name().unwrap());
        println!("Match      |      Score");
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for m in scores[..5].iter() {
            println!("{:#?} | {:#?}", m.1, m.0);
        }
    }
}
