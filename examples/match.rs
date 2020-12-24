use abow::opencv_utils;
use abow::vocab::Vocabulary;
use abow::BoW;
use abow::Similarity;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

fn main() {
    let mut file = std::fs::File::open("vocabs/test.voc").unwrap();
    let mut buffer = Vec::<u8>::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
    let voc: Vocabulary = bincode::deserialize(&buffer).unwrap();
    println!("Vocabulary: {:#?}", voc);

    let path = std::path::Path::new("data/test");
    let mut bows: Vec<(String, BoW)> = Vec::new();
    for entry in path.read_dir().expect("read_dir call failed") {
        if let Ok(entry) = entry {
            println!("{:?}", entry.path());
            let new_feat = crate::opencv_utils::load_img_get_kps(&entry.path());
            bows.push((
                entry
                    .path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
                voc.transform(&new_feat).unwrap(),
            ));
        }
    }
    for (f1, bow1) in bows.iter().take(100) {
        let mut sims: Vec<(f32, String)> = Vec::new();
        for (f2, bow2) in bows.iter() {
            let d = bow1.l1(bow2);
            sims.push((d, f2.clone()));
        }
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        sims = sims[..10].to_owned();

        println!("\nTop 10 Matches for {}:", f1);
        println!("Match      |      Score");
        for m in sims {
            println!("{} | {}", m.1, m.0);
        }
    }
}
