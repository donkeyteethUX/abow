use abow::{all_kps_from_dir, vocab::Vocabulary};

fn main() {
    const LEVELS: usize = 3;

    // Extract orb descriptors from images
    let features = all_kps_from_dir("data/train").unwrap();
    println!("Detected {} ORB keypoints.", features.len());

    // Create vocabulary from features
    let voc = Vocabulary::<LEVELS>::create(&features, 9).unwrap();
    println!("\nVocabulary = {:#?}", voc);

    // Save vocab and load it again just for fun
    voc.save("vocabs/test.voc").unwrap();
    let loaded_voc = Vocabulary::load("vocabs/test.voc").unwrap();

    // Make sure save & load worked
    assert_eq!(voc, loaded_voc);
}
