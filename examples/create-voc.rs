use abow::{all_kps_from_dir, vocab::Vocabulary};

fn main() {

    // Extract orb descriptors from images
    let features = all_kps_from_dir("data/train").unwrap();
    println!("Detected {} ORB features.", features.len());

    // Create vocabulary from features
    let voc = Vocabulary::create(&features, 9, 3);
    println!("\nVocabulary = {:#?}", voc);

    // Save vocab and load it again just for fun
    voc.save("vocabs/test.voc").unwrap();
    let loaded_voc = Vocabulary::load("vocabs/test.voc").unwrap();

    // Make sure save & load worked
    assert_eq!(voc, loaded_voc);
}
