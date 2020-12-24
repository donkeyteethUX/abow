use abow::{vocab::Vocabulary, Desc};

fn main() {
    let mut features: Vec<Desc> = Vec::new();
    let path = std::path::Path::new("data");
    for entry in path.read_dir().expect("read_dir call failed") {
        if let Ok(entry) = entry {
            println!("{:?}", entry.path());
            features.extend(abow::opencv_utils::load_img_get_kps(&entry.path()));
        }
    }
    println!("Detected {} ORB keypoints.", features.len());
    let voc = Vocabulary::create(&features, 5, 5).unwrap();

    {
        let serialized = bincode::serialize(&voc).unwrap();
        let mut file = std::fs::File::create("vocabs/test.voc").unwrap();
        std::io::Write::write_all(&mut file, &serialized).unwrap();
    }

    let mut file = std::fs::File::open("vocabs/test.voc").unwrap();
    let mut buffer = Vec::<u8>::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
    let deserialized: Vocabulary = bincode::deserialize(&buffer).unwrap();

    println!("\nDeserialized = {:#?}", deserialized);

    assert_eq!(voc, deserialized);
}
