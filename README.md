# A Bag of Words
A rust crate for converting collections of image feature descriptors into a "Bag-of-Words" representation for fast matching of images in localizaton / SLAM systems. Uses hierarchical k-means clustering to create a "vocabulary" of common visual features. The vocabulary can then be used to transform an arbitrary image or collection of image keypoint descriptors into a compact bag of words (bow) vector. Bow vectors can be matched very quickly to give a measure of image similarity.

## Setup
This crate is primarily designed for use with user-provided keypoint descriptors. Currently, 32-bit binary descriptors are supported (ORB or BRIEF are popular examples). However this crate does provide convenience functions to compute ORB descriptors from images, using [opencv](https://github.com/opencv/opencv) and [opencv-rust](https://github.com/twistedfall/opencv-rust/).

These functions can be enabled or disabled using the feature flag "opencv". This feature is enabled by default, so if you don't want to use opencv, update your Cargo.toml with:
```toml
abow = {version = "0.1", default-features = false}
```
Otherwise, you'll need to install OpenCV. Instuctions and troubleshooting are available at https://github.com/twistedfall/opencv-rust.

## Executable Examples
Create a descriptor vocabulary from a set of images and save it:
```console
foo@bar:~/repos/ABoW$ cargo run --release --example create-voc

Vocabulary = Vocabulary {
    Word/Leaf Nodes: 3125,
    Other Nodes: 780,
    Levels: 5,
    Branching Factor: 5,
    Total Training Features: 131376,
    Min Word Cluster Size: 1,
    Max Word Cluster Size: 373,
    Mean Word Cluster Size: 42,
}
```
Load a vocabulary, transform a sequence of images into BoW, and compute best matches between them:
```console
foo@bar:~/repos/ABoW$ cargo run --release --example match

Top 5 Matches for "100.jpg":
Match     | Score
"100.jpg" | 1.0
"102.jpg" | 0.4220034
"101.jpg" | 0.4040035
"98.jpg"  | 0.3740036
"99.jpg"  | 0.37200385
```

## References
This library is largely based on the C++ repositories [DBoW2](https://github.com/dorian3d/DBoW2/) and [fbow](https://github.com/rmsalinas/fbow).

