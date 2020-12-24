# A Bag of Words
A library for converting collections of image feature descriptors into a "Bag-of-Words" representation for fast matching of images in localizaton / SLAM systems. Uses heirarchical k-means clustering to create a "vocabulary" of common visual features. The vocabulary can then be used to transform an arbitrary image or collection of image keypoint descriptors into a compact bag of words (bow) vector. Bow vectors can be matched very quickly to give a measure of image similarity.

## References
This library is largely based on the C++ repositories DBoW2 and fbow:
https://github.com/dorian3d/DBoW2/
https://github.com/rmsalinas/fbow
