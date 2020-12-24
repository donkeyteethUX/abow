use bincode;
use bitvec::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fmt, io, path::Path};
use thiserror::Error;

use crate::{hamming, BoW, Desc};

#[derive(Serialize, Deserialize, PartialEq)]
pub struct Vocabulary {
    pub(crate) blocks: Vec<Block>,
    pub(crate) k: usize,
    pub(crate) l: usize,
    pub(crate) next_block_id: usize,
    pub(crate) next_leaf_id: usize,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(crate) struct Block {
    pub(crate) id: NodeId,
    pub(crate) children: Children,
}

#[derive(Serialize, Deserialize, PartialEq)]
pub(crate) struct Children {
    pub(crate) features: Vec<Desc>,
    pub(crate) weights: Vec<f32>,
    pub(crate) cluster_size: Vec<usize>,
    pub(crate) ids: Vec<NodeId>,
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq)]
pub(crate) enum NodeId {
    Block(usize),
    Leaf(usize),
}
type Result<T> = std::result::Result<T, BowErr>;
#[derive(Error, Debug)]
pub enum BowErr {
    #[error("Io Error")]
    Io(#[from] io::Error),
    #[error("Vocabulary Serialization Error")]
    Bincode(#[from] bincode::Error),
    #[error("BoW error")]
    Unknown,
}

impl NodeId {
    fn get_bid(&self) -> usize {
        match self {
            NodeId::Block(i) => *i,
            NodeId::Leaf(_) => unreachable!(),
        }
    }
}

impl fmt::Debug for Children {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Children")
            .field("ids", &self.ids)
            .field("weights", &self.weights)
            .field("cluster size", &self.cluster_size)
            .finish()
    }
}

impl fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut clust_sizes: Vec<usize> = Vec::new();
        for b in self.blocks.iter() {
            for (i, &c) in b.children.cluster_size.iter().enumerate() {
                if matches!(b.children.ids[i], NodeId::Leaf(_)) {
                    clust_sizes.push(c);
                }
            }
        }
        let sum = clust_sizes.iter().sum::<usize>();
        f.debug_struct("Vocabulary")
            .field("Word/Leaf Nodes", &self.next_leaf_id)
            .field("Other Nodes", &self.next_block_id)
            .field("Levels", &self.l)
            .field("Branching Factor", &self.k)
            .field("Total Training Features", &sum)
            .field(
                "Min Word Cluster Size",
                &(clust_sizes.iter().min().unwrap()),
            )
            .field(
                "Max Word Cluster Size",
                &(clust_sizes.iter().max().unwrap()),
            )
            .field("Mean Word Cluster Size", &(sum / clust_sizes.len()))
            .finish()
    }
}

impl Vocabulary {
    /// Load an ABoW vocabulary from file
    pub fn load<P: AsRef<Path>>(file: P) -> Result<Self> {
        let mut file = std::fs::File::open(file)?;
        let mut buffer: Vec<u8> = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buffer)?;
        Ok(bincode::deserialize(&buffer)?)
    }

    /// Transform a vector of 32 bit binary descriptors into its bag of words
    /// representation with respect to the Vocabulary. Descriptor is l1 normalized.
    pub fn transform(&self, features: &Vec<Desc>) -> Result<BoW> {
        let mut bow: BoW = vec![0.; self.next_leaf_id];
        for feature in features {
            let mut block = &self.blocks[0];
            let mut best: (u8, usize) = (u8::MAX, 0);
            // traverse tree
            loop {
                for child in 0..block.children.ids.len() {
                    let child_feat = &block.children.features[child];
                    let d = hamming(feature, child_feat);
                    if d < best.0 {
                        best = (d, child)
                    }
                }
                match block.children.ids[best.1] {
                    NodeId::Block(id) => {
                        block = &self.blocks[id];
                    }
                    NodeId::Leaf(word_id) => {
                        // add word/leaf id and weight to result
                        let weight = block.children.weights[best.1];
                        match bow.get_mut(word_id - 1) {
                            Some(w) => *w += weight,
                            None => {
                                bow[word_id - 1] = weight;
                            }
                        }
                        break;
                    }
                }
                best = (u8::MAX, 0);
            }
        }
        // Normalize BoW
        let sum: f32 = bow.iter().sum();
        if sum > 0. {
            let inv_sum = 1. / sum;
            for w in bow.iter_mut() {
                *w *= inv_sum;
            }
        }
        Ok(bow)
    }

    pub fn create(features: &Vec<Desc>, k: usize, l: usize) -> Result<Self> {
        // Start with root of tree
        let mut v = Self::empty(k, l);

        // Build with recursive k-means clustering of features
        v.cluster(features, &NodeId::Block(0), 1);

        // Sort by block id
        v.blocks.sort_by(|a, b| a.id.get_bid().cmp(&b.id.get_bid()));

        Ok(v)
    }

    fn cluster(&mut self, features: &Vec<Desc>, parent_id: &NodeId, curr_level: usize) {
        println!(
            "KMeans step with {} features. parent id: {:?}, level {}",
            features.len(),
            parent_id,
            curr_level
        );
        if features.is_empty() {
            return;
        }
        let mut clusters: Vec<Desc> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();

        if features.len() <= self.k {
            // Only one feature per cluster
            for (i, f) in features.iter().enumerate() {
                clusters.push(*f);
                groups.push(vec![i]);
            }
        } else {
            // Proceed with kmeans clustering
            clusters = self.initialize_clusters(features);
            groups = vec![Vec::new(); self.k];

            loop {
                let mut new_groups: Vec<Vec<usize>> = vec![Vec::new(); self.k];
                for (i, f) in features.iter().enumerate() {
                    let mut best: (usize, u8) = (0, u8::MAX);
                    for (j, c) in clusters.iter().enumerate() {
                        let d = hamming(c, f);
                        if d < best.1 {
                            best = (j, d);
                        }
                    }
                    new_groups[best.0].push(i);
                }

                if groups == new_groups {
                    break; // converged
                }

                // update clusters
                clusters = new_groups
                    .iter()
                    .map(|group| {
                        let desc = group
                            .iter()
                            .map(|&i| features.get(i).unwrap())
                            .collect::<Vec<&Desc>>();
                        Self::desc_mean(desc)
                    })
                    .collect();
                groups = new_groups;
            }
        }

        // Create block
        let ids: Vec<_> = groups
            .iter()
            .map(|g| self.next_node_id(curr_level == self.l || g.len() == 1))
            .collect();

        let children = Children {
            weights: vec![1.; groups.len()],
            ids: ids.clone(),
            cluster_size: groups.iter().map(|g| g.len()).collect(),
            features: clusters,
        };
        let block = Block {
            id: *parent_id,
            children,
        };
        self.blocks.push(block);

        // Recurse
        if curr_level < self.l {
            for (i, id) in ids
                .iter()
                .filter(|&n| matches!(n, NodeId::Block(_)))
                .enumerate()
            {
                let features: Vec<Desc> = groups[i].iter().map(|&j| features[j]).collect();
                self.cluster(&features, id, curr_level + 1);
            }
        }
    }

    /// Initialize clusters for kmeans. Currently uses random initialization. todo: kmeans++
    fn initialize_clusters(&self, features: &Vec<Desc>) -> Vec<Desc> {
        let mut rng = thread_rng();
        features
            .as_slice()
            .choose_multiple(&mut rng, self.k)
            .cloned()
            .collect()
    }

    #[inline]
    fn desc_mean(descriptors: Vec<&Desc>) -> Desc {
        let n2 = descriptors.len() / 2;
        let mut counts = vec![0; std::mem::size_of::<Desc>() * 8];
        let mut result: Desc = [0; std::mem::size_of::<Desc>()];
        let result_bits = result.view_bits_mut::<Msb0>();
        for d in descriptors {
            for (i, b) in d.view_bits::<Msb0>().iter().enumerate() {
                if *b {
                    counts[i] += 1;
                }
            }
        }
        for (i, &c) in counts.iter().enumerate() {
            if c > n2 {
                result_bits.set(i, true);
            }
        }
        result
    }

    fn next_node_id(&mut self, leaf: bool) -> NodeId {
        match leaf {
            true => {
                self.next_leaf_id += 1;
                NodeId::Leaf(self.next_leaf_id)
            }
            false => {
                self.next_block_id += 1;
                NodeId::Block(self.next_block_id)
            }
        }
    }
    fn empty(k: usize, l: usize) -> Self {
        Self {
            blocks: Vec::new(),
            k,
            l,
            next_block_id: 0,
            next_leaf_id: 0,
        }
    }
}

#[cfg(test)]
#[cfg(feature = "useopencv")]
mod tests {

    use super::*;

    #[test]
    fn create() {
        let mut features: Vec<Desc> = Vec::new();
        let path = std::path::Path::new("../fbow/data");
        for entry in path.read_dir().expect("read_dir call failed") {
            if let Ok(entry) = entry {
                println!("{:?}", entry.path());
                features.extend(crate::opencv_utils::load_img_get_kps(&entry.path()));
            }
        }
        println!("Detected {} ORB keypoints.", features.len());
        let voc = Vocabulary::create(&features, 3, 3);
        println!("Vocabulary: {:#?}", voc);
    }
    #[test]
    fn create_and_transform() {
        let mut features: Vec<Desc> = Vec::new();
        let path = std::path::Path::new("../fbow/data");
        for entry in path.read_dir().expect("read_dir call failed") {
            if let Ok(entry) = entry {
                println!("{:?}", entry.path());
                features.extend(crate::opencv_utils::load_img_get_kps(&entry.path()));
            }
        }
        println!("Detected {} ORB keypoints.", features.len());
        let voc = Vocabulary::create(&features, 3, 3).unwrap();
        println!("Vocabulary: {:#?}", voc);
        let new_feat = crate::opencv_utils::load_img_get_kps(&std::path::PathBuf::from(
            "../fbow/data/test/image_00508.jpg",
        ));

        let t = voc.transform(&new_feat).unwrap();
        println!("Resulting bag-of-words representation:\n {:#?}", t);
    }

    #[test]
    fn create_and_save_and_load() {
        let mut features: Vec<Desc> = Vec::new();
        let path = std::path::Path::new("data");
        for entry in path.read_dir().expect("read_dir call failed") {
            if let Ok(entry) = entry {
                println!("{:?}", entry.path());
                features.extend(crate::opencv_utils::load_img_get_kps(&entry.path()));
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

    #[test]
    fn load_and_transform() {
        let mut file = std::fs::File::open("vocabs/test.voc").unwrap();
        let mut buffer = Vec::<u8>::new();
        std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
        let voc: Vocabulary = bincode::deserialize(&buffer).unwrap();

        println!("Vocabulary: {:#?}", voc);
        let new_feat = crate::opencv_utils::load_img_get_kps(&std::path::PathBuf::from(
            "../fbow/data/test/image_00508.jpg",
        ));

        let t = voc.transform(&new_feat);
        println!("Resulting bag-of-words representation:\n {:#?}", t);
    }
}
