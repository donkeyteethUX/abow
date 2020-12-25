use bincode;
use bitvec::{order::Msb0, view::BitView};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use std::{fmt, path::Path};

use crate::*;

#[derive(Serialize, Deserialize, PartialEq)]
/// Feature vocabulary built from a collection of image keypoint descriptors. Can be:
/// 1. Created.
/// 2. Saved to a file.
/// 3. Loaded from a file.
/// 4. Used to transform a new set of descriptors into a BoW.
pub struct Vocabulary {
    blocks: Vec<Block>,
    k: usize,
    l: usize,
    num_blocks: usize,
    num_leaves: usize,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Block {
    id: NodeId,
    children: Children,
}

#[derive(Serialize, Deserialize, PartialEq)]
struct Children {
    features: Vec<Desc>,
    weights: Vec<f32>,
    cluster_size: Vec<usize>,
    ids: Vec<NodeId>,
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq)]
enum NodeId {
    Block(usize),
    Leaf(usize),
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
            .field("Word/Leaf Nodes", &self.num_leaves)
            .field("Other Nodes", &self.num_blocks)
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
    /// Load an ABoW vocabulary from a file
    pub fn load<P: AsRef<Path>>(file: P) -> Result<Self> {
        let mut file = std::fs::File::open(file)?;
        let mut buffer: Vec<u8> = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buffer)?;
        Ok(bincode::deserialize(&buffer)?)
    }

    /// Save vocabulary to a file
    pub fn save<P: AsRef<Path>>(&self, file: P) -> Result<()> {
        let serialized = bincode::serialize(&self)?;
        let mut file = std::fs::File::create(file)?;
        std::io::Write::write_all(&mut file, &serialized)?;
        Ok(())
    }

    /// Transform a vector of binary descriptors into its bag of words
    /// representation with respect to the Vocabulary. Descriptor is l1 normalized.
    pub fn transform(&self, features: &Vec<Desc>) -> Result<BoW> {
        let mut bow: BoW = vec![0.; self.num_leaves];
        for feature in features {
            let mut block = &self.blocks[0];
            let mut best: (u8, usize) = (u8::MAX, 0);
            // traverse tree
            loop {
                for child in 0..block.children.ids.len() {
                    let child_feat = &block.children.features[child];
                    let d = Self::hamming(feature, child_feat);
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
        // Normalize BoW vector
        let sum: f32 = bow.iter().sum();
        if sum > 0. {
            let inv_sum = 1. / sum;
            for w in bow.iter_mut() {
                *w *= inv_sum;
            }
        }

        Ok(bow)
    }

    /// Build a vocabulary from a collection of descriptors.
    ///
    /// # Arguments
    /// k: Branching factor
    ///
    /// l: Max number of levels
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
                        let d = Self::hamming(c, f);
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
    /// Compute the mean of a collection of binary arrays (descriptors).
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

    #[inline]
    /// Hamming distance between two binary arrays (descriptors).
    fn hamming(x: &[u8], y: &[u8]) -> u8 {
        x.iter()
            .zip(y)
            .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones() as u8)
    }

    /// Provide the next NodeId, either leaf/word or block.
    fn next_node_id(&mut self, leaf: bool) -> NodeId {
        match leaf {
            true => {
                self.num_leaves += 1;
                NodeId::Leaf(self.num_leaves)
            }
            false => {
                self.num_blocks += 1;
                NodeId::Block(self.num_blocks)
            }
        }
    }
    fn empty(k: usize, l: usize) -> Self {
        Self {
            blocks: Vec::new(),
            k,
            l,
            num_blocks: 0,
            num_leaves: 0,
        }
    }
}
