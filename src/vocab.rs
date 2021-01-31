#[cfg(feature = "bincode")]
use bincode;
use bitvec::{order::Msb0, view::BitView};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, ToSmallVec};
use std::fmt;

use crate::*;

#[derive(Serialize, Deserialize, PartialEq, Clone)]
/// Feature vocabulary built from a collection of image keypoint descriptors. Can be:
/// 1. Created.
/// 2. Saved to a file & loaded from a file (requires bincode feature, enabled by default).
/// 3. Used to transform a new set of descriptors into a BoW representation (and
///    optionally get DirectIndex from features to nodes).
pub struct Vocabulary<const L: usize> {
    blocks: Vec<Block<L>>,
    k: usize,
    num_blocks: usize,
    num_leaves: usize,
}

/// Vocabulary API
impl<const L: usize> Vocabulary<{ L }> {
    /// Transform a vector of binary descriptors into its bag of words
    /// representation with respect to the Vocabulary. Descriptor is l1 normalized.
    pub fn transform(&self, features: &Vec<Desc>) -> BowResult<BoW> {
        self.transform_generic(features, false).map(|(bow, _)| bow)
    }

    /// Transform a vector of binary descriptors into its bag of words
    /// representation with respect to the Vocabulary. Descriptor is l1 normalized.
    ///
    /// Also provides "direct index" from the features to their corresponding nodes in the Vocabulary tree.
    ///
    /// The direct index for `feature[i]` is `di = DirectIdx[i]` where
    /// `di.len() <= l` (number of levels), and `di[j]` is the id of the node matching `feature[i]`
    /// at level `j` in the Vocabulary tree.
    pub fn transform_with_direct_idx(
        &self,
        features: &Vec<Desc>,
    ) -> BowResult<(BoW, DirectIdx<L>)> {
        self.transform_generic(features, true)
    }

    /// Build a vocabulary from a collection of descriptors.
    ///
    /// Args: (k: Branching factor, l: Max number of levels)
    pub fn create(features: &Vec<Desc>, k: usize) -> BowResult<Self> {
        // Start with root of tree
        let mut v = Self::empty(k);

        // Build with recursive k-means clustering of features
        v.cluster(features, vec![0], 1);

        // Sort by block id
        v.blocks.sort_by(|a, b| a.id.get_bid().cmp(&b.id.get_bid()));

        Ok(v)
    }

    /// Load an ABoW vocabulary from a file
    #[cfg(feature = "bincode")]
    pub fn load<P: AsRef<std::path::Path>>(file: P) -> BowResult<Self> {
        let mut file = std::fs::File::open(file)?;
        let mut buffer: Vec<u8> = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buffer)?;
        Ok(bincode::deserialize(&buffer)?)
    }

    /// Save vocabulary to a file
    #[cfg(feature = "bincode")]
    pub fn save<P: AsRef<std::path::Path>>(&self, file: P) -> BowResult<()> {
        let serialized = bincode::serialize(&self)?;
        let mut file = std::fs::File::create(file)?;
        std::io::Write::write_all(&mut file, &serialized)?;
        Ok(())
    }
}

/////////////////////                Helpers                 ////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
/// A unit representing a non-leaf node in the vocabulary
struct Block<const L: usize> {
    id: NodeId<L>,
    children: Children<L>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone)]
/// Data structure representing the child nodes of a block, which may
/// or may not be leaves
struct Children<const L: usize> {
    features: Vec<Desc>,
    weights: Vec<f32>,
    cluster_size: Vec<usize>,
    ids: Vec<NodeId<L>>,
}

#[derive(Debug, Clone, PartialEq)]
/// Unique identifier for a node. The Leaf variant stores ids of all its parents,
/// which is equivalent to the DirectIndex for any feature matching that leaf.
enum NodeId<const L: usize> {
    Block(usize),
    Leaf(SmallVec<[usize; L]>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Unique identifier for a node. The Leaf variant stores ids of all its parents,
/// which is equivalent to the DirectIndex for any feature matching that leaf.
enum SerializableNodeId {
    Block(usize),
    Leaf(Vec<usize>),
}

impl<const L: usize> Serialize for NodeId<{ L }> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ser = match self {
            NodeId::Block(id) => SerializableNodeId::Block(*id),
            NodeId::Leaf(l_id) => SerializableNodeId::Leaf(l_id.to_vec()),
        };

        ser.serialize(serializer)
    }
}
impl<'de, const L: usize> Deserialize<'de> for NodeId<{ L }> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ser = SerializableNodeId::deserialize(deserializer)?;

        Ok(match ser {
            SerializableNodeId::Block(id) => NodeId::Block(id),
            SerializableNodeId::Leaf(l_id) => NodeId::Leaf(l_id.to_smallvec()),
        })
    }
}

impl<const L: usize> Vocabulary<{ L }> {
    fn transform_generic(&self, features: &Vec<Desc>, di: bool) -> BowResult<(BoW, DirectIdx<L>)> {
        let mut bow: BoW = vec![0.; self.num_leaves];
        let mut direct_idx: DirectIdx<L> = Vec::with_capacity(features.len());
        for feature in features {
            // start at root block
            let mut block = &self.blocks[0];

            // traverse tree
            loop {
                let mut best_child: (u8, usize) = (u8::MAX, 0);
                for (child, child_feat) in block.children.features.iter().enumerate() {
                    let d = Self::hamming(feature, child_feat);
                    if d < best_child.0 {
                        best_child = (d, child)
                    }
                }
                match &block.children.ids[best_child.1] {
                    NodeId::Block(id) => {
                        block = &self.blocks[*id];
                    }
                    NodeId::Leaf(ids) => {
                        if di {
                            // add word parent ids to direct index
                            direct_idx.push(ids.clone());
                        }
                        // add word/leaf id and weight to result
                        let word_id = *ids.last().unwrap();
                        let weight = block.children.weights[best_child.1];
                        match bow.get_mut(word_id) {
                            Some(w) => *w += weight,
                            None => {
                                bow[word_id] = weight;
                            }
                        }
                        break;
                    }
                }
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

        Ok((bow, direct_idx))
    }

    fn cluster(&mut self, features: &Vec<Desc>, parent_ids: Vec<usize>, curr_level: usize) {
        println!(
            "KMeans step with {} features. parents: {:?}, level {}",
            features.len(),
            parent_ids,
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
            .map(|g| self.next_node_id(curr_level == L || g.len() == 1, &parent_ids))
            .collect();

        let children = Children {
            weights: vec![1.; groups.len()],
            ids: ids.clone(),
            cluster_size: groups.iter().map(|g| g.len()).collect(),
            features: clusters,
        };
        let block = Block {
            id: NodeId::Block(*parent_ids.last().unwrap()),
            children,
        };
        self.blocks.push(block);

        // Recurse
        if curr_level < L {
            for (i, id) in ids
                .iter()
                .filter(|&n| matches!(n, NodeId::Block(_)))
                .enumerate()
            {
                // get features from child cluster
                let features: Vec<Desc> = groups[i].iter().map(|&j| features[j]).collect();

                // update parent ids
                let mut ids = parent_ids.clone();
                ids.push(id.get_bid());

                // cluster on child cluster
                self.cluster(&features, ids, curr_level + 1);
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
    fn next_node_id(&mut self, leaf: bool, parent_ids: &Vec<usize>) -> NodeId<L> {
        match leaf {
            true => {
                // Leaf node will hold the block ids of its parents in addition to leaf id, to facilitate getting direct index later
                let mut new_parent_ids = parent_ids[1..].to_smallvec(); // Clone ids but drop the first parent which is always 0
                new_parent_ids.push(self.num_leaves); // Add leaf id
                self.num_leaves += 1;
                NodeId::Leaf(new_parent_ids)
            }
            false => {
                self.num_blocks += 1;
                NodeId::Block(self.num_blocks)
            }
        }
    }
    fn empty(k: usize) -> Self {
        Self {
            blocks: Vec::new(),
            k,
            num_blocks: 0,
            num_leaves: 0,
        }
    }
}

impl<const L: usize> NodeId<L> {
    fn get_bid(&self) -> usize {
        match self {
            NodeId::Block(i) => *i,
            NodeId::Leaf(_) => unreachable!(),
        }
    }
}

impl<const L: usize> fmt::Debug for Children<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Children")
            .field("ids", &self.ids)
            .field("weights", &self.weights)
            .field("cluster size", &self.cluster_size)
            .finish()
    }
}

impl<const L: usize> fmt::Debug for Vocabulary<{ L }> {
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
            .field("Levels", &L)
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
