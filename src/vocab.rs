use bitvec::{order::Msb0, view::BitView};
use rand::{
    distributions::{weighted::WeightedIndex, Distribution},
    seq::SliceRandom,
    thread_rng, Rng,
};
use serde::{Deserialize, Serialize};
use smallvec::ToSmallVec;
use std::fmt;

use crate::*;

enum ClusterInitMethod {
    #[allow(dead_code)]
    Random,
    #[allow(clippy::upper_case_acronyms)]
    KMeansPP,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Default)]
/// Visual vocabulary built from a collection of image features.
pub struct Vocabulary {
    blocks: Vec<Block>,
    k: usize,
    levels: usize,
    num_blocks: usize,
    num_leaves: usize,
}

/// Vocabulary API
impl Vocabulary {
    /// Transform a vector of binary descriptors into its bag of words
    /// representation with respect to the Vocabulary. Descriptor is l1 normalized.
    /// Returns Err if features is empty.
    pub fn transform(&self, features: &[Desc]) -> BowResult<BoW> {
        self.transform_inner(features, false).map(|res| res.0)
    }

    /// Transform a vector of binary descriptors into its bag of words
    /// representation with respect to the Vocabulary. Descriptor is l1 normalized.
    /// Returns Err if features is empty.
    ///
    /// Also provides "direct index" from the features to their corresponding nodes in the Vocabulary tree.
    ///
    /// The direct index for `feature[i]` is `di = DirectIdx[i]` where
    /// `di.len() <= l` (number of levels), and `di[j]` is the id of the node matching `feature[i]`
    /// at level `j` in the Vocabulary tree.
    pub fn transform_with_direct_idx(&self, features: &[Desc]) -> BowResult<(BoW, DirectIdx)> {
        self.transform_inner(features, true)
    }

    /// Build a vocabulary from a collection of descriptors.
    ///
    /// Args:
    /// - k: Branching factor
    /// - l: Max number of levels (Should be <= 5)
    pub fn create(features: &[Desc], k: usize, l: usize) -> Self {
        // Start with root of tree
        let mut v = Self::empty(k, l);

        // Build with recursive k-means clustering of features
        v.cluster(features, vec![0], 1);

        // Sort by block id
        v.blocks.sort_by(|a, b| a.id.get_bid().cmp(&b.id.get_bid()));

        v
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

//###################                Helpers                 #########################
//####################################################################################

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
/// A unit representing a non-leaf node in the vocabulary
struct Block {
    id: NodeId,
    children: Children,
}

#[derive(Serialize, Deserialize, PartialEq, Clone)]
/// Data structure representing the child nodes of a block, which may
/// or may not be leaves
struct Children {
    features: Vec<Desc>,
    weights: Vec<f32>,
    cluster_size: Vec<usize>,
    ids: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Unique identifier for a node. The Leaf variant stores ids of all its parents,
/// which is equivalent to the DirectIndex for any feature matching that leaf.
enum NodeId {
    Block(usize),
    Leaf(IdPath),
}

impl Vocabulary {
    fn transform_inner(&self, features: &[Desc], di: bool) -> BowResult<(BoW, DirectIdx)> {
        if features.is_empty() {
            return Err(BowErr::NoFeatures);
        }

        let mut bow = BoW(vec![0.; self.num_leaves]);
        let mut direct_idx: DirectIdx = Vec::with_capacity(features.len());
        for feature in features {
            // start at root block
            let mut block = &self.blocks[0];

            // traverse tree
            loop {
                let mut best_child: (u8, usize) = (u8::MAX, 0);
                for (child, child_feat) in block.children.features.iter().enumerate() {
                    let d = hamming(feature, child_feat);
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
                        match bow.0.get_mut(word_id) {
                            Some(w) => *w += weight,
                            None => bow.0[word_id] = weight,
                        }
                        break;
                    }
                }
            }
        }
        // Normalize BoW vector
        let sum: f32 = bow.0.iter().sum();
        if sum > 0. {
            let inv_sum = 1. / sum;
            for w in bow.0.iter_mut() {
                *w *= inv_sum;
            }
        }

        Ok((bow, direct_idx))
    }

    fn cluster(&mut self, features: &[Desc], parent_ids: Vec<usize>, curr_level: usize) {
        // println!(
        //     "KMeans step with {} features. parents: {:?}, level {}",
        //     features.len(),
        //     parent_ids,
        //     curr_level
        // );

        let mut clusters = self.initialize_clusters(features, ClusterInitMethod::KMeansPP);
        let mut groups = vec![Vec::new(); clusters.len()];

        loop {
            let mut new_groups: Vec<Vec<usize>> = vec![Vec::new(); groups.len()];
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
                    let desc = group.iter().map(|&i| &features[i]).collect();
                    Self::desc_mean(desc)
                })
                .collect();
            groups = new_groups;
        }

        // remove empty groups which rarely occur
        groups.retain(|g| !g.is_empty());
        clusters.retain(|c| c != &[0_u8; std::mem::size_of::<Desc>()]);
        assert_eq!(groups.len(), clusters.len());

        // create block
        let ids: Vec<_> = groups
            .iter()
            .map(|g| self.next_node_id(curr_level == self.levels || g.len() == 1, &parent_ids))
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
        if curr_level < self.levels {
            for (i, id) in ids
                .iter()
                .enumerate()
                .filter(|&(_, n)| matches!(n, NodeId::Block(_)))
            {
                // get features from child cluster
                let features: Vec<Desc> = groups[i].iter().map(|&j| features[j]).collect();

                // update parent ids
                let mut ids = parent_ids.clone();
                ids.push(id.get_bid());

                // perform clustering on child features
                self.cluster(&features, ids, curr_level + 1);
            }
        }
    }

    /// Initialize clusters for kmeans
    fn initialize_clusters(&self, features: &[Desc], method: ClusterInitMethod) -> Vec<Desc> {
        // if fewer than k unique features, simply return them
        if features.len() <= self.k {
            return features.to_vec();
        }

        let mut deduped = features.to_vec();
        deduped.sort_unstable();
        deduped.dedup();

        if deduped.len() <= self.k {
            return deduped;
        }

        match method {
            ClusterInitMethod::Random => self.init_random(features),
            ClusterInitMethod::KMeansPP => self.init_kmeanspp(features),
        }
    }

    fn init_random(&self, features: &[Desc]) -> Vec<Desc> {
        let mut rng = thread_rng();
        features
            .choose_multiple(&mut rng, self.k)
            .cloned()
            .collect()
    }

    fn init_kmeanspp(&self, features: &[Desc]) -> Vec<Desc> {
        let mut rng = thread_rng();
        let mut features = features.to_owned();
        let mut centroids = Vec::with_capacity(self.k);
        // 1. Randomly select the first centroid.
        let random_idx = rng.gen_range(0..features.len());
        centroids.push(features.remove(random_idx));

        while centroids.len() < self.k {
            // 2. For each data point compute its distance from the nearest, previously chosen centroid.
            let mut dists: Vec<f32> = vec![std::u8::MAX as f32; features.len()];
            for (i, f) in features.iter().enumerate() {
                for c in centroids.iter() {
                    dists[i] = f32::min(hamming(f, c) as f32, dists[i]);
                }
            }
            // 3. Select the next centroid from the data points such that the probability of choosing a point
            // as centroid is directly proportional to its distance from the nearest, previously chosen centroid.
            let centroid_weights = WeightedIndex::new(dists).expect("weighted index err");
            let weighted_random_idx = centroid_weights.sample(&mut rng);
            centroids.push(features.remove(weighted_random_idx));
        }

        centroids
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

    /// Provide the next NodeId, either leaf/word or block.
    fn next_node_id(&mut self, leaf: bool, parent_ids: &[usize]) -> NodeId {
        if leaf {
            // Leaf node will hold the block ids of its parents in addition to leaf id, to facilitate getting direct index later
            let mut new_parent_ids = parent_ids[1..].to_smallvec(); //  drop the first parent which is always 0
            new_parent_ids.push(self.num_leaves); // Add leaf id
            self.num_leaves += 1;
            NodeId::Leaf(new_parent_ids)
        } else {
            self.num_blocks += 1;
            NodeId::Block(self.num_blocks)
        }
    }
    fn empty(k: usize, l: usize) -> Self {
        Self {
            blocks: Vec::new(),
            k,
            num_blocks: 0,
            num_leaves: 0,
            levels: l,
        }
    }
}

#[inline]
/// Hamming distance between two binary arrays (descriptors).
fn hamming(x: &[u8], y: &[u8]) -> u8 {
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones() as u8)
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
        clust_sizes.sort_unstable();
        f.debug_struct("Vocabulary")
            .field("Word/Leaf Nodes", &self.num_leaves)
            .field("Other Nodes", &self.num_blocks)
            .field("Levels", &self.levels)
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
            .field(
                "Median Word Cluster Size",
                &clust_sizes[clust_sizes.len() / 2],
            )
            .finish()
    }
}
