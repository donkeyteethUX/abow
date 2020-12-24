#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct FbowParams {
    desc_name: Vec<u8>,       // descriptor name. May be empty
    alignment: u32,           // memory aligment
    nblocks: u32,             // total number of blocks
    desc_size_bytes_wp: u64,  // size of the descriptor (includes padding)
    block_size_bytes_wp: u64, // size of a block (includes padding)
    feature_off_start: u64,   // within a block, where the features start
    child_off_start: u64,     // within a block,where the children offset part starts
    total_size: u64,
    desc_type: i32, // original descriptor size (without padding)
    desc_size: i32, // original descriptor type (without padding)
    m_k: u32,       // number of children per node
}

impl FbowParams {
    fn load(bytes: Vec<u8>) -> Self {
        Self {
            desc_name: bytes[..50].try_into().unwrap(),
            alignment: u32::from_le_bytes(bytes[52..56].try_into().unwrap()),
            nblocks: u32::from_ne_bytes(bytes[56..60].try_into().unwrap()),
            desc_size_bytes_wp: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            block_size_bytes_wp: u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
            feature_off_start: u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
            child_off_start: u64::from_le_bytes(bytes[88..96].try_into().unwrap()),
            total_size: u64::from_le_bytes(bytes[96..104].try_into().unwrap()),
            desc_type: i32::from_le_bytes(bytes[104..108].try_into().unwrap()),
            desc_size: i32::from_le_bytes(bytes[108..112].try_into().unwrap()),
            m_k: u32::from_le_bytes(bytes[112..116].try_into().unwrap()),
        }
    }
}

impl Block {
    fn load(bytes: Vec<u8>, params: &FbowParams) -> Self {
        // println!("is_leaf bytes: {:#?}:", &bytes[2..4]);
        let feat_st: usize = params.feature_off_start as usize;
        let child_st: usize = params.child_off_start as usize;
        let features: Vec<Desc> = bytes[feat_st..child_st]
            .chunks(std::mem::size_of::<Desc>())
            .map(|c| c.try_into().unwrap())
            .collect();
        let mut weights: Vec<f32> = Vec::new();
        let mut ids: Vec<NodeId> = Vec::new();
        for i in 0..(params.m_k as usize) {
            let start = params.child_off_start as usize + i * 8;

            let id = u32::from_ne_bytes(bytes[start..(start + 4)].try_into().unwrap());
            let w = f32::from_ne_bytes(bytes[(start + 4)..(start + 8)].try_into().unwrap());
            let leaf = id & 0x80000000 != 0;
            match leaf {
                true => {
                    ids.push(NodeId::Leaf((id & 0x7FFFFFFF) as usize));
                }
                false => {
                    ids.push(NodeId::Leaf((id & 0x7FFFFFFF) as usize));
                }
            }
            weights.push(w);
        }

        Self {
            // n: u16::from_ne_bytes(bytes[0..2].try_into().unwrap()) as u8,
            id: NodeId::Block(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()) as usize),
            children: Children {
                features,
                weights,
                cluster_size: Vec::new(), // fake obviously
                ids,
            },
        }
    }
}

impl Vocabulary {
    pub fn load_voc<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = File::open(path)?;

        let mut b = f.bytes().map(|b| b.unwrap());
        let _sig_bytes: Vec<u8> = b.by_ref().take(8).collect();
        let param_bytes: Vec<u8> = b.by_ref().take(120).collect();
        let params = FbowParams::load(param_bytes);

        println!("params: {:?}", params);

        // Check that binary descriptors have correct length
        assert_eq!(
            std::mem::size_of::<Desc>(),
            params.desc_size_bytes_wp as usize,
            "Descriptor size mismatch!"
        );

        let data: Vec<u8> = b.collect();
        // println!("sig: {}", sig);
        // println!("params: {:#?}", params);
        assert_eq!(params.total_size, data.len() as u64);
        let mut blocks: Vec<_> = Vec::new();

        for i in 0..(params.nblocks as usize) {
            let start: usize = i * params.block_size_bytes_wp as usize;
            let end = start + params.block_size_bytes_wp as usize;
            let bytes = data[start..end].to_vec();
            let b = Block::load(bytes, &params);
            // println!("block: {:#?}", b);
            blocks.push(b);
        }

        Ok(Self {
            blocks,
            k: params.m_k as usize,
            l: 0,
            next_block_id: 0,
            next_leaf_id: 0,
        })
    }
}
