use tch::{Kind, Tensor};

/// True for source keys that attention may read. Shape [batch, 1, source_len].
pub fn source_mask(src: &Tensor, pad_id: i64) -> Tensor {
    src.ne(pad_id).unsqueeze(1)
}

/// True for visible target keys. Shape [batch, target_len, target_len].
pub fn target_mask(tgt: &Tensor, pad_id: i64) -> Tensor {
    let len = tgt.size()[1];
    let causal = Tensor::ones([len, len], (Kind::Bool, tgt.device())).tril(0);
    tgt.ne(pad_id).unsqueeze(1).logical_and(&causal)
}

/// A teacher-forcing batch built from right-padded Int64 token IDs.
pub struct Batch {
    pub src: Tensor,
    pub tgt_input: Tensor,
    pub tgt_output: Tensor,
    pub src_keep: Tensor,
    pub tgt_keep: Tensor,
}

impl Batch {
    pub fn new(src: Tensor, tgt: &Tensor, pad_id: i64) -> Self {
        let len = tgt.size()[1] - 1;
        assert!(len > 0, "target requires at least two positions");
        let tgt_input = tgt.narrow(1, 0, len);
        let tgt_output = tgt.narrow(1, 1, len);
        let src_keep = source_mask(&src, pad_id);
        let tgt_keep = target_mask(&tgt_input, pad_id);
        Self {
            src,
            tgt_input,
            tgt_output,
            src_keep,
            tgt_keep,
        }
    }
}
