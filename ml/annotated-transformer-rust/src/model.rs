use tch::{nn, no_grad, Kind, Tensor};

use crate::{source_mask, target_mask};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub src_vocab: i64,
    pub tgt_vocab: i64,
    pub layers: usize,
    pub d_model: i64,
    pub heads: i64,
    pub d_ff: i64,
    pub dropout: f64,
    pub max_len: i64,
}

impl Config {
    /// The tutorial's full-size model dimensions.
    pub fn base(src_vocab: i64, tgt_vocab: i64) -> Self {
        Self {
            src_vocab,
            tgt_vocab,
            layers: 6,
            d_model: 512,
            heads: 8,
            d_ff: 2048,
            dropout: 0.1,
            max_len: 5000,
        }
    }
}

fn xavier(fan_in: i64, fan_out: i64) -> nn::Init {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    nn::Init::Uniform {
        lo: -bound,
        up: bound,
    }
}

fn linear(path: nn::Path<'_>, input: i64, output: i64) -> nn::Linear {
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: xavier(input, output),
            bs_init: Some(nn::Init::Const(0.0)),
            bias: true,
        },
    )
}

/// Projections use [batch, heads, positions, head_dim] internally.
struct MultiHeadAttention {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    out: nn::Linear,
    heads: i64,
    head_dim: i64,
    dropout: f64,
}

impl MultiHeadAttention {
    fn new(path: nn::Path<'_>, cfg: Config) -> Self {
        let width = cfg.d_model;
        Self {
            q: linear(&path / "query", width, width),
            k: linear(&path / "key", width, width),
            v: linear(&path / "value", width, width),
            out: linear(&path / "output", width, width),
            heads: cfg.heads,
            head_dim: width / cfg.heads,
            dropout: cfg.dropout,
        }
    }

    fn project(&self, tokens: &Tensor, projection: &nn::Linear) -> Tensor {
        let (batch, len, _) = tokens.size3().unwrap();
        tokens
            .apply(projection)
            .view([batch, len, self.heads, self.head_dim])
            .transpose(1, 2)
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, keep: &Tensor, train: bool) -> Tensor {
        let query = self.project(q, &self.q);
        let key = self.project(k, &self.k);
        let value = self.project(v, &self.v);
        let keep = keep.unsqueeze(1); // Broadcast over heads.
        let scores = query.matmul(&key.transpose(-2, -1)) / (self.head_dim as f64).sqrt();

        // A fully masked row gets a finite softmax input and then zero weights.
        let empty_row = keep.any_dim(-1, true).logical_not();
        let blocked = keep.logical_or(&empty_row).logical_not();
        let weights = scores
            .masked_fill(&blocked, f64::NEG_INFINITY)
            .softmax(-1, scores.kind())
            .masked_fill(&keep.logical_not(), 0.0)
            .dropout(self.dropout, train);

        weights
            .matmul(&value)
            .transpose(1, 2)
            .contiguous()
            .view([q.size()[0], q.size()[1], self.heads * self.head_dim])
            .apply(&self.out)
    }
}

struct FeedForward {
    expand: nn::Linear,
    contract: nn::Linear,
    dropout: f64,
}

impl FeedForward {
    fn new(path: nn::Path<'_>, cfg: Config) -> Self {
        Self {
            expand: linear(&path / "expand", cfg.d_model, cfg.d_ff),
            contract: linear(&path / "contract", cfg.d_ff, cfg.d_model),
            dropout: cfg.dropout,
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.expand)
            .relu()
            .dropout(self.dropout, train)
            .apply(&self.contract)
    }
}

/// The tutorial uses sample standard deviation and adds epsilon after the square root.
struct LayerNorm {
    scale: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn new(path: nn::Path<'_>, width: i64) -> Self {
        Self {
            scale: path.ones("scale", &[width]),
            bias: path.zeros("bias", &[width]),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean_dim(&[-1_i64][..], true, x.kind());
        let std = x.std_dim(&[-1_i64][..], true, true);
        (&self.scale * (x - mean)) / (std + 1e-6) + &self.bias
    }
}

struct Sublayer {
    norm: LayerNorm,
    dropout: f64,
}

impl Sublayer {
    fn new(path: nn::Path<'_>, cfg: Config) -> Self {
        Self {
            norm: LayerNorm::new(path, cfg.d_model),
            dropout: cfg.dropout,
        }
    }

    fn forward(&self, x: &Tensor, train: bool, f: impl FnOnce(&Tensor) -> Tensor) -> Tensor {
        // Follow the notebook's pre-norm implementation.
        x + f(&self.norm.forward(x)).dropout(self.dropout, train)
    }
}

struct EncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_residual: Sublayer,
    ff_residual: Sublayer,
}

impl EncoderLayer {
    fn new(path: nn::Path<'_>, cfg: Config) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(&path / "self_attention", cfg),
            feed_forward: FeedForward::new(&path / "feed_forward", cfg),
            attention_residual: Sublayer::new(&path / "attention_norm", cfg),
            ff_residual: Sublayer::new(&path / "ff_norm", cfg),
        }
    }

    fn forward(&self, x: &Tensor, keep: &Tensor, train: bool) -> Tensor {
        let x = self.attention_residual.forward(x, train, |z| {
            self.self_attention.forward(z, z, z, keep, train)
        });
        self.ff_residual
            .forward(&x, train, |z| self.feed_forward.forward(z, train))
    }
}

struct DecoderLayer {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    self_residual: Sublayer,
    cross_residual: Sublayer,
    ff_residual: Sublayer,
}

impl DecoderLayer {
    fn new(path: nn::Path<'_>, cfg: Config) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(&path / "self_attention", cfg),
            cross_attention: MultiHeadAttention::new(&path / "cross_attention", cfg),
            feed_forward: FeedForward::new(&path / "feed_forward", cfg),
            self_residual: Sublayer::new(&path / "self_norm", cfg),
            cross_residual: Sublayer::new(&path / "cross_norm", cfg),
            ff_residual: Sublayer::new(&path / "ff_norm", cfg),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        memory: &Tensor,
        src_keep: &Tensor,
        tgt_keep: &Tensor,
        train: bool,
    ) -> Tensor {
        let x = self.self_residual.forward(x, train, |z| {
            self.self_attention.forward(z, z, z, tgt_keep, train)
        });
        let x = self.cross_residual.forward(&x, train, |z| {
            self.cross_attention
                .forward(z, memory, memory, src_keep, train)
        });
        self.ff_residual
            .forward(&x, train, |z| self.feed_forward.forward(z, train))
    }
}

pub struct Transformer {
    src_embed: nn::Embedding,
    tgt_embed: nn::Embedding,
    positions: Tensor,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
    encoder_norm: LayerNorm,
    decoder_norm: LayerNorm,
    generator: nn::Linear,
    cfg: Config,
}

impl Transformer {
    pub fn new(path: &nn::Path<'_>, cfg: Config) -> Self {
        assert!(cfg.heads > 0 && cfg.d_model > 1 && cfg.d_model % cfg.heads == 0);
        assert!(cfg.layers > 0 && cfg.d_ff > 0 && cfg.max_len > 0);
        assert!(cfg.src_vocab > 0 && cfg.tgt_vocab > 2 && (0.0..1.0).contains(&cfg.dropout));

        let embed = |name, vocab| {
            nn::embedding(
                path / name,
                vocab,
                cfg.d_model,
                nn::EmbeddingConfig {
                    ws_init: xavier(vocab, cfg.d_model),
                    ..Default::default()
                },
            )
        };
        let build_positions = || {
            let layout = (Kind::Float, path.device());
            let positions = Tensor::arange(cfg.max_len, layout).unsqueeze(1);
            let exponents = Tensor::arange_start_step(0, cfg.d_model, 2, layout);
            let angles = positions * (exponents * (-10000.0_f64.ln() / cfg.d_model as f64)).exp();
            Tensor::stack(&[angles.sin(), angles.cos()], -1)
                .flatten(1, 2)
                .narrow(1, 0, cfg.d_model)
        };

        Self {
            src_embed: embed("src_embedding", cfg.src_vocab),
            tgt_embed: embed("tgt_embedding", cfg.tgt_vocab),
            positions: path.add("positions", build_positions(), false),
            encoder: (0..cfg.layers)
                .map(|i| EncoderLayer::new(path / "encoder" / i, cfg))
                .collect(),
            decoder: (0..cfg.layers)
                .map(|i| DecoderLayer::new(path / "decoder" / i, cfg))
                .collect(),
            encoder_norm: LayerNorm::new(path / "encoder_norm", cfg.d_model),
            decoder_norm: LayerNorm::new(path / "decoder_norm", cfg.d_model),
            generator: linear(path / "generator", cfg.d_model, cfg.tgt_vocab),
            cfg,
        }
    }

    fn embed(&self, ids: &Tensor, embedding: &nn::Embedding, train: bool) -> Tensor {
        let len = ids.size()[1];
        assert!(len <= self.cfg.max_len, "sequence exceeds max_len");
        let tokens = ids.apply(embedding) * (self.cfg.d_model as f64).sqrt();
        let positions = self.positions.narrow(0, 0, len).to_kind(tokens.kind());
        (tokens + positions).dropout(self.cfg.dropout, train)
    }

    /// Encode source IDs into [batch, source_len, d_model].
    pub fn encode(&self, src: &Tensor, src_keep: &Tensor, train: bool) -> Tensor {
        let mut x = self.embed(src, &self.src_embed, train);
        for layer in &self.encoder {
            x = layer.forward(&x, src_keep, train);
        }
        self.encoder_norm.forward(&x)
    }

    /// Decode a whole target prefix into [batch, target_len, d_model].
    pub fn decode(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        src_keep: &Tensor,
        tgt_keep: &Tensor,
        train: bool,
    ) -> Tensor {
        let mut x = self.embed(tgt, &self.tgt_embed, train);
        for layer in &self.decoder {
            x = layer.forward(&x, memory, src_keep, tgt_keep, train);
        }
        self.decoder_norm.forward(&x)
    }

    pub fn log_probs(&self, hidden: &Tensor) -> Tensor {
        hidden.apply(&self.generator).log_softmax(-1, Kind::Float)
    }

    /// Return [batch, target_len, tgt_vocab] log probabilities.
    pub fn forward(
        &self,
        src: &Tensor,
        tgt: &Tensor,
        src_keep: &Tensor,
        tgt_keep: &Tensor,
        train: bool,
    ) -> Tensor {
        let memory = self.encode(src, src_keep, train);
        self.log_probs(&self.decode(tgt, &memory, src_keep, tgt_keep, train))
    }

    /// Greedy decoding for a batch. Length includes BOS. With EOS, finished rows
    /// receive PAD while other rows keep decoding.
    pub fn greedy_decode(
        &self,
        src: &Tensor,
        bos: i64,
        eos: Option<i64>,
        pad: i64,
        max_len: i64,
    ) -> Tensor {
        assert!((1..=self.cfg.max_len).contains(&max_len));
        no_grad(|| {
            let src_keep = source_mask(src, pad);
            let memory = self.encode(src, &src_keep, false);
            let batch = src.size()[0];
            let mut ids = Tensor::full([batch, 1], bos, (Kind::Int64, src.device()));
            let mut done = Tensor::zeros([batch, 1], (Kind::Bool, src.device()));
            for prefix_len in 1..max_len {
                let keep = target_mask(&ids, pad);
                let hidden = self.decode(&ids, &memory, &src_keep, &keep, false);
                let next = hidden
                    .select(1, prefix_len - 1)
                    .apply(&self.generator)
                    .argmax(-1, true)
                    .masked_fill(&done, pad);
                if let Some(eos) = eos {
                    done = done.logical_or(&next.eq(eos));
                }
                ids = Tensor::cat(&[ids, next], 1);
                if eos.is_some() && done.all().int64_value(&[]) != 0 {
                    break;
                }
            }
            ids
        })
    }
}
