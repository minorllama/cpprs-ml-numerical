use std::{collections::HashMap, error::Error, fs};

use annotated_transformer::{learning_rate, Batch, Transformer};
use tch::{nn, no_grad, Device, Kind, Tensor};

use crate::{adam, batch_loss, small_config, BOS, EOS, PAD, UNK};

const BATCH_SIZE: usize = 32;
const SMOOTHING: f64 = 0.1;

type SentencePair = (String, String);
type EncodedPair = (Vec<i64>, Vec<i64>);

fn read_tsv(path: &str) -> Result<Vec<SentencePair>, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let mut pairs = Vec::new();
    for (i, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (src, tgt) = line
            .split_once('\t')
            .filter(|(src, tgt)| {
                !src.trim().is_empty() && !tgt.trim().is_empty() && !tgt.contains('\t')
            })
            .ok_or_else(|| {
                format!(
                    "{path}:{}: expected two nonempty tab-separated fields",
                    i + 1
                )
            })?;
        pairs.push((src.trim().to_owned(), tgt.trim().to_owned()));
    }
    if pairs.is_empty() {
        return Err(format!("{path}: no sentence pairs").into());
    }
    Ok(pairs)
}

struct Vocab {
    ids: HashMap<String, i64>,
    words: Vec<String>,
}

impl Vocab {
    fn new(sentences: impl Iterator<Item = impl AsRef<str>>) -> Self {
        let mut vocab = Self {
            ids: HashMap::new(),
            words: ["<pad>", "<bos>", "<eos>", "<unk>"]
                .map(String::from)
                .to_vec(),
        };
        for sentence in sentences {
            for word in sentence.as_ref().split_whitespace() {
                // Special IDs are inserted by encode, so even a literal "<pad>"
                // found in text gets its own ordinary token ID.
                if !vocab.ids.contains_key(word) {
                    vocab.ids.insert(word.to_owned(), vocab.words.len() as i64);
                    vocab.words.push(word.to_owned());
                }
            }
        }
        vocab
    }

    fn encode(&self, sentence: &str) -> Vec<i64> {
        std::iter::once(BOS)
            .chain(
                sentence
                    .split_whitespace()
                    .map(|word| self.ids.get(word).copied().unwrap_or(UNK)),
            )
            .chain(std::iter::once(EOS))
            .collect()
    }

    fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .copied()
            .skip(1)
            .take_while(|&id| id != EOS)
            .filter(|&id| id != PAD && id != BOS)
            .map(|id| self.words[id as usize].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn encode_pairs(
    pairs: &[SentencePair],
    src: &Vocab,
    tgt: &Vocab,
    max_len: i64,
    path: &str,
) -> Result<Vec<EncodedPair>, Box<dyn Error>> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, (s, t))| {
            let source = src.encode(s);
            let target = tgt.encode(t);
            if source.len().max(target.len()) > max_len as usize {
                return Err(format!(
                    "{path}: pair {} exceeds {} words per sentence",
                    i + 1,
                    max_len - 2
                )
                .into());
            }
            Ok((source, target))
        })
        .collect()
}

fn pad_sequences(rows: &[&[i64]], device: Device) -> Tensor {
    let width = rows.iter().map(|row| row.len()).max().unwrap();
    let mut ids = vec![PAD; rows.len() * width];
    for (row, dest) in rows.iter().zip(ids.chunks_exact_mut(width)) {
        dest[..row.len()].copy_from_slice(row);
    }
    Tensor::from_slice(&ids)
        .view([rows.len() as i64, width as i64])
        .to_device(device)
}

fn make_batch<'a>(rows: impl Iterator<Item = &'a EncodedPair>, device: Device) -> (Batch, usize) {
    let (source, target): (Vec<_>, Vec<_>) =
        rows.map(|(s, t)| (s.as_slice(), t.as_slice())).unzip();
    let tokens = target.iter().map(|t| t.len() - 1).sum();
    let src = pad_sequences(&source, device);
    let tgt = pad_sequences(&target, device);
    (Batch::new(src, &tgt, PAD), tokens)
}

fn validation_nll(model: &Transformer, data: &[EncodedPair], device: Device) -> f64 {
    no_grad(|| {
        let mut loss = 0.0;
        let mut count = 0;
        for rows in data.chunks(BATCH_SIZE) {
            let (batch, tokens) = make_batch(rows.iter(), device);
            loss += batch_loss(model, &batch, 0.0, false).double_value(&[]) * tokens as f64;
            count += tokens;
        }
        loss / count as f64
    })
}

pub fn run(
    train_path: &str,
    valid_path: &str,
    epochs: usize,
    device: Device,
) -> Result<(), Box<dyn Error>> {
    let train = read_tsv(train_path)?;
    let valid = read_tsv(valid_path)?;
    let src_vocab = Vocab::new(train.iter().map(|(src, _)| src));
    let tgt_vocab = Vocab::new(train.iter().map(|(_, tgt)| tgt));
    let cfg = small_config(src_vocab.words.len() as i64, tgt_vocab.words.len() as i64);
    let train = encode_pairs(&train, &src_vocab, &tgt_vocab, cfg.max_len, train_path)?;
    let encoded_valid = encode_pairs(&valid, &src_vocab, &tgt_vocab, cfg.max_len, valid_path)?;

    let store = nn::VarStore::new(device);
    let model = Transformer::new(&store.root(), cfg);
    let mut opt = adam(&store)?;
    let warmup = (train.len().div_ceil(BATCH_SIZE).saturating_mul(epochs) / 10).clamp(10, 400);
    let mut step = 0;
    for epoch in 1..=epochs {
        let permutation = Tensor::randperm(train.len() as i64, (Kind::Int64, Device::Cpu));
        let indices = Vec::<i64>::try_from(permutation)?;
        for chunk in indices.chunks(BATCH_SIZE) {
            let (batch, _) = make_batch(chunk.iter().map(|&i| &train[i as usize]), device);
            step += 1;
            opt.set_lr(learning_rate(step, cfg.d_model, warmup, 0.5));
            opt.backward_step(&batch_loss(&model, &batch, SMOOTHING, true));
        }
        println!(
            "epoch {epoch:3} | validation NLL {:.4}",
            validation_nll(&model, &encoded_valid, device)
        );
    }

    for ((source, target), (ids, _)) in valid.iter().zip(&encoded_valid).take(4) {
        let input = pad_sequences(&[ids.as_slice()], device);
        let output = model.greedy_decode(&input, BOS, Some(EOS), PAD, cfg.max_len);
        let output = Vec::<i64>::try_from(output.squeeze_dim(0).to_device(Device::Cpu))?;
        println!(
            "input: {source}\nexpected: {target}\npredicted: {}\n",
            tgt_vocab.decode(&output)
        );
    }
    Ok(())
}
