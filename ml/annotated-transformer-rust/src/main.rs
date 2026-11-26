mod translation;

use std::{env, error::Error};

use annotated_transformer::{label_smoothed_kl, learning_rate, Batch, Config, Transformer};
use tch::{nn, nn::OptimizerConfig, no_grad, Device, Kind, Tensor};

const PAD: i64 = 0;
const BOS: i64 = 1;
const EOS: i64 = 2;
const UNK: i64 = 3;

fn small_config(src_vocab: i64, tgt_vocab: i64) -> Config {
    Config {
        layers: 2,
        d_model: 64,
        heads: 4,
        d_ff: 128,
        max_len: 128,
        ..Config::base(src_vocab, tgt_vocab)
    }
}

fn adam(store: &nn::VarStore) -> tch::Result<nn::Optimizer> {
    nn::Adam {
        beta1: 0.9,
        beta2: 0.98,
        eps: 1e-9,
        ..Default::default()
    }
    .build(store, 0.0)
}

fn batch_loss(model: &Transformer, batch: &Batch, smoothing: f64, train: bool) -> Tensor {
    let output = model.forward(
        &batch.src,
        &batch.tgt_input,
        &batch.src_keep,
        &batch.tgt_keep,
        train,
    );
    label_smoothed_kl(&output, &batch.tgt_output, PAD, smoothing)
}

const COPY_VOCAB: i64 = 11;
const COPY_LEN: i64 = 10;

fn copy_sequences(batch_size: i64, device: Device) -> Tensor {
    let first = Tensor::full([batch_size, 1], BOS, (Kind::Int64, device));
    let rest = Tensor::randint_low(
        2,
        COPY_VOCAB,
        [batch_size, COPY_LEN - 1],
        (Kind::Int64, device),
    );
    Tensor::cat(&[first, rest], 1)
}

fn run_copy(steps: usize, device: Device) -> Result<(), Box<dyn Error>> {
    let cfg = small_config(COPY_VOCAB, COPY_VOCAB);
    let store = nn::VarStore::new(device);
    let model = Transformer::new(&store.root(), cfg);
    let mut optimizer = adam(&store)?;
    let held_out = copy_sequences(128, device);

    let eval_batch = Batch::new(held_out.shallow_clone(), &held_out, PAD);
    println!(
        "initial validation NLL {:.4}",
        no_grad(|| batch_loss(&model, &eval_batch, 0.0, false).double_value(&[]))
    );
    for step in 1..=steps {
        let tokens = copy_sequences(64, device);
        let batch = Batch::new(tokens.shallow_clone(), &tokens, PAD);
        optimizer.set_lr(learning_rate(step, cfg.d_model, 100, 1.0));
        let loss = batch_loss(&model, &batch, 0.0, true);
        optimizer.backward_step(&loss);
        if step % 100 == 0 || step == steps {
            let val = no_grad(|| batch_loss(&model, &eval_batch, 0.0, false).double_value(&[]));
            println!(
                "step {step:4} | train NLL {:.4} | validation NLL {val:.4}",
                loss.double_value(&[])
            );
        }
    }

    let prediction = model.greedy_decode(&held_out, BOS, None, PAD, COPY_LEN);
    let exact = prediction.eq_tensor(&held_out).all_dim(1, false);
    let accuracy = exact
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    println!(
        "greedy exact-match on 128 validation sequences: {:.1}%",
        100.0 * accuracy
    );
    for row in 0..3 {
        let source = Vec::<i64>::try_from(held_out.get(row).to_device(Device::Cpu))?;
        let copied = Vec::<i64>::try_from(prediction.get(row).to_device(Device::Cpu))?;
        println!("source {source:?} -> {copied:?}");
    }
    Ok(())
}

fn positive_count(arg: Option<&String>, default: usize) -> Result<usize, Box<dyn Error>> {
    let count = arg.map_or(Ok(default), |number| number.parse::<usize>())?;
    if count == 0 {
        return Err("steps or epochs must be positive".into());
    }
    Ok(count)
}

const USAGE: &str = "Usage:
  annotated-transformer [copy [steps]]
  annotated-transformer translate TRAIN.tsv VALID.tsv [epochs]";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args
        .first()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!("{USAGE}");
        return Ok(());
    }
    let device = Device::cuda_if_available();
    tch::manual_seed(42);
    if device == Device::Cpu {
        tch::set_num_threads(2);
    }
    match args.first().map(String::as_str) {
        None => run_copy(800, device),
        Some("copy") if args.len() <= 2 => run_copy(positive_count(args.get(1), 800)?, device),
        Some("translate") if (3..=4).contains(&args.len()) => {
            translation::run(&args[1], &args[2], positive_count(args.get(3), 10)?, device)
        }
        _ => Err(USAGE.into()),
    }
}
