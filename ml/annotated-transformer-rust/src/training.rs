use tch::{Kind, Reduction, Tensor};

/// Inverse-square-root decay with linear warmup. Step 1 is the first update.
pub fn learning_rate(step: usize, width: i64, warmup: usize, factor: f64) -> f64 {
    assert!(width > 0 && warmup > 0);
    let step = step.max(1) as f64;
    let warmup = warmup as f64;
    factor * (width as f64).powf(-0.5) * step.powf(-0.5).min(step * warmup.powf(-1.5))
}

/// KL(target || model) per non-padding token, following the tutorial.
/// Input probabilities are in log space with vocabulary along the last axis.
pub fn label_smoothed_kl(log_probs: &Tensor, targets: &Tensor, pad: i64, smoothing: f64) -> Tensor {
    let vocab = *log_probs.size().last().unwrap();
    assert!(vocab > 2 && (0..vocab).contains(&pad) && (0.0..1.0).contains(&smoothing));
    let log_probs = log_probs.reshape([-1, vocab]);
    let targets = targets.reshape([-1, 1]);

    let non_pad = targets.ne(pad);
    let valid_classes = Tensor::arange(vocab, (Kind::Int64, log_probs.device())).ne(pad);
    let distribution = Tensor::full_like(&log_probs, smoothing / (vocab - 2) as f64).scatter_value(
        1,
        &targets,
        1.0 - smoothing,
    ) * valid_classes
        * &non_pad;

    let token_count = non_pad.sum(Kind::Float).clamp_min(1.0);
    log_probs.kl_div(&distribution, Reduction::Sum, false) / token_count
}
