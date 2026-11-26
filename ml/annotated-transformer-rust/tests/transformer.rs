use annotated_transformer::{label_smoothed_kl, source_mask, target_mask, Config, Transformer};
use tch::{nn, no_grad, Device, Kind, Tensor};

#[test]
fn decoder_prefix_is_invariant_to_future_tokens_and_source_padding() {
    no_grad(|| {
        let store = nn::VarStore::new(Device::Cpu);
        let cfg = Config {
            layers: 1,
            d_model: 12,
            heads: 3,
            d_ff: 24,
            dropout: 0.0,
            max_len: 8,
            ..Config::base(12, 12)
        };
        let model = Transformer::new(&store.root(), cfg);

        let run = |src: &[i64], tgt: &[i64]| {
            let src = Tensor::from_slice(src).unsqueeze(0);
            let tgt = Tensor::from_slice(tgt).unsqueeze(0);
            model.forward(
                &src,
                &tgt,
                &source_mask(&src, 0),
                &target_mask(&tgt, 0),
                false,
            )
        };
        let first = run(&[1, 4, 7], &[1, 6, 3]);
        let second = run(&[1, 4, 7, 0, 0], &[1, 6, 9]);
        assert_eq!(first.size(), [1, 3, 12]);
        assert!(first
            .narrow(1, 0, 2)
            .allclose(&second.narrow(1, 0, 2), 1e-5, 1e-5, false));
        assert_eq!(run(&[0, 0], &[0, 0]).isfinite().all().int64_value(&[]), 1);
    });
}

#[test]
fn smoothing_excludes_padding_class_and_padding_rows() {
    let log_probs = Tensor::from_slice(&[0.20_f32, 0.50, 0.15, 0.15, 0.25, 0.25, 0.25, 0.25])
        .view([2, 4])
        .log();
    let targets = Tensor::from_slice(&[1_i64, 0]);
    let loss = label_smoothed_kl(&log_probs, &targets, 0, 0.2);
    let expected = 0.8 * (0.8_f64 / 0.5).ln() + 0.2 * (0.1_f64 / 0.15).ln();
    assert!((loss.double_value(&[]) - expected).abs() < 1e-6);

    let all_pad = Tensor::zeros([2], (Kind::Int64, Device::Cpu));
    assert_eq!(
        label_smoothed_kl(&log_probs, &all_pad, 0, 0.2).double_value(&[]),
        0.0
    );
}
