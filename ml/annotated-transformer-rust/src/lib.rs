//! Encoder–decoder Transformer following the executable Harvard tutorial.

mod data;
mod model;
mod training;

pub use data::{source_mask, target_mask, Batch};
pub use model::{Config, Transformer};
pub use training::{label_smoothed_kl, learning_rate};
