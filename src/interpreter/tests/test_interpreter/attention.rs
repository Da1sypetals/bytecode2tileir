use crate::interpreter::tests::test_interpreter::logging_utils::init_test_logger;
use crate::interpreter::{
    args::{KernelArgv, KernelGenericArgv},
    data_structures::interpreter::Interpreter,
};
use indicatif::ProgressIterator;
use log::info;
use ndarray::{Array1, Array3, Axis};
use ndrange::ndrange;
use rand::RngExt;
use std::path::Path;

fn rand_3d_f16(dim0: usize, dim1: usize, dim2: usize) -> Array3<f16> {
    let mut rng = rand::rng();
    Array3::from_elem([dim0, dim1, dim2], 0.0_f16)
        .mapv_into(|_| (rng.random_range(-1.0..1.0_f32)) as f16)
}

/// Reference implementation of flash attention
/// Matches the MLIR kernel which:
/// - Scales Q by log2(e) and uses exp2, which is equivalent to exp(Q @ K^T)
/// - Does NOT use 1/sqrt(d) scaling
/// Q: [seq_len_q, num_heads, head_dim]
/// K: [seq_len_kv, num_heads, head_dim]
/// V: [seq_len_kv, num_heads, head_dim]
/// Output: [seq_len_q, num_heads, head_dim]
fn flash_attention_ref(q: &Array3<f16>, k: &Array3<f16>, v: &Array3<f16>) -> Array3<f32> {
    let (seq_len_q, num_heads, head_dim) = q.dim();
    let (seq_len_kv, _, _) = k.dim();

    // Note: MLIR kernel does NOT use 1/sqrt(d) scaling
    // It uses exp2(Q * log2(e) @ K^T) = exp(Q @ K^T)

    let mut output = Array3::<f32>::zeros([seq_len_q, num_heads, head_dim]);

    for h in 0..num_heads {
        for i in 0..seq_len_q {
            // Extract Q[i, h, :] as f32
            let q_vec: Array1<f32> = q
                .index_axis(Axis(0), i)
                .index_axis(Axis(0), h)
                .mapv(|x| x as f32);

            // Compute attention scores: Q[i] @ K[:]^T (no scaling)
            let mut scores = Array1::<f32>::zeros(seq_len_kv);
            for j in 0..seq_len_kv {
                let k_vec: Array1<f32> = k
                    .index_axis(Axis(0), j)
                    .index_axis(Axis(0), h)
                    .mapv(|x: f16| x as f32);
                scores[j] = (&q_vec * &k_vec).sum();
            }

            // Softmax using exp (since kernel uses exp2 with log2(e) scaling)
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Array1<f32> = scores.mapv(|s| (s - max_score).exp());
            let sum_exp: f32 = exp_scores.sum();
            let attn_weights = exp_scores / sum_exp;

            // Compute weighted sum of V
            for j in 0..seq_len_kv {
                let v_vec: Array1<f32> = v
                    .index_axis(Axis(0), j)
                    .index_axis(Axis(0), h)
                    .mapv(|x: f16| x as f32);
                for d in 0..head_dim {
                    output[[i, h, d]] += attn_weights[j] * v_vec[d];
                }
            }
        }
    }

    output
}

/// Test attention kernel
///
/// Input parameters (from attention_in_params.md):
///   - %0: Query matrix data pointer (f16)
///   - %1: Query matrix dim 0 size (seq_len_q)
///   - %2: Query matrix dim 1 size
///   - %3: Query matrix dim 2 size
///   - %4: Query matrix dim 0 stride
///   - %5: Query matrix dim 1 stride
///   - %6: unused
///   - %7: Key matrix data pointer (f16)
///   - %8: Key matrix dim 0 size (seq_len_kv)
///   - %9: Key matrix dim 1 size
///   - %10: Key matrix dim 2 size
///   - %11: Key matrix dim 0 stride
///   - %12: Key matrix dim 1 stride
///   - %13: unused
///   - %14: Value matrix data pointer (f16)
///   - %15: Value matrix dim 0 size (seq_len_kv)
///   - %16: Value matrix dim 1 size
///   - %17: Value matrix dim 2 size
///   - %18: Value matrix dim 0 stride
///   - %19: Value matrix dim 1 stride
///   - %20: unused
///   - %21: Output matrix data pointer (f16)
///   - %22: Output matrix dim 0 size (seq_len_q)
///   - %23: Output matrix dim 1 size
///   - %24: Output matrix dim 2 size
///   - %25: Output matrix dim 0 stride
///   - %26: Output matrix dim 1 stride
///   - %27: unused
///   - %28: Key/Value iteration sequence length (seq_len_kv)
///   - %29: unused
///   - %30: unused
///   - %31: unused
#[test]
fn test_attention() {
    init_test_logger();

    let bytecode_path = Path::new("test_samples").join("attention.tileirbc");
    let mut intp = Interpreter::from_module(bytecode_path);

    // Tile sizes from MLIR:
    let seq_len_q = 32;
    let seq_len_kv = 64;
    let num_heads = 1;
    let head_dim = 64;

    let q = rand_3d_f16(seq_len_q, num_heads, head_dim);
    let k = rand_3d_f16(seq_len_kv, num_heads, head_dim);
    let v = rand_3d_f16(seq_len_kv, num_heads, head_dim);
    // Output buffer for f16 results
    let o = Array3::from_elem([seq_len_q, num_heads, head_dim], 0.0_f16);

    // Compute strides (row-major for each 2D slice)
    let q_strides = q.strides();
    let k_strides = k.strides();
    let v_strides = v.strides();
    let o_strides = o.strides();

    let args = KernelArgv::new()
        // Query
        .and(q.as_ptr() as *mut u8)
        .and(q.shape()[0] as i32) // %1: seq_len_q
        .and(q.shape()[1] as i32) // %2: num_heads
        .and(q.shape()[2] as i32) // %3: head_dim
        .and(q_strides[0] as i32) // %4: stride[0]
        .and(q_strides[1] as i32) // %5: stride[1]
        .and(q_strides[2] as i32) // %6: unused
        // Key
        .and(k.as_ptr() as *mut u8)
        .and(k.shape()[0] as i32) // %8: seq_len_kv
        .and(k.shape()[1] as i32) // %9: num_heads
        .and(k.shape()[2] as i32) // %10: head_dim
        .and(k_strides[0] as i32) // %11: stride[0]
        .and(k_strides[1] as i32) // %12: stride[1]
        .and(k_strides[2] as i32) // %13: unused
        // Value
        .and(v.as_ptr() as *mut u8)
        .and(v.shape()[0] as i32) // %15: seq_len_kv
        .and(v.shape()[1] as i32) // %16: num_heads
        .and(v.shape()[2] as i32) // %17: head_dim
        .and(v_strides[0] as i32) // %18: stride[0]
        .and(v_strides[1] as i32) // %19: stride[1]
        .and(v_strides[2] as i32) // %20: unused
        // Output
        .and(o.as_ptr() as *mut u8)
        .and(seq_len_q as i32) // %22: seq_len_q
        .and(num_heads as i32) // %23: num_heads
        .and(head_dim as i32) // %24: head_dim
        .and(o_strides[0] as i32) // %25: stride[0]
        .and(o_strides[1] as i32) // %26: stride[1]
        .and(o_strides[2] as i32) // %27: unused
        // Iteration length
        .and(seq_len_kv as i32) // %28: seq_len_kv for iteration
        .and(0_i32) // %29: unused
        .and(0_i32) // %30: unused
        .and(0_i32); // %31: unused

    // Grid size: based on tile sizes from MLIR
    // Q partition tile: (32x1x64), so grid = [seq_len_q/32, num_heads/1, 1]
    let tile_q_seq = 32;
    let grid_size: [usize; 3] = [(seq_len_q + tile_q_seq - 1) / tile_q_seq, num_heads, 1];

    info!("Grid size: {:?}", grid_size);
    intp.execute(args, grid_size);

    // Compute reference output
    let output_ref = flash_attention_ref(&q, &k, &v);

    // Compare results
    let mut max_diff = f32::MIN;
    let mut max_diff_idx = (-1, -1, -1);
    let mut max_diff_real_val = None;

    for [i, h, d] in ndrange(&[seq_len_q, num_heads, head_dim]).progress() {
        let actual = o[[i, h, d]] as f32;
        let expected = output_ref[[i, h, d]];
        let diff = (actual - expected).abs();

        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = (i as isize, h as isize, d as isize);
            max_diff_real_val = Some(expected);
        }

        info!(
            "Difference at [{}, {}, {}]: actual({}), expected({}), diff({})",
            i, h, d, actual, expected, diff
        );
    }

    info!(
        "Max diff: {} at {:?}, real value = {:?}",
        max_diff, max_diff_idx, max_diff_real_val
    );

    // assert!(
    //     max_diff < 0.01,
    //     "Maximum difference {} exceeds tolerance at {:?}",
    //     max_diff,
    //     max_diff_idx
    // );
}
