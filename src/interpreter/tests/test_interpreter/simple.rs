use crate::interpreter::tests::test_interpreter::logging_utils::init_test_logger;
use crate::interpreter::{
    args::{KernelArgv, KernelGenericArgv},
    data_structures::interpreter::Interpreter,
};
use core::f32;
use indicatif::ProgressIterator;
use log::info;
use ndarray::Array2;
use ndrange::ndrange;
use rand::RngExt;
use std::path::Path;

fn rand_2d_f16(row: usize, col: usize) -> Array2<f16> {
    let mut rng = rand::rng();
    Array2::from_elem([row, col], 0.0_f16).mapv_into(|_| rng.random_range(0.0..1.0_f32) as f16)
}

fn rand_2d_f32(row: usize, col: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    Array2::from_elem([row, col], 0.0_f32).mapv_into(|_| rng.random_range(0.0..1.0_f32))
}

//   - %0: base pointer of A
//   - %1: rows of A (M, shape[0])
//   - %2: columns of A (K, shape[1])
//   - %3: leading dimension of A (stride[0])
//   - %4: unused/reserved
//   - %5: base pointer of B
//   - %6: rows of B (K, shape[0])
//   - %7: columns of B (N, shape[1])
//   - %8: leading dimension of B (stride[0])
//   - %9: unused/reserved
//   - %10: base pointer of C (output)
//   - %11: rows of C (M, shape[0])
//   - %12: columns of C (N, shape[1])
//   - %13: leading dimension of C (stride[0])
//   - %14: unused/reserved
//   - %15: rows of A (M, duplicate)
//   - %16: columns of B (N, duplicate)
//   - %17: columns of A / rows of B (K, duplicate)
#[test]
fn test_matmul_1() {
    init_test_logger();

    let bytecode_path = Path::new("test_samples").join("mm.tileirbc");
    let mut intp = Interpreter::from_module(bytecode_path);

    let [m, n, k] = [128, 64, 128];

    // Consistent with TileIR
    let [tm, tn, tk] = [64, 32, 16];

    let a = rand_2d_f16(m, k);
    let b = rand_2d_f16(k, n);
    let c = Array2::<f32>::zeros([m, n]);

    let args = KernelArgv::new()
        .and(a.as_ptr() as *mut u8)
        .and(a.shape()[0] as i32)
        .and(a.shape()[1] as i32)
        .and(a.strides()[0] as i32)
        .and(a.strides()[1] as i32)
        .and(b.as_ptr() as *mut u8)
        .and(b.shape()[0] as i32)
        .and(b.shape()[1] as i32)
        .and(b.strides()[0] as i32)
        .and(b.strides()[1] as i32)
        .and(c.as_ptr() as *mut u8)
        .and(c.shape()[0] as i32)
        .and(c.shape()[1] as i32)
        .and(c.strides()[0] as i32)
        .and(c.strides()[1] as i32)
        .and(a.shape()[0] as i32)
        .and(b.shape()[1] as i32)
        .and(a.shape()[1] as i32);

    let grid_size = [m / tm, n / tn, 1];
    intp.execute(args, grid_size);

    let mut max_diff = f32::MIN;
    let mut max_diff_idx = [-1, -1];
    let mut max_diff_real_val = None;
    for [i_m, i_n] in ndrange(&[m, n]).progress() {
        let c_ref = (0..k).map(|i_k| a[[i_m, i_k]] * b[[i_k, i_n]]).sum::<f16>() as f32;

        let diff = (c[[i_m, i_n]] - c_ref).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = [i_m as isize, i_n as isize];
            max_diff_real_val = Some(c_ref);
        }

        info!(
            "Difference at [{}, {}] : {} ; actual({}), expected({})",
            i_m,
            i_n,
            diff,
            c[[i_m, i_n]],
            c_ref
        );

        // assert!(
        //     c_ref[[i_m, i_n]] == c[[i_m, i_n]],
        //     "Mismatch at [{}, {}] : actual({}) != expected({})",
        //     i_m,
        //     i_n,
        //     c[[i_m, i_n]],
        //     c_ref[[i_m, i_n]]
        // );
    }

    println!("Max diff: {max_diff} at {max_diff_idx:?}, real value = {max")
}
