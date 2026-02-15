use core::range;

use ndarray::{Array3, Array4, Array5};

use crate::interpreter::data_structures::{elem_type::ElemType, tile::Tile};

#[test]
fn test_core_1() {
    let tile = Tile::iota(128, ElemType::I16);
    dbg!(&tile);

    let tile2 = Tile::zeros(&[8, 4, 1, 32], ElemType::F16);
    dbg!(tile2.shape());

    let tile3 = tile2.broadcast(&[2, 8, 4, 16, 32]);
    dbg!(tile3.shape());

    let tile4_shape = [2, 16, 4];
    let tile4 = tile.clone().reshape(&tile4_shape);
    dbg!(&tile4);

    let tile5 = tile4.permute(&[2, 0, 1]);
    dbg!(&tile5);

    for (i, j, k) in Array3::<i32>::zeros(tile4_shape.clone())
        .indexed_iter()
        .map(|a| (a.0 .0 as i64, a.0 .1 as i64, a.0 .2 as i64))
    {
        assert_eq!(
            tile4.get_scalar(&[i, j, k]),
            // permuted
            tile5.get_scalar(&[k, i, j]),
            "Mismatch at tile4[({}, {}, {})] and tile5[({}, {}, {})]: {:?} != {:?}",
            i,
            j,
            k,
            k,
            i,
            j,
            tile4.get_scalar(&[i, j, k]),
            tile5.get_scalar(&[k, i, j]),
        );
    }
}

#[test]
fn test_broadcast_with_reshape() {
    let tile = Tile::iota(128, ElemType::I16);

    let tile6_shape = [2, 1, 16, 4];
    let tile6 = tile.reshape(&tile6_shape);
    let tile6_b_shape = [8, 2, 4, 16, 4];
    let tile6_broadcast = tile6.broadcast(&tile6_b_shape);
    let tile6_bp = tile6_broadcast.permute(&[2, 4, 3, 0, 1]);

    let zeros_5d = Array5::<i32>::zeros(tile6_b_shape.clone());
    let range_5d = zeros_5d.indexed_iter().map(|a| {
        (
            a.0 .0 as i64,
            a.0 .1 as i64,
            a.0 .2 as i64,
            a.0 .3 as i64,
            a.0 .4 as i64,
        )
    });

    dbg!(tile6_broadcast.shape());
    dbg!(tile6_bp.shape());

    for (ia, ib, ic, id, ie) in range_5d {
        assert_eq!(
            tile6.get_scalar(&[ib, 0, id, ie]),
            tile6_broadcast.get_scalar(&[ia, ib, ic, id, ie]),
            "Mismatch at tile6[({ib}, 0, {id}, {ie})] and tile6_broadcast[({ia}, {ib}, {ic}, {id}, {ie})]: {:?} != {:?}",
            tile6.get_scalar(&[ib, 0, id, ie]),
            tile6_broadcast.get_scalar(&[ia, ib, ic, id, ie]),
        );
        assert_eq!(
            tile6.get_scalar(&[ib, 0, id, ie]),
            tile6_bp.get_scalar(&[ic, ie, id, ia ,ib]),
            "Boradcast&transpose mismatch:\nMismatch at tile6[({ib}, 0, {id}, {ie})] and tile6_bp[({ic}, {ie}, {id}, {ia}, {ib})]: {:?} != {:?}",
            tile6.get_scalar(&[ib, 0, id, ie]),
            tile6_bp.get_scalar(&[ic, ie, id, ia, ib]),
        )
    }
}

#[test]
fn test_cat() {
    let tile1_orig = Tile::iota(128, ElemType::I32);
    let tile1_shape = [2, 1, 16, 4];
    let tile1 = tile1_orig.reshape(&tile1_shape);

    let tile2_orig = Tile::iota(128, ElemType::I32);
    let tile2_shape = [2, 1, 4, 16];
    let tile2_reshaped = tile2_orig.reshape(&tile2_shape);
    let tile2 = tile2_reshaped.permute(&[0, 1, 3, 2]);

    let tile3 = tile1.cat(&tile2, 2);
    dbg!(tile3.shape());
    assert_eq!(tile3.shape(), vec![2, 1, 32, 4]);

    let zeros_4d = Array4::<i32>::zeros(tile1_shape.clone());
    let range_4d = zeros_4d
        .indexed_iter()
        .map(|a| (a.0 .0 as i64, a.0 .1 as i64, a.0 .2 as i64, a.0 .3 as i64));

    for (ia, ib, ic, id) in range_4d {
        assert_eq!(
            tile3.get_scalar(&[ia, ib, ic, id]),
            tile1.get_scalar(&[ia, ib, ic, id])
        );
        assert_eq!(
            tile3.get_scalar(&[ia, ib, ic + 16, id]),
            tile2_reshaped.get_scalar(&[ia, ib, id, ic])
        );
        println!("({ia}, {ib}, {ic}, {id}) test passed")
    }
}

#[should_panic(expected = "Broadcast failed")]
#[test]
fn test_core_invalid_broadcast() {
    let tile2 = Tile::zeros(&[8, 4, 1, 32], ElemType::F16);
    let tile4 = tile2.broadcast(&[2, 8, 4, 16, 4]);
    dbg!(tile4);
}

#[should_panic(expected = "Permutation is invalid")]
#[test]
fn test_core_invalid_permute() {
    let tile2 = Tile::zeros(&[8, 4, 1, 32], ElemType::F16);
    let tile4 = tile2.permute(&[0, 1, 3, 3]);
    dbg!(tile4);
}
