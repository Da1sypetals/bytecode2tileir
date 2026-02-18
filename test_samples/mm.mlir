cuda_tile.module @kernels {
  entry @matmul_kernel(%0: tile<ptr<f16>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<ptr<f16>>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<i32>, %10: tile<ptr<f32>>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>) optimization_hints=<sm_75 = {}> {
    %18 = make_token : token
    %19 = constant <i32: 16> : tile<i32>
    %20 = assume div_by<16>, %0 : tile<ptr<f16>>
    %21 = assume bounded<0, ?>, %1 : tile<i32>
    %22 = assume div_by<16>, %21 : tile<i32>
    %23 = assume bounded<0, ?>, %2 : tile<i32>
    %24 = assume div_by<16>, %23 : tile<i32>
    %25 = assume bounded<0, ?>, %3 : tile<i32>
    %26 = assume div_by<8>, %25 : tile<i32>
    %27 = make_tensor_view %20, shape = [%22, %24], strides = [%26, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %28 = assume div_by<16>, %5 : tile<ptr<f16>>
    %29 = assume bounded<0, ?>, %6 : tile<i32>
    %30 = assume div_by<16>, %29 : tile<i32>
    %31 = assume bounded<0, ?>, %7 : tile<i32>
    %32 = assume div_by<16>, %31 : tile<i32>
    %33 = assume bounded<0, ?>, %8 : tile<i32>
    %34 = assume div_by<8>, %33 : tile<i32>
    %35 = make_tensor_view %28, shape = [%30, %32], strides = [%34, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %36 = assume div_by<16>, %10 : tile<ptr<f32>>
    %37 = assume bounded<0, ?>, %11 : tile<i32>
    %38 = assume div_by<16>, %37 : tile<i32>
    %39 = assume bounded<0, ?>, %12 : tile<i32>
    %40 = assume div_by<16>, %39 : tile<i32>
    %41 = assume bounded<0, ?>, %13 : tile<i32>
    %42 = assume div_by<4>, %41 : tile<i32>
    %43 = make_tensor_view %36, shape = [%38, %40], strides = [%42, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
    %44, %45, %46 = get_tile_block_id : tile<i32>
    %47, %48, %49 = get_tile_block_id : tile<i32>
    %50 = divi %24, %19 signed rounding<negative_inf> : tile<i32>
    %51 = constant <f32: 0.000000e+00> : tile<64x32xf32>
    %52 = constant <i32: 0> : tile<i32>
    %53 = constant <i32: 1> : tile<i32>
    %63 = for %54 in (%52 to %50, step %53) : tile<i32> iter_values(%55 = %51) -> (tile<64x32xf32>) {
      %56 = make_partition_view %27 : partition_view<tile=(64x16), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
      %57, %58 = load_view_tko weak %56[%44, %54] token = %18 : partition_view<tile=(64x16), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<64x16xf16>, token
      %59 = make_partition_view %35 : partition_view<tile=(16x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
      %60, %61 = load_view_tko weak %59[%54, %48] token = %18 : partition_view<tile=(16x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<16x32xf16>, token
      %62 = mmaf %57, %60, %55 : tile<64x16xf16>, tile<16x32xf16>, tile<64x32xf32>
      continue %62 : tile<64x32xf32>
    }
    %64 = make_partition_view %43 : partition_view<tile=(64x32), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>
    %65 = store_view_tko weak %63, %64[%44, %48] token = %18 : tile<64x32xf32>, partition_view<tile=(64x32), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
    return
  }
}
