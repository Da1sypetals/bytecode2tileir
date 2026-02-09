cuda_tile.module @kernels {
  entry @matmul_kernel(%0: tile<ptr<f16>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<ptr<f16>>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<i32>, %10: tile<ptr<f16>>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>) optimization_hints=<sm_90 = {}> {
    %18 = make_token : token
    %19 = constant <i32: 128> : tile<i32>
    %20 = constant <i32: 128> : tile<i32>
    %21 = assume div_by<16>, %0 : tile<ptr<f16>>
    %22 = assume bounded<0, ?>, %1 : tile<i32>
    %23 = assume div_by<16>, %22 : tile<i32>
    %24 = assume bounded<0, ?>, %2 : tile<i32>
    %25 = assume div_by<16>, %24 : tile<i32>
    %26 = assume bounded<0, ?>, %3 : tile<i32>
    %27 = assume div_by<8>, %26 : tile<i32>
    %28 = make_tensor_view %21, shape = [%23, %25], strides = [%27, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %29 = assume div_by<16>, %5 : tile<ptr<f16>>
    %30 = assume bounded<0, ?>, %6 : tile<i32>
    %31 = assume div_by<16>, %30 : tile<i32>
    %32 = assume bounded<0, ?>, %7 : tile<i32>
    %33 = assume div_by<16>, %32 : tile<i32>
    %34 = assume bounded<0, ?>, %8 : tile<i32>
    %35 = assume div_by<8>, %34 : tile<i32>
    %36 = make_tensor_view %29, shape = [%31, %33], strides = [%35, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %37 = assume div_by<16>, %10 : tile<ptr<f16>>
    %38 = assume bounded<0, ?>, %11 : tile<i32>
    %39 = assume div_by<16>, %38 : tile<i32>
    %40 = assume bounded<0, ?>, %12 : tile<i32>
    %41 = assume div_by<16>, %40 : tile<i32>
    %42 = assume bounded<0, ?>, %13 : tile<i32>
    %43 = assume div_by<8>, %42 : tile<i32>
    %44 = make_tensor_view %37, shape = [%39, %41], strides = [%43, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %45 = constant <i32: 8> : tile<i32>
    %46, %47, %48 = get_tile_block_id : tile<i32>
    %49 = divi %23, %19 signed rounding<positive_inf> : tile<i32>
    %50 = divi %33, %20 signed rounding<positive_inf> : tile<i32>
    %51 = muli %45, %50 : tile<i32>
    %52 = divi %46, %51 signed rounding<negative_inf> : tile<i32>
    %53 = muli %52, %45 : tile<i32>
    %54 = subi %49, %53 : tile<i32>
    %55 = mini %54, %45 signed : tile<i32>
    %56 = remi %46, %55 signed : tile<i32>
    %57 = constant <i32: 0> : tile<i32>
    %58 = cmpi less_than %56, %57, signed : tile<i32> -> tile<i1>
    %59 = cmpi less_than %55, %57, signed : tile<i32> -> tile<i1>
    %60 = xori %58, %59 : tile<i1>
    %61 = cmpi not_equal %56, %57, signed : tile<i32> -> tile<i1>
    %62 = andi %60, %61 : tile<i1>
    %63 = addi %56, %55 : tile<i32>
    %64 = select %62, %63, %56 : tile<i1>, tile<i32>
    %65 = addi %53, %64 : tile<i32>
    %66 = remi %46, %51 signed : tile<i32>
    %67 = constant <i32: 0> : tile<i32>
    %68 = cmpi less_than %66, %67, signed : tile<i32> -> tile<i1>
    %69 = cmpi less_than %51, %67, signed : tile<i32> -> tile<i1>
    %70 = xori %68, %69 : tile<i1>
    %71 = cmpi not_equal %66, %67, signed : tile<i32> -> tile<i1>
    %72 = andi %70, %71 : tile<i1>
    %73 = addi %66, %51 : tile<i32>
    %74 = select %72, %73, %66 : tile<i1>, tile<i32>
    %75 = divi %74, %55 signed rounding<negative_inf> : tile<i32>
    %76 = make_partition_view %28 : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    %77, %78 = get_index_space_shape %76 : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]> -> tile<i32>
    %79 = constant <f32: 0.000000e+00> : tile<128x128xf32>
    %80 = constant <i32: 0> : tile<i32>
    %81 = constant <i32: 1> : tile<i32>
    %91 = for %82 in (%80 to %78, step %81) : tile<i32> iter_values(%83 = %79) -> (tile<128x128xf32>) {
      %84 = make_partition_view %28 : partition_view<masked tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
      %85, %86 = load_view_tko weak %84[%65, %82] token = %18 : partition_view<masked tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<128x32xf16>, token
      %87 = make_partition_view %36 : partition_view<masked tile=(32x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
      %88, %89 = load_view_tko weak %87[%82, %75] token = %18 : partition_view<masked tile=(32x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<32x128xf16>, token
      %90 = mmaf %85, %88, %83 : tile<128x32xf16>, tile<32x128xf16>, tile<128x128xf32>
      continue %90 : tile<128x128xf32>
    }
    %92 = ftof %91  : tile<128x128xf32> -> tile<128x128xf16>
    %93 = make_partition_view %44 : partition_view<tile=(128x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    %94 = store_view_tko weak %92, %93[%65, %75] token = %18 : tile<128x128xf16>, partition_view<tile=(128x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
    return
  }
}
