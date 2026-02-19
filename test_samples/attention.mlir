cuda_tile.module @kernels {
  entry @flash_attention(%0: tile<ptr<f16>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<ptr<f16>>, %8: tile<i32>, %9: tile<i32>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<ptr<f16>>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<i32>, %20: tile<i32>, %21: tile<ptr<f16>>, %22: tile<i32>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>, %30: tile<i32>, %31: tile<i32>) optimization_hints=<sm_75 = {}> {
    %32 = make_token : token
    %33 = constant <i32: 64> : tile<i32>
    %34 = assume div_by<16>, %0 : tile<ptr<f16>>
    %35 = assume bounded<0, ?>, %1 : tile<i32>
    %36 = assume bounded<0, ?>, %2 : tile<i32>
    %37 = assume div_by<16>, %36 : tile<i32>
    %38 = assume bounded<0, ?>, %3 : tile<i32>
    %39 = assume div_by<16>, %38 : tile<i32>
    %40 = assume bounded<0, ?>, %4 : tile<i32>
    %41 = assume div_by<8>, %40 : tile<i32>
    %42 = assume bounded<0, ?>, %5 : tile<i32>
    %43 = assume div_by<8>, %42 : tile<i32>
    %44 = make_tensor_view %34, shape = [%35, %37, %39], strides = [%41, %43, 1] : tile<i32> -> tensor_view<?x?x?xf16, strides=[?,?,1]>
    %45 = assume div_by<16>, %7 : tile<ptr<f16>>
    %46 = assume bounded<0, ?>, %8 : tile<i32>
    %47 = assume bounded<0, ?>, %9 : tile<i32>
    %48 = assume div_by<16>, %47 : tile<i32>
    %49 = assume bounded<0, ?>, %10 : tile<i32>
    %50 = assume div_by<16>, %49 : tile<i32>
    %51 = assume bounded<0, ?>, %11 : tile<i32>
    %52 = assume div_by<8>, %51 : tile<i32>
    %53 = assume bounded<0, ?>, %12 : tile<i32>
    %54 = assume div_by<8>, %53 : tile<i32>
    %55 = make_tensor_view %45, shape = [%46, %48, %50], strides = [%52, %54, 1] : tile<i32> -> tensor_view<?x?x?xf16, strides=[?,?,1]>
    %56 = assume div_by<16>, %14 : tile<ptr<f16>>
    %57 = assume bounded<0, ?>, %15 : tile<i32>
    %58 = assume bounded<0, ?>, %16 : tile<i32>
    %59 = assume div_by<16>, %58 : tile<i32>
    %60 = assume bounded<0, ?>, %17 : tile<i32>
    %61 = assume div_by<16>, %60 : tile<i32>
    %62 = assume bounded<0, ?>, %18 : tile<i32>
    %63 = assume div_by<8>, %62 : tile<i32>
    %64 = assume bounded<0, ?>, %19 : tile<i32>
    %65 = assume div_by<8>, %64 : tile<i32>
    %66 = make_tensor_view %56, shape = [%57, %59, %61], strides = [%63, %65, 1] : tile<i32> -> tensor_view<?x?x?xf16, strides=[?,?,1]>
    %67 = assume div_by<16>, %21 : tile<ptr<f16>>
    %68 = assume bounded<0, ?>, %22 : tile<i32>
    %69 = assume bounded<0, ?>, %23 : tile<i32>
    %70 = assume div_by<16>, %69 : tile<i32>
    %71 = assume bounded<0, ?>, %24 : tile<i32>
    %72 = assume div_by<16>, %71 : tile<i32>
    %73 = assume bounded<0, ?>, %25 : tile<i32>
    %74 = assume div_by<8>, %73 : tile<i32>
    %75 = assume bounded<0, ?>, %26 : tile<i32>
    %76 = assume div_by<8>, %75 : tile<i32>
    %77 = make_tensor_view %67, shape = [%68, %70, %72], strides = [%74, %76, 1] : tile<i32> -> tensor_view<?x?x?xf16, strides=[?,?,1]>
    %78, %79, %80 = get_tile_block_id : tile<i32>
    %81, %82, %83 = get_tile_block_id : tile<i32>
    %84 = divi %28, %33 signed rounding<positive_inf> : tile<i32>
    %85 = constant <i32: 0> : tile<i32>
    %86 = make_partition_view %44 : partition_view<masked tile=(32x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>
    %87, %88 = load_view_tko weak %86[%78, %82, %85] token = %32 : partition_view<masked tile=(32x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>, tile<i32> -> tile<32x1x64xf16>, token
    %89 = constant <f16: 1.442383e+00> : tile<f16>
    %90 = reshape %89 : tile<f16> -> tile<1x1x1xf16>
    %91 = broadcast %90 : tile<1x1x1xf16> -> tile<32x1x64xf16>
    %92 = mulf %87, %91  : tile<32x1x64xf16>
    %93 = constant <f32: 0.000000e+00> : tile<32x64xf32>
    %94 = constant <f32: 0.000000e+00> : tile<32x1xf32>
    %95 = constant <i32: 0> : tile<i32>
    %96 = constant <i32: 1> : tile<i32>
    %97 = constant <f32: 0.000000e+00> : tile<32x64xf32>
    %98 = constant <i32: 0> : tile<i32>
    %99 = reshape %92 : tile<32x1x64xf16> -> tile<32x64xf16>
    %100 = iota : tile<64xi32>
    %101 = reshape %28 : tile<i32> -> tile<1xi32>
    %102 = broadcast %101 : tile<1xi32> -> tile<64xi32>
    %103 = constant <f32: -1.000000e+07> : tile<f32>
    %104 = constant <f32: 0.000000e+00> : tile<f32>
    %105 = reshape %104 : tile<f32> -> tile<1xf32>
    %106 = broadcast %105 : tile<1xf32> -> tile<64xf32>
    %107 = reshape %103 : tile<f32> -> tile<1xf32>
    %108 = broadcast %107 : tile<1xf32> -> tile<64xf32>
    %109 = constant <i32: 0> : tile<i32>
    %141, %142 = for %110 in (%95 to %84, step %96) : tile<i32> iter_values(%111 = %93, %112 = %94) -> (tile<32x64xf32>, tile<32x1xf32>) {
      %113 = make_partition_view %55 : partition_view<masked tile=(64x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>
      %114, %115 = load_view_tko weak %113[%110, %82, %98] token = %32 : partition_view<masked tile=(64x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>, tile<i32> -> tile<64x1x64xf16>, token
      %116 = permute %114 [2, 1, 0] : tile<64x1x64xf16> -> tile<64x1x64xf16>
      %117 = reshape %116 : tile<64x1x64xf16> -> tile<64x64xf16>
      %118 = mmaf %99, %117, %97 : tile<32x64xf16>, tile<64x64xf16>, tile<32x64xf32>
      %119 = muli %110, %33 : tile<i32>
      %120 = reshape %119 : tile<i32> -> tile<1xi32>
      %121 = broadcast %120 : tile<1xi32> -> tile<64xi32>
      %122 = addi %100, %121 : tile<64xi32>
      %123 = cmpi less_than %122, %102, signed : tile<64xi32> -> tile<64xi1>
      %124 = select %123, %106, %108 : tile<64xi1>, tile<64xf32>
      %125 = reshape %124 : tile<64xf32> -> tile<1x64xf32>
      %126 = broadcast %125 : tile<1x64xf32> -> tile<32x64xf32>
      %127 = addf %118, %126  : tile<32x64xf32>
      %128 = exp2 %127 flush_to_zero : tile<32x64xf32>
      %132 = reduce %128 dim=1 identities=[0.000000e+00 : f32] : tile<32x64xf32> -> tile<32xf32>
      (%129: tile<f32>, %130: tile<f32>) {
        %131 = addf %129, %130  : tile<f32>
        yield %131 : tile<f32>
      }
      %133 = reshape %132 : tile<32xf32> -> tile<32x1xf32>
      %134 = addf %112, %133  : tile<32x1xf32>
      %135 = make_partition_view %66 : partition_view<masked tile=(64x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>
      %136, %137 = load_view_tko weak %135[%110, %82, %109] token = %32 : partition_view<masked tile=(64x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>, tile<i32> -> tile<64x1x64xf16>, token
      %138 = ftof %128  : tile<32x64xf32> -> tile<32x64xf16>
      %139 = reshape %136 : tile<64x1x64xf16> -> tile<64x64xf16>
      %140 = mmaf %138, %139, %111 : tile<32x64xf16>, tile<64x64xf16>, tile<32x64xf32>
      continue %140, %134 : tile<32x64xf32>, tile<32x1xf32>
    }
    %143 = broadcast %142 : tile<32x1xf32> -> tile<32x64xf32>
    %144 = divf %141, %143 rounding<approx> flush_to_zero : tile<32x64xf32>
    %145 = reshape %144 : tile<32x64xf32> -> tile<32x1x64xf32>
    %146 = constant <i32: 0> : tile<i32>
    %147 = ftof %145  : tile<32x1x64xf32> -> tile<32x1x64xf16>
    %148 = make_partition_view %77 : partition_view<tile=(32x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>
    %149 = store_view_tko weak %147, %148[%78, %82, %146] token = %32 : tile<32x1x64xf16>, partition_view<tile=(32x1x64), tensor_view<?x?x?xf16, strides=[?,?,1]>, dim_map=[0, 1, 2]>, tile<i32> -> token
    return
  }
}
