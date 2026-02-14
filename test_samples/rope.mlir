cuda_tile.module @kernels {
  entry @rope_kernel(%0: tile<ptr<f16>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<i32>, %10: tile<i32>, %11: tile<ptr<f16>>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<i32>, %19: tile<i32>, %20: tile<i32>, %21: tile<i32>, %22: tile<ptr<f16>>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>, %27: tile<i32>, %28: tile<i32>, %29: tile<i32>, %30: tile<i32>, %31: tile<ptr<f16>>, %32: tile<i32>, %33: tile<i32>, %34: tile<i32>, %35: tile<i32>, %36: tile<i32>, %37: tile<i32>, %38: tile<i32>, %39: tile<i32>, %40: tile<i32>, %41: tile<i32>, %42: tile<i32>, %43: tile<i32>, %44: tile<i32>) optimization_hints=<sm_90 = {}> {
    %45 = make_token : token
    %46 = constant <i32: 1> : tile<i32>
    %47 = assume div_by<16>, %0 : tile<ptr<f16>>
    %48 = assume bounded<0, ?>, %1 : tile<i32>
    %49 = assume bounded<0, ?>, %2 : tile<i32>
    %50 = assume bounded<0, ?>, %3 : tile<i32>
    %51 = assume bounded<0, ?>, %4 : tile<i32>
    %52 = assume bounded<0, ?>, %5 : tile<i32>
    %53 = assume div_by<16>, %52 : tile<i32>
    %54 = assume bounded<0, ?>, %6 : tile<i32>
    %55 = assume div_by<8>, %54 : tile<i32>
    %56 = assume bounded<0, ?>, %7 : tile<i32>
    %57 = assume div_by<8>, %56 : tile<i32>
    %58 = assume bounded<0, ?>, %8 : tile<i32>
    %59 = assume div_by<8>, %58 : tile<i32>
    %60 = assume bounded<0, ?>, %9 : tile<i32>
    %61 = assume div_by<8>, %60 : tile<i32>
    %62 = make_tensor_view %47, shape = [%48, %49, %50, %51, %53], strides = [%55, %57, %59, %61, 1] : tile<i32> -> tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>
    %63 = assume div_by<16>, %11 : tile<ptr<f16>>
    %64 = assume bounded<0, ?>, %12 : tile<i32>
    %65 = assume bounded<0, ?>, %13 : tile<i32>
    %66 = assume bounded<0, ?>, %14 : tile<i32>
    %67 = assume bounded<0, ?>, %15 : tile<i32>
    %68 = assume bounded<0, ?>, %16 : tile<i32>
    %69 = assume div_by<16>, %68 : tile<i32>
    %70 = assume bounded<0, ?>, %17 : tile<i32>
    %71 = assume div_by<8>, %70 : tile<i32>
    %72 = assume bounded<0, ?>, %18 : tile<i32>
    %73 = assume div_by<8>, %72 : tile<i32>
    %74 = assume bounded<0, ?>, %19 : tile<i32>
    %75 = assume div_by<8>, %74 : tile<i32>
    %76 = assume bounded<0, ?>, %20 : tile<i32>
    %77 = assume div_by<8>, %76 : tile<i32>
    %78 = make_tensor_view %63, shape = [%64, %65, %66, %67, %69], strides = [%71, %73, %75, %77, 1] : tile<i32> -> tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>
    %79 = assume div_by<16>, %22 : tile<ptr<f16>>
    %80 = assume bounded<0, ?>, %23 : tile<i32>
    %81 = assume bounded<0, ?>, %24 : tile<i32>
    %82 = assume bounded<0, ?>, %25 : tile<i32>
    %83 = assume bounded<0, ?>, %26 : tile<i32>
    %84 = assume div_by<16>, %83 : tile<i32>
    %85 = assume bounded<0, ?>, %27 : tile<i32>
    %86 = assume div_by<8>, %85 : tile<i32>
    %87 = assume bounded<0, ?>, %28 : tile<i32>
    %88 = assume div_by<8>, %87 : tile<i32>
    %89 = assume bounded<0, ?>, %29 : tile<i32>
    %90 = assume div_by<8>, %89 : tile<i32>
    %91 = make_tensor_view %79, shape = [%80, %81, %82, %84], strides = [%86, %88, %90, 1] : tile<i32> -> tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>
    %92 = assume div_by<16>, %31 : tile<ptr<f16>>
    %93 = assume bounded<0, ?>, %32 : tile<i32>
    %94 = assume bounded<0, ?>, %33 : tile<i32>
    %95 = assume bounded<0, ?>, %34 : tile<i32>
    %96 = assume bounded<0, ?>, %35 : tile<i32>
    %97 = assume div_by<16>, %96 : tile<i32>
    %98 = assume bounded<0, ?>, %36 : tile<i32>
    %99 = assume div_by<8>, %98 : tile<i32>
    %100 = assume bounded<0, ?>, %37 : tile<i32>
    %101 = assume div_by<8>, %100 : tile<i32>
    %102 = assume bounded<0, ?>, %38 : tile<i32>
    %103 = assume div_by<8>, %102 : tile<i32>
    %104 = make_tensor_view %92, shape = [%93, %94, %95, %97], strides = [%99, %101, %103, 1] : tile<i32> -> tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>
    %105, %106, %107 = get_tile_block_id : tile<i32>
    %108 = divi %105, %46 signed rounding<negative_inf> : tile<i32>
    %109 = remi %105, %46 signed : tile<i32>
    %110 = constant <i32: 0> : tile<i32>
    %111 = cmpi less_than %109, %110, signed : tile<i32> -> tile<i1>
    %112 = constant <i1: false> : tile<i1>
    %113 = xori %111, %112 : tile<i1>
    %114 = cmpi not_equal %109, %110, signed : tile<i32> -> tile<i1>
    %115 = andi %113, %114 : tile<i1>
    %116 = addi %109, %46 : tile<i32>
    %117 = select %115, %116, %109 : tile<i1>, tile<i32>
    %118 = constant <i32: 1> : tile<i32>
    %119 = cmpi equal %80, %118, signed : tile<i32> -> tile<i1>
    %121 = if %119 -> (tile<i32>) {
      %120 = constant <i32: 0> : tile<i32>
      yield %120 : tile<i32>
    } else {
      yield %108 : tile<i32>
    }
    %122 = constant <i32: 0> : tile<i32>
    %123 = constant <i32: 0> : tile<i32>
    %124 = make_partition_view %91 : partition_view<masked tile=(1x1x1x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>
    %125, %126 = load_view_tko weak %124[%121, %117, %122, %123] token = %45 : partition_view<masked tile=(1x1x1x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x1x64xf16>, token
    %127 = reshape %125 : tile<1x1x1x64xf16> -> tile<1x64xf16>
    %128 = constant <i32: 0> : tile<i32>
    %129 = constant <i32: 0> : tile<i32>
    %130 = make_partition_view %104 : partition_view<masked tile=(1x1x1x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>
    %131, %132 = load_view_tko weak %130[%121, %117, %128, %129] token = %45 : partition_view<masked tile=(1x1x1x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x1x64xf16>, token
    %133 = reshape %131 : tile<1x1x1x64xf16> -> tile<1x64xf16>
    %134 = constant <i32: 0> : tile<i32>
    %135 = constant <i32: 0> : tile<i32>
    %136 = constant <i32: 0> : tile<i32>
    %137 = make_partition_view %62 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %138, %139 = load_view_tko weak %137[%108, %134, %117, %135, %136] token = %45 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> tile<1x8x1x1x64xf16>, token
    %140 = join_tokens %45, %139 : token
    %141 = reshape %138 : tile<1x8x1x1x64xf16> -> tile<8x64xf16>
    %142 = constant <i32: 0> : tile<i32>
    %143 = constant <i32: 1> : tile<i32>
    %144 = constant <i32: 0> : tile<i32>
    %145 = make_partition_view %62 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %146, %147 = load_view_tko weak %145[%108, %142, %117, %143, %144] token = %45 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> tile<1x8x1x1x64xf16>, token
    %148 = join_tokens %140, %147 : token
    %149 = reshape %146 : tile<1x8x1x1x64xf16> -> tile<8x64xf16>
    %150 = broadcast %127 : tile<1x64xf16> -> tile<8x64xf16>
    %151 = broadcast %133 : tile<1x64xf16> -> tile<8x64xf16>
    %152 = mulf %149, %151  : tile<8x64xf16>
    %153 = negf %152 : tile<8x64xf16>
    %154 = fma %141, %150, %153  : tile<8x64xf16>
    %155 = broadcast %127 : tile<1x64xf16> -> tile<8x64xf16>
    %156 = broadcast %133 : tile<1x64xf16> -> tile<8x64xf16>
    %157 = mulf %141, %156  : tile<8x64xf16>
    %158 = fma %149, %155, %157  : tile<8x64xf16>
    %159 = constant <i32: 0> : tile<i32>
    %160 = constant <i32: 0> : tile<i32>
    %161 = constant <i32: 0> : tile<i32>
    %162 = reshape %154 : tile<8x64xf16> -> tile<1x8x1x1x64xf16>
    %163 = make_partition_view %62 : partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %164 = store_view_tko weak %162, %163[%108, %159, %117, %160, %161] token = %148 : tile<1x8x1x1x64xf16>, partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> token
    %165 = constant <i32: 0> : tile<i32>
    %166 = constant <i32: 1> : tile<i32>
    %167 = constant <i32: 0> : tile<i32>
    %168 = reshape %158 : tile<8x64xf16> -> tile<1x8x1x1x64xf16>
    %169 = make_partition_view %62 : partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %170 = store_view_tko weak %168, %169[%108, %165, %117, %166, %167] token = %164 : tile<1x8x1x1x64xf16>, partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> token
    %171 = constant <i32: 0> : tile<i32>
    %172 = constant <i32: 0> : tile<i32>
    %173 = constant <i32: 0> : tile<i32>
    %174 = make_partition_view %78 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %175, %176 = load_view_tko weak %174[%108, %171, %117, %172, %173] token = %45 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> tile<1x8x1x1x64xf16>, token
    %177 = join_tokens %45, %176 : token
    %178 = reshape %175 : tile<1x8x1x1x64xf16> -> tile<8x64xf16>
    %179 = constant <i32: 0> : tile<i32>
    %180 = constant <i32: 1> : tile<i32>
    %181 = constant <i32: 0> : tile<i32>
    %182 = make_partition_view %78 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %183, %184 = load_view_tko weak %182[%108, %179, %117, %180, %181] token = %45 : partition_view<masked tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> tile<1x8x1x1x64xf16>, token
    %185 = join_tokens %177, %184 : token
    %186 = reshape %183 : tile<1x8x1x1x64xf16> -> tile<8x64xf16>
    %187 = broadcast %127 : tile<1x64xf16> -> tile<8x64xf16>
    %188 = broadcast %133 : tile<1x64xf16> -> tile<8x64xf16>
    %189 = mulf %186, %188  : tile<8x64xf16>
    %190 = negf %189 : tile<8x64xf16>
    %191 = fma %178, %187, %190  : tile<8x64xf16>
    %192 = broadcast %127 : tile<1x64xf16> -> tile<8x64xf16>
    %193 = broadcast %133 : tile<1x64xf16> -> tile<8x64xf16>
    %194 = mulf %178, %193  : tile<8x64xf16>
    %195 = fma %186, %192, %194  : tile<8x64xf16>
    %196 = constant <i32: 0> : tile<i32>
    %197 = constant <i32: 0> : tile<i32>
    %198 = constant <i32: 0> : tile<i32>
    %199 = reshape %191 : tile<8x64xf16> -> tile<1x8x1x1x64xf16>
    %200 = make_partition_view %78 : partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %201 = store_view_tko weak %199, %200[%108, %196, %117, %197, %198] token = %185 : tile<1x8x1x1x64xf16>, partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> token
    %202 = constant <i32: 0> : tile<i32>
    %203 = constant <i32: 1> : tile<i32>
    %204 = constant <i32: 0> : tile<i32>
    %205 = reshape %195 : tile<8x64xf16> -> tile<1x8x1x1x64xf16>
    %206 = make_partition_view %78 : partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>
    %207 = store_view_tko weak %205, %206[%108, %202, %117, %203, %204] token = %201 : tile<1x8x1x1x64xf16>, partition_view<tile=(1x8x1x1x64), tensor_view<?x?x?x?x?xf16, strides=[?,?,?,?,1]>, dim_map=[0, 1, 2, 3, 4]>, tile<i32> -> token
    return
  }
}
