cuda_tile.module @kernels {
  entry @fmha_kernel(%0: tile<ptr<f16>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f16>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<ptr<f16>>, %19: tile<i32>, %20: tile<i32>, %21: tile<i32>, %22: tile<i32>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>, %27: tile<ptr<f16>>, %28: tile<i32>, %29: tile<i32>, %30: tile<i32>, %31: tile<i32>, %32: tile<i32>, %33: tile<i32>, %34: tile<i32>, %35: tile<i32>, %36: tile<f32>, %37: tile<i32>, %38: tile<i32>, %39: tile<i32>, %40: tile<i32>, %41: tile<i32>, %42: tile<i32>, %43: tile<i1>, %44: tile<i1>) optimization_hints=<sm_90 = {}> {
    %45 = make_token : token
    %46 = constant <i32: 8> : tile<i32>
    %47 = constant <i32: 128> : tile<i32>
    %48 = constant <i32: 128> : tile<i32>
    %49 = constant <i32: 1> : tile<i32>
    %50 = assume div_by<16>, %0 : tile<ptr<f16>>
    %51 = assume bounded<0, ?>, %1 : tile<i32>
    %52 = assume bounded<0, ?>, %2 : tile<i32>
    %53 = assume bounded<0, ?>, %3 : tile<i32>
    %54 = assume bounded<0, ?>, %4 : tile<i32>
    %55 = assume div_by<16>, %54 : tile<i32>
    %56 = assume bounded<0, ?>, %5 : tile<i32>
    %57 = assume div_by<8>, %56 : tile<i32>
    %58 = assume bounded<0, ?>, %6 : tile<i32>
    %59 = assume div_by<8>, %58 : tile<i32>
    %60 = assume bounded<0, ?>, %7 : tile<i32>
    %61 = assume div_by<8>, %60 : tile<i32>
    %62 = make_tensor_view %50, shape = [%51, %52, %53, %55], strides = [%57, %59, %61, 1] : tile<i32> -> tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>
    %63 = assume div_by<16>, %9 : tile<ptr<f16>>
    %64 = assume bounded<0, ?>, %10 : tile<i32>
    %65 = assume bounded<0, ?>, %11 : tile<i32>
    %66 = assume bounded<0, ?>, %12 : tile<i32>
    %67 = assume div_by<16>, %66 : tile<i32>
    %68 = assume bounded<0, ?>, %13 : tile<i32>
    %69 = assume bounded<0, ?>, %14 : tile<i32>
    %70 = assume div_by<8>, %69 : tile<i32>
    %71 = assume bounded<0, ?>, %15 : tile<i32>
    %72 = assume div_by<8>, %71 : tile<i32>
    %73 = make_tensor_view %63, shape = [%64, %65, %67, %68], strides = [%70, %72, 1, 1] : tile<i32> -> tensor_view<?x?x?x?xf16, strides=[?,?,1,1]>
    %74 = assume div_by<16>, %18 : tile<ptr<f16>>
    %75 = assume bounded<0, ?>, %19 : tile<i32>
    %76 = assume bounded<0, ?>, %20 : tile<i32>
    %77 = assume bounded<0, ?>, %21 : tile<i32>
    %78 = assume bounded<0, ?>, %22 : tile<i32>
    %79 = assume div_by<16>, %78 : tile<i32>
    %80 = assume bounded<0, ?>, %23 : tile<i32>
    %81 = assume div_by<8>, %80 : tile<i32>
    %82 = assume bounded<0, ?>, %24 : tile<i32>
    %83 = assume div_by<8>, %82 : tile<i32>
    %84 = assume bounded<0, ?>, %25 : tile<i32>
    %85 = assume div_by<8>, %84 : tile<i32>
    %86 = make_tensor_view %74, shape = [%75, %76, %77, %79], strides = [%81, %83, %85, 1] : tile<i32> -> tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>
    %87 = assume div_by<16>, %27 : tile<ptr<f16>>
    %88 = assume bounded<0, ?>, %28 : tile<i32>
    %89 = assume bounded<0, ?>, %29 : tile<i32>
    %90 = assume bounded<0, ?>, %30 : tile<i32>
    %91 = assume bounded<0, ?>, %31 : tile<i32>
    %92 = assume div_by<16>, %91 : tile<i32>
    %93 = assume bounded<0, ?>, %32 : tile<i32>
    %94 = assume div_by<8>, %93 : tile<i32>
    %95 = assume bounded<0, ?>, %33 : tile<i32>
    %96 = assume div_by<8>, %95 : tile<i32>
    %97 = assume bounded<0, ?>, %34 : tile<i32>
    %98 = assume div_by<8>, %97 : tile<i32>
    %99 = make_tensor_view %87, shape = [%88, %89, %90, %92], strides = [%94, %96, %98, 1] : tile<i32> -> tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>
    %100, %101, %102 = get_tile_block_id : tile<i32>
    %103, %104, %105 = get_tile_block_id : tile<i32>
    %106 = divi %104, %46 signed rounding<negative_inf> : tile<i32>
    %107 = remi %104, %46 signed : tile<i32>
    %108 = constant <i32: 0> : tile<i32>
    %109 = cmpi less_than %107, %108, signed : tile<i32> -> tile<i1>
    %110 = constant <i1: false> : tile<i1>
    %111 = xori %109, %110 : tile<i1>
    %112 = cmpi not_equal %107, %108, signed : tile<i32> -> tile<i1>
    %113 = andi %111, %112 : tile<i1>
    %114 = addi %107, %46 : tile<i32>
    %115 = select %113, %114, %107 : tile<i1>, tile<i32>
    %116 = divi %115, %49 signed rounding<negative_inf> : tile<i32>
    %117 = constant <f32: 1.442695e+00> : tile<f32>
    %118 = mulf %36, %117  : tile<f32>
    %119 = muli %100, %47 : tile<i32>
    %120 = iota : tile<128xi32>
    %121 = reshape %119 : tile<i32> -> tile<1xi32>
    %122 = broadcast %121 : tile<1xi32> -> tile<128xi32>
    %123 = addi %122, %120 : tile<128xi32>
    %124 = reshape %37 : tile<i32> -> tile<1xi32>
    %125 = broadcast %124 : tile<1xi32> -> tile<128xi32>
    %126 = addi %123, %125 : tile<128xi32>
    %127 = reshape %126 : tile<128xi32> -> tile<128x1xi32>
    %128 = iota : tile<128xi32>
    %129 = reshape %128 : tile<128xi32> -> tile<1x128xi32>
    %130 = constant <f32: 0xFF800000> : tile<128x1xf32>
    %131 = constant <f32: 0.000000e+00> : tile<128x1xf32>
    %132 = constant <f32: 0.000000e+00> : tile<128x64xf32>
    %133 = constant <i32: 0> : tile<i32>
    %134 = make_partition_view %62 : partition_view<tile=(1x1x128x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>
    %135, %136 = load_view_tko weak %134[%106, %115, %100, %133] token = %45 : partition_view<tile=(1x1x128x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x128x64xf16>, token
    %137 = reshape %135 : tile<1x1x128x64xf16> -> tile<128x64xf16>
    %138 = constant <i32: 1> : tile<i32>
    %139 = addi %100, %138 : tile<i32>
    %140 = muli %139, %47 : tile<i32>
    %141 = addi %37, %140 : tile<i32>
    %142 = muli %100, %47 : tile<i32>
    %143 = addi %37, %142 : tile<i32>
    %144 = divi %143, %48 signed rounding<negative_inf> : tile<i32>
    %145 = divi %67, %48 signed rounding<negative_inf> : tile<i32>
    %146 = mini %144, %145 signed : tile<i32>
    %147 = mini %141, %67 signed : tile<i32>
    %148 = divi %147, %48 signed rounding<positive_inf> : tile<i32>
    %149 = constant <i32: 0> : tile<i32>
    %150 = constant <i32: 1> : tile<i32>
    %151 = constant <i32: 0> : tile<i32>
    %152 = constant <f32: 0.000000e+00> : tile<128x128xf32>
    %153 = reshape %118 : tile<f32> -> tile<1x1xf32>
    %154 = broadcast %153 : tile<1x1xf32> -> tile<128x1xf32>
    %155 = reshape %118 : tile<f32> -> tile<1x1xf32>
    %156 = broadcast %155 : tile<1x1xf32> -> tile<128x128xf32>
    %157 = constant <i32: 0> : tile<i32>
    %213, %214, %215 = for %158 in (%149 to %148, step %150) : tile<i32> iter_values(%159 = %132, %160 = %131, %161 = %130) -> (tile<128x64xf32>, tile<128x1xf32>, tile<128x1xf32>) {
      %162 = make_partition_view %73 : partition_view<tile=(1x1x64x128), tensor_view<?x?x?x?xf16, strides=[?,?,1,1]>, dim_map=[0, 1, 3, 2]>
      %163, %164 = load_view_tko weak %162[%106, %116, %151, %158] token = %45 optimization_hints = <sm_90 = {latency = 2}> : partition_view<tile=(1x1x64x128), tensor_view<?x?x?x?xf16, strides=[?,?,1,1]>, dim_map=[0, 1, 3, 2]>, tile<i32> -> tile<1x1x64x128xf16>, token
      %165 = reshape %163 : tile<1x1x64x128xf16> -> tile<64x128xf16>
      %166 = mmaf %137, %165, %152 : tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
      %167 = cmpi greater_than_or_equal %158, %146, signed : tile<i32> -> tile<i1>
      %185 = if %167 -> (tile<128x128xf32>) {
        %168 = muli %158, %48 : tile<i32>
        %169 = reshape %168 : tile<i32> -> tile<1x1xi32>
        %170 = broadcast %169 : tile<1x1xi32> -> tile<1x128xi32>
        %171 = addi %170, %129 : tile<1x128xi32>
        %172 = constant <i1: true> : tile<128x128xi1>
        %173 = broadcast %127 : tile<128x1xi32> -> tile<128x128xi32>
        %174 = broadcast %171 : tile<1x128xi32> -> tile<128x128xi32>
        %175 = cmpi greater_than_or_equal %173, %174, signed : tile<128x128xi32> -> tile<128x128xi1>
        %176 = andi %172, %175 : tile<128x128xi1>
        %177 = constant <f32: 0.000000e+00> : tile<f32>
        %178 = constant <f32: 0xFF800000> : tile<f32>
        %179 = reshape %177 : tile<f32> -> tile<1x1xf32>
        %180 = broadcast %179 : tile<1x1xf32> -> tile<128x128xf32>
        %181 = reshape %178 : tile<f32> -> tile<1x1xf32>
        %182 = broadcast %181 : tile<1x1xf32> -> tile<128x128xf32>
        %183 = select %176, %180, %182 : tile<128x128xi1>, tile<128x128xf32>
        %184 = addf %166, %183  : tile<128x128xf32>
        yield %184 : tile<128x128xf32>
      } else {
        yield %166 : tile<128x128xf32>
      }
      %189 = reduce %185 dim=1 identities=[0xFF800000 : f32] : tile<128x128xf32> -> tile<128xf32>
      (%186: tile<f32>, %187: tile<f32>) {
        %188 = maxf %186, %187 : tile<f32>
        yield %188 : tile<f32>
      }
      %190 = reshape %189 : tile<128xf32> -> tile<128x1xf32>
      %191 = mulf %190, %154  : tile<128x1xf32>
      %192 = maxf %161, %191 : tile<128x1xf32>
      %193 = broadcast %192 : tile<128x1xf32> -> tile<128x128xf32>
      %194 = negf %193 : tile<128x128xf32>
      %195 = fma %185, %156, %194  : tile<128x128xf32>
      %196 = exp2 %195 flush_to_zero : tile<128x128xf32>
      %200 = reduce %196 dim=1 identities=[0.000000e+00 : f32] : tile<128x128xf32> -> tile<128xf32>
      (%197: tile<f32>, %198: tile<f32>) {
        %199 = addf %197, %198  : tile<f32>
        yield %199 : tile<f32>
      }
      %201 = reshape %200 : tile<128xf32> -> tile<128x1xf32>
      %202 = subf %161, %192  : tile<128x1xf32>
      %203 = exp2 %202 flush_to_zero : tile<128x1xf32>
      %204 = fma %160, %203, %201  : tile<128x1xf32>
      %205 = broadcast %203 : tile<128x1xf32> -> tile<128x64xf32>
      %206 = mulf %159, %205  : tile<128x64xf32>
      %207 = make_partition_view %86 : partition_view<tile=(1x1x128x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>
      %208, %209 = load_view_tko weak %207[%106, %116, %158, %157] token = %45 optimization_hints = <sm_90 = {latency = 4}> : partition_view<tile=(1x1x128x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x128x64xf16>, token
      %210 = reshape %208 : tile<1x1x128x64xf16> -> tile<128x64xf16>
      %211 = ftof %196  : tile<128x128xf32> -> tile<128x128xf16>
      %212 = mmaf %211, %210, %206 : tile<128x128xf16>, tile<128x64xf16>, tile<128x64xf32>
      continue %212, %204, %192 : tile<128x64xf32>, tile<128x1xf32>, tile<128x1xf32>
    }
    %216 = broadcast %214 : tile<128x1xf32> -> tile<128x64xf32>
    %217 = divf %213, %216 rounding<approx> flush_to_zero : tile<128x64xf32>
    %218 = reshape %217 : tile<128x64xf32> -> tile<1x1x128x64xf32>
    %219 = ftof %218  : tile<1x1x128x64xf32> -> tile<1x1x128x64xf16>
    %220 = constant <i32: 0> : tile<i32>
    %221 = make_partition_view %99 : partition_view<tile=(1x1x128x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>
    %222 = store_view_tko weak %219, %221[%106, %115, %100, %220] token = %45 : tile<1x1x128x64xf16>, partition_view<tile=(1x1x128x64), tensor_view<?x?x?x?xf16, strides=[?,?,?,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> token
    return
  }
}
