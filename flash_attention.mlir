cuda_tile.module @kernels {
  entry @flash_attention_forward_v2(%0: tile<ptr<f32>>, %1: tile<i32>, %2: tile<i32>, %3: tile<i32>, %4: tile<i32>, %5: tile<i32>, %6: tile<i32>, %7: tile<i32>, %8: tile<i32>, %9: tile<ptr<f32>>, %10: tile<i32>, %11: tile<i32>, %12: tile<i32>, %13: tile<i32>, %14: tile<i32>, %15: tile<i32>, %16: tile<i32>, %17: tile<i32>, %18: tile<ptr<f32>>, %19: tile<i32>, %20: tile<i32>, %21: tile<i32>, %22: tile<i32>, %23: tile<i32>, %24: tile<i32>, %25: tile<i32>, %26: tile<i32>, %27: tile<ptr<f32>>, %28: tile<i32>, %29: tile<i32>, %30: tile<i32>, %31: tile<i32>, %32: tile<i32>, %33: tile<i32>, %34: tile<i32>, %35: tile<i32>, %36: tile<i32>, %37: tile<i32>, %38: tile<i32>) optimization_hints=<sm_90 = {}> {
    %39 = assume div_by<4>, %0 : tile<ptr<f32>>
    %40 = assume bounded<0, ?>, %1 : tile<i32>
    %41 = assume bounded<0, ?>, %2 : tile<i32>
    %42 = assume bounded<0, ?>, %3 : tile<i32>
    %43 = assume bounded<0, ?>, %4 : tile<i32>
    %44 = assume div_by<1>, %40 : tile<i32>
    %45 = assume div_by<1>, %41 : tile<i32>
    %46 = assume div_by<1>, %42 : tile<i32>
    %47 = assume div_by<1>, %43 : tile<i32>
    %48 = assume bounded<0, ?>, %5 : tile<i32>
    %49 = assume bounded<0, ?>, %6 : tile<i32>
    %50 = assume bounded<0, ?>, %7 : tile<i32>
    %51 = assume bounded<0, ?>, %8 : tile<i32>
    %52 = assume div_by<1>, %48 : tile<i32>
    %53 = assume div_by<1>, %49 : tile<i32>
    %54 = assume div_by<1>, %50 : tile<i32>
    %55 = assume div_by<1>, %51 : tile<i32>
    %56 = make_tensor_view %39, shape = [%44, %45, %46, %47], strides = [1048576, 65536, 64, 1] : tile<i32> -> tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>
    %57 = assume div_by<4>, %9 : tile<ptr<f32>>
    %58 = assume bounded<0, ?>, %10 : tile<i32>
    %59 = assume bounded<0, ?>, %11 : tile<i32>
    %60 = assume bounded<0, ?>, %12 : tile<i32>
    %61 = assume bounded<0, ?>, %13 : tile<i32>
    %62 = assume div_by<1>, %58 : tile<i32>
    %63 = assume div_by<1>, %59 : tile<i32>
    %64 = assume div_by<1>, %60 : tile<i32>
    %65 = assume div_by<1>, %61 : tile<i32>
    %66 = assume bounded<0, ?>, %14 : tile<i32>
    %67 = assume bounded<0, ?>, %15 : tile<i32>
    %68 = assume bounded<0, ?>, %16 : tile<i32>
    %69 = assume bounded<0, ?>, %17 : tile<i32>
    %70 = assume div_by<1>, %66 : tile<i32>
    %71 = assume div_by<1>, %67 : tile<i32>
    %72 = assume div_by<1>, %68 : tile<i32>
    %73 = assume div_by<1>, %69 : tile<i32>
    %74 = make_tensor_view %57, shape = [%62, %63, %64, %65], strides = [1048576, 65536, 64, 1] : tile<i32> -> tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>
    %75 = assume div_by<4>, %18 : tile<ptr<f32>>
    %76 = assume bounded<0, ?>, %19 : tile<i32>
    %77 = assume bounded<0, ?>, %20 : tile<i32>
    %78 = assume bounded<0, ?>, %21 : tile<i32>
    %79 = assume bounded<0, ?>, %22 : tile<i32>
    %80 = assume div_by<1>, %76 : tile<i32>
    %81 = assume div_by<1>, %77 : tile<i32>
    %82 = assume div_by<1>, %78 : tile<i32>
    %83 = assume div_by<1>, %79 : tile<i32>
    %84 = assume bounded<0, ?>, %23 : tile<i32>
    %85 = assume bounded<0, ?>, %24 : tile<i32>
    %86 = assume bounded<0, ?>, %25 : tile<i32>
    %87 = assume bounded<0, ?>, %26 : tile<i32>
    %88 = assume div_by<1>, %84 : tile<i32>
    %89 = assume div_by<1>, %85 : tile<i32>
    %90 = assume div_by<1>, %86 : tile<i32>
    %91 = assume div_by<1>, %87 : tile<i32>
    %92 = make_tensor_view %75, shape = [%80, %81, %82, %83], strides = [1048576, 65536, 64, 1] : tile<i32> -> tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>
    %93 = assume div_by<4>, %27 : tile<ptr<f32>>
    %94 = assume bounded<0, ?>, %28 : tile<i32>
    %95 = assume bounded<0, ?>, %29 : tile<i32>
    %96 = assume bounded<0, ?>, %30 : tile<i32>
    %97 = assume bounded<0, ?>, %31 : tile<i32>
    %98 = assume div_by<1>, %94 : tile<i32>
    %99 = assume div_by<1>, %95 : tile<i32>
    %100 = assume div_by<1>, %96 : tile<i32>
    %101 = assume div_by<1>, %97 : tile<i32>
    %102 = assume bounded<0, ?>, %32 : tile<i32>
    %103 = assume bounded<0, ?>, %33 : tile<i32>
    %104 = assume bounded<0, ?>, %34 : tile<i32>
    %105 = assume bounded<0, ?>, %35 : tile<i32>
    %106 = assume div_by<1>, %102 : tile<i32>
    %107 = assume div_by<1>, %103 : tile<i32>
    %108 = assume div_by<1>, %104 : tile<i32>
    %109 = assume div_by<1>, %105 : tile<i32>
    %110 = make_tensor_view %93, shape = [%98, %99, %100, %101], strides = [1048576, 65536, 64, 1] : tile<i32> -> tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>
    %111 = make_token : token
    %112, %113, %114 = get_tile_block_id : tile<i32>
    %115, %116, %117 = get_tile_block_id : tile<i32>
    %118, %119, %120 = get_tile_block_id : tile<i32>
    %121 = constant <i32: 32> : tile<i32>
    %122 = divi %62, %121 signed rounding<negative_inf> : tile<i32>
    %123 = constant <i32: 0> : tile<i32>
    %124 = make_partition_view %56 : partition_view<tile=(1x1x32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>
    %125, %126 = load_view_tko weak %124[%112, %116, %120, %123] token = %111 : partition_view<tile=(1x1x32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x32x64xf32>, token
    %127 = reshape %125 : tile<1x1x32x64xf32> -> tile<32x64xf32>
    %128 = constant <f32: 0.000000e+00> : tile<32x64xf32>
    %129 = constant <f32: 0.000000e+00> : tile<32x1xf32>
    %130 = constant <f32: -1.000000e+10> : tile<32x1xf32>
    %131 = constant <i32: 0> : tile<i32>
    %132 = constant <i32: 1> : tile<i32>
    %133 = constant <i32: 0> : tile<i32>
    %134 = constant <i32: 0> : tile<i32>
    %135 = constant <f32: 0.000000e+00> : tile<32x32xf32>
    %136 = constant <f32: 8.000000e+00> : tile<f32>
    %137 = reshape %136 : tile<f32> -> tile<1x1xf32>
    %138 = broadcast %137 : tile<1x1xf32> -> tile<32x32xf32>
    %179, %180, %181 = for %139 in (%131 to %122, step %132) : tile<i32> iter_values(%140 = %129, %141 = %130, %142 = %128) -> (tile<32x1xf32>, tile<32x1xf32>, tile<32x64xf32>) {
      %143 = make_partition_view %74 : partition_view<tile=(1x1x32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>
      %144, %145 = load_view_tko weak %143[%112, %116, %139, %133] token = %111 : partition_view<tile=(1x1x32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x32x64xf32>, token
      %146 = make_partition_view %92 : partition_view<tile=(1x1x32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>
      %147, %148 = load_view_tko weak %146[%112, %116, %139, %134] token = %111 : partition_view<tile=(1x1x32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> tile<1x1x32x64xf32>, token
      %149 = reshape %144 : tile<1x1x32x64xf32> -> tile<32x64xf32>
      %150 = reshape %147 : tile<1x1x32x64xf32> -> tile<32x64xf32>
      %151 = permute %149 [1, 0] : tile<32x64xf32> -> tile<64x32xf32>
      %152 = mmaf %127, %151, %135 : tile<32x64xf32>, tile<64x32xf32>, tile<32x32xf32>
      %153 = divf %152, %138  : tile<32x32xf32>
      %157 = reduce %153 dim=1 identities=[0xFF800000 : f32] : tile<32x32xf32> -> tile<32xf32>
      (%154: tile<f32>, %155: tile<f32>) {
        %156 = maxf %154, %155 : tile<f32>
        yield %156 : tile<f32>
      }
      %158 = reshape %157 : tile<32xf32> -> tile<32x1xf32>
      %159 = cat %141, %158 dim = 1 : tile<32x1xf32>, tile<32x1xf32> -> tile<32x2xf32>
      %163 = reduce %159 dim=1 identities=[0xFF800000 : f32] : tile<32x2xf32> -> tile<32xf32>
      (%160: tile<f32>, %161: tile<f32>) {
        %162 = maxf %160, %161 : tile<f32>
        yield %162 : tile<f32>
      }
      %164 = reshape %163 : tile<32xf32> -> tile<32x1xf32>
      %165 = broadcast %164 : tile<32x1xf32> -> tile<32x32xf32>
      %166 = subf %153, %165  : tile<32x32xf32>
      %167 = exp %166 : tile<32x32xf32>
      %171 = reduce %167 dim=1 identities=[-0.000000e+00 : f32] : tile<32x32xf32> -> tile<32xf32>
      (%168: tile<f32>, %169: tile<f32>) {
        %170 = addf %168, %169  : tile<f32>
        yield %170 : tile<f32>
      }
      %172 = reshape %171 : tile<32xf32> -> tile<32x1xf32>
      %173 = subf %141, %164  : tile<32x1xf32>
      %174 = exp %173 : tile<32x1xf32>
      %175 = fma %140, %174, %172  : tile<32x1xf32>
      %176 = broadcast %174 : tile<32x1xf32> -> tile<32x64xf32>
      %177 = mulf %142, %176  : tile<32x64xf32>
      %178 = mmaf %167, %150, %177 : tile<32x32xf32>, tile<32x64xf32>, tile<32x64xf32>
      continue %175, %164, %178 : tile<32x1xf32>, tile<32x1xf32>, tile<32x64xf32>
    }
    %182 = broadcast %179 : tile<32x1xf32> -> tile<32x64xf32>
    %183 = divf %181, %182  : tile<32x64xf32>
    %184 = constant <i32: 0> : tile<i32>
    %185 = make_partition_view %110 : partition_view<tile=(32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>
    %186 = store_view_tko weak %183, %185[%112, %116, %120, %184] token = %111 : tile<32x64xf32>, partition_view<tile=(32x64), tensor_view<?x?x?x?xf32, strides=[1048576,65536,64,1]>, dim_map=[0, 1, 2, 3]>, tile<i32> -> token
    return
  }
}
