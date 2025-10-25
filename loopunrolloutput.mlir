module {
  func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %arg2[%arg3, %arg4] : memref<64x64xf32>
      }
    }
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xf32>
        }
      }
    }
    return
  }
  func.func @main() {
    %alloc = memref.alloc() : memref<64x64xf32>
    %alloc_0 = memref.alloc() : memref<64x64xf32>
    %alloc_1 = memref.alloc() : memref<64x64xf32>
    affine.for %arg0 = 0 to 64 {
      affine.for %arg1 = 0 to 64 {
        %cst = arith.constant 1.000000e+00 : f32
        affine.store %cst, %alloc[%arg0, %arg1] : memref<64x64xf32>
        affine.store %cst, %alloc_0[%arg0, %arg1] : memref<64x64xf32>
      }
    }
    call @matmul(%alloc, %alloc_0, %alloc_1) : (memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) -> ()
    memref.dealloc %alloc : memref<64x64xf32>
    memref.dealloc %alloc_0 : memref<64x64xf32>
    memref.dealloc %alloc_1 : memref<64x64xf32>
    return
  }
}

