module {
  func.func @no_dependencies(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      %0 = affine.load %arg0[%i] : memref<64xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg1[%i] : memref<64xf32>
    }
    return
  }



func.func @raw_distance_one(%arg0: memref<64xf32>) {
    affine.for %i = 1 to 64 {
      %0 = affine.load %arg0[%i - 1] : memref<64xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%i] : memref<64xf32>
    }
    return
  }
  
}