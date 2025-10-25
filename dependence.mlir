// Define affine map for subtraction at top level
#map_sub = affine_map<(d0) -> (d0 - 2)>

func.func @complex_test(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>, %D: memref<32x32xf32>) {
  // Extra loop with cross-element dependencies
  affine.for %i = 2 to 31 {
    affine.for %j = 2 to 31 {
      %i_minus_1 = affine.apply #map_sub(%i)
      %j_minus_1 = affine.apply #map_sub(%j)
      %c1 = affine.load %C[%i, %j] : memref<32x32xf32>
      %c2 = affine.load %C[%i_minus_1, %j] : memref<32x32xf32>
      %c3 = affine.load %C[%i, %j_minus_1] : memref<32x32xf32>
      %sum1 = arith.addf %c1, %c2 : f32
      %sum2 = arith.addf %sum1, %c3 : f32
      affine.store %sum2, %D[%i, %j] : memref<32x32xf32>
    }
  }
  
  // Some diagonal write-after-read dependencies
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %valA = affine.load %A[%i, %i] : memref<32x32xf32>
      %prevD = affine.load %D[%i, %i] : memref<32x32xf32>
      %sumD = arith.addf %prevD, %valA : f32
      affine.store %sumD, %D[%i, %i] : memref<32x32xf32>
    }
  }
  
  return
}