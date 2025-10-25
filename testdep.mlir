module{
  
func.func @waw_distance_two(%arg0: memref<64xf32>) {
    affine.for %i = 2 to 64 {
      %cst1 = arith.constant 1.000000e+00 : f32
      affine.store %cst1, %arg0[%i - 2] : memref<64xf32>
      %cst2 = arith.constant 2.000000e+00 : f32  
      affine.store %cst2, %arg0[%i] : memref<64xf32>
    }
    return
  }
}