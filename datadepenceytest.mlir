module {
  func.func @distance_three(%A: memref<20xf32>) {
    // This loop creates dependencies with distance 3
    
    affine.for %i = 4 to 12 {
      // RAW with distance 3: Write to A[i-3], read from A[i] 
      %val1 = affine.load %A[%i - 4] : memref<20xf32>    // READ A[i-3]
      
      // WAR with distance 3: Read from A[i], write to A[i+3]
      %val2 = affine.load %A[%i] : memref<20xf32>        // READ A[i]
      %sum = arith.addf %val1, %val2 : f32
      affine.store %sum, %A[%i + 4] : memref<20xf32>     // WRITE A[i+3]
      
      // WAW with distance 3: Multiple writes to locations 3 apart
      %const = arith.constant 2.0 : f32
      affine.store %const, %A[%i] : memref<20xf32>       // WRITE A[i]
      affine.store %const, %A[%i + 4] : memref<20xf32>   // WRITE A[i+3]
    }
    
    return
  }
}