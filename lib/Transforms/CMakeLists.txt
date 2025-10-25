# Add CleanupPass to the list of sources
add_mlir_library(CompilerTransforms
  CleanupPass.cpp
  FuseMatmulInit.cpp
  DependencyAnalysisTestPass.cpp
  DependencyAnalysis.cpp
  MemoryAccessAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRTransformUtils
  MLIRNova
  MLIRArithDialect
  MLIRMathDialect 
  MLIRFuncDialect  
  MLIRAffineDialect
  MLIRMemRefDialect     
)
#===----------------------------------------------------------------------===//