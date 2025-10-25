#===- CMakeLists.txt - MLIR Transform Passes --------------------*- CMake -*-===#
# Gen pass declarations from TableGen definitions
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(NovaTransformPassIncGen)

# Generate pass documentation
add_mlir_doc(Passes TransformPasses Passes/ -gen-pass-doc)

add_mlir_dialect_library(MLIRTransformPasses
  # AffineFullUnroll implementation (local files)
  AffineFullUnroll.cpp
  
  # Affine Loop Transform implementations (from lib/Transforms/Affine/)
  ${PROJECT_SOURCE_DIR}/lib/Transforms/DependencyAnalysis.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/DependencyAnalysisTestPass.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/CleanupPass.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/MemoryAccessAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms

  DEPENDS
    NovaTransformPassIncGen

  LINK_LIBS PUBLIC
    # Core MLIR libraries
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRAffineAnalysis
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRPass
    MLIRTransforms
    
    # IR and support libraries
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
    
    # Analysis libraries
    MLIRAnalysis
)

# Configure include directories
target_include_directories(MLIRTransformPasses PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include/Compiler/Transforms
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# ===----------------------------------------------------------------------===//
# Install Headers
# ===----------------------------------------------------------------------===//

install(FILES
  # Main transform headers
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Passes.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/AffineFullUnroll.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/DependencyAnalysis.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/DependencyAnalysisTestPass.h
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/MemoryAccessAnalysis.h
  ${CMAKE_CURRENT_BINARY_DIR}/Passes.h.inc
  DESTINATION include/Compiler/Transforms
)


if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
# ===----------------------------------------------------------------------===//