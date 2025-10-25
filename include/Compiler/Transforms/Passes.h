#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "AffineFullUnroll.h"
#include "DependencyAnalysisTestPass.h"

namespace mlir {
namespace nova {

#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif