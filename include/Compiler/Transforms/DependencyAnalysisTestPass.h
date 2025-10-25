#ifndef MLIR_TRANSFORMS_DEPENDENCYANALYSISTESTPASS_H
#define MLIR_TRANSFORMS_DEPENDENCYANALYSISTESTPASS_H

#include "Compiler/Transforms/DependencyAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

class DependencyAnalysisTestPass: public PassWrapper<DependencyAnalysisTestPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final {
    return "test-dependency-analysis";
  }
  
  StringRef getDescription() const final {
    return "Test pass for loop dependency analysis";
  }
  
  void runOnOperation() override;
};

void registerDependencyAnalysisTestPass();

} // namespace nova
} // namespace mlir

#endif // MLIR_TRANSFORMS_DEPENDENCYANALYSISTESTPASS_H