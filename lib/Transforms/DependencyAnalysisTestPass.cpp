#include "Compiler/Transforms/DependencyAnalysisTestPass.h"
#include "Compiler/Transforms/DependencyAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // Add this include
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace nova {

void analyzeLoop(affine::AffineForOp forOp) {
  DependencyAnalysis depAnalysis(forOp);
  llvm::outs() << "\n=== Dependency Analysis for loop at ";
  forOp->getLoc().print(llvm::outs());
  llvm::outs() << " ===\n";
  llvm::outs().flush();
  if (!depAnalysis.hasDependencies()) {
    llvm::outs() << "  No loop-carried dependencies found.\n";
  } else {
    llvm::outs() << "  Found " << depAnalysis.getDependencies().size()
                 << " dependencies:\n";
    for (const auto &dep : depAnalysis.getDependencies()) {
      llvm::outs() << "    - Type: ";
      if (dep.isRAW)
        llvm::outs() << "RAW (Read-After-Write)";
      else if (dep.isWAR)
        llvm::outs() << "WAR (Write-After-Read)";
      else if (dep.isWAW)
        llvm::outs() << "WAW (Write-After-Write)";
      else
        llvm::outs() << "Unknown";
      llvm::outs() << "\n      Distance: ";
      if (dep.distance == UINT_MAX)
        llvm::outs() << "Unknown";
      else
        llvm::outs() << dep.distance;
      llvm::outs() << "\n      Memref: ";
      
      // FIXED: Use the correct MLIR API
      if (isa<BlockArgument>(dep.memref)) {
        // For function arguments, try to get a meaningful name
        auto blockArg = cast<BlockArgument>(dep.memref);
        llvm::outs() << "arg" << blockArg.getArgNumber();
      } else if (auto allocOp = dep.memref.getDefiningOp<memref::AllocOp>()) {
        llvm::outs() << "alloc";
      } else {
        // For any other case, just print the value
        dep.memref.print(llvm::outs());
      }
      llvm::outs() << " : " << dep.memref.getType();
      llvm::outs() << "\n";
    }
  }
  llvm::outs() << "\n  Summary:\n";
  unsigned minDist = depAnalysis.getMinDependencyDistance();
  llvm::outs() << "    Minimum dependency distance: ";
  if (minDist == UINT_MAX)
    llvm::outs() << "None (no dependencies)\n";
  else
    llvm::outs() << minDist << "\n";
  unsigned chainLen = depAnalysis.getDependencyChainLength();
  llvm::outs() << "    Dependency chain length: " << chainLen << "\n";
  bool tightDeps1 = depAnalysis.hasTightDependencies(2);
  llvm::outs() << "    Has tight dependencies (≤ 2): "
               << (tightDeps1 ? "Yes" : "No") << "\n";
  bool tightDeps = depAnalysis.hasTightDependencies(4);
  llvm::outs() << "    Has tight dependencies (≤ 4): "
               << (tightDeps ? "Yes" : "No") << "\n";
  llvm::outs() << "========================================\n\n";
  llvm::outs().flush(); 

  // Recursively analyze nested loops
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (auto nestedForOp = dyn_cast<affine::AffineForOp>(nestedOp)) {
      analyzeLoop(nestedForOp);
    }
  }
}

void DependencyAnalysisTestPass::runOnOperation() {
  func::FuncOp func = getOperation();
  for (Block &block : func.getBody()) {
    for (Operation &op : block) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        analyzeLoop(forOp);
      }
    }
  }
}

// Register the pass
void registerDependencyAnalysisTestPass() {
  mlir::PassRegistration<DependencyAnalysisTestPass>();
}

} // namespace nova
} // namespace mlir