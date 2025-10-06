#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"

static LogicalResult BinaryInferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

   //create a function specific for this operation and call it here
  // Check we have exactly 2 operands
  if (operands.empty() || operands.size() != 2) {
      return mlir::emitOptionalError(*loc,"Expected non-empty operands for the operation and exactly 2 operands");}
  // Get the types of the operands
  auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());
  // Verify both operands are tensors
  if (!lhsType || !rhsType) {
    if (loc){
      mlir::emitError(*loc, "nova.add operands must be tensors");
      return failure();
    }
    return failure();
  }
  // Check if the operand types are the same
  if(operands[0].getType() == operands[1].getType()){
    inferredReturnTypes.push_back(operands[0].getType());
    return success();
  }

  // we assume result type is same as lhs type
  inferredReturnTypes.push_back(operands[0].getType());
  
  return success();
    }
//----------------add---------------------------
LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//----------------sub---------------------------
LogicalResult SubOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//----------------mul---------------------------
LogicalResult MulOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//----------------div---------------------------
LogicalResult DivOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//----------------rem---------------------------
LogicalResult RemOp::inferReturnTypes(     
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//----------------pow---------------------------
LogicalResult PowOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//----------------maximum---------------------------        
LogicalResult MaxOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//----------------minimum---------------------------
LogicalResult MinOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
