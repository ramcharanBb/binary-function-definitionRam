#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Diagnostics.h" 
using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"


//----------------------add op functions ---------------------
// 1 . inferReturnTypes
// 2 . verify


LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

  //create a function specific for this operation and call it here

  if(operands[0].getType() == operands[1].getType()){
    inferredReturnTypes.push_back(operands[0].getType());
    return success();
  }
  return failure();
}

LogicalResult AddOp::verify() {

  /*constraints on input : 
  1 -> 2 operands 
  2 -> They need to be same or compatible type 
  3 -> compatible means they can be made same by broadcasting
  4 -> should supported data types : int,float,


  constraint on output:
  1-> result and operands type should be same
  2 -> if broadcaseted result type needs to be same as operands type after broadcasting*/



  // example : Checking that the operand and result types are all the same.
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  if (lhsType != rhsType || lhsType != resultType) {
    return emitOpError("requires all operand and result types to be the same");
  }
  return success();
}

//----------------------sub op functions ---------------------
// 1 . inferReturnTypes


LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

  //create a function specific for this operation and call it here

  if(operands[0].getType() == operands[1].getType()){
    inferredReturnTypes.push_back(operands[0].getType());
    return success();
  }
  return failure();
}
  /*constraints on input : 
  1 -> 2 operands 
  2 -> They need to be same or compatible type 
  3 -> compatible means they can be made same by broadcasting
  4 -> should supported data types : int,float,

  constraint on output:
  1-> result and operands type should be same
  2 -> if broadcaseted result type needs to be same as operands type after broadcasting*/
