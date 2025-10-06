#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"



//--------------------------------------------------------------
// Helper inferreturn function for binary operations
//---------------------------------------------------------------

/*constraints on input : 
  1 -> 2 operands 
  2 -> They need to be same or compatible type 
  3 -> compatible means they can be made same by broadcasting
  4 -> should supported data types : int,float,

  constraint on output:
  1-> result and operands type should be same
  2 -> if broadcaseted result type needs to be same as operands type after broadcasting*/

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

//-------------------------------------------------    
//addOp   
//------------------------------------------------- 

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
    
//-------------------------------------------------    
//SubOp  
//------------------------------------------------- 
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

//-------------------------------------------------    
//MulOp     
//------------------------------------------------- 
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

//-------------------------------------------------    
//DivideOp     
//------------------------------------------------- 

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

//-------------------------------------------------    
//remainderOp     
//------------------------------------------------- 
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

//-------------------------------------------------    
//powerOp     
//------------------------------------------------- 

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

//-------------------------------------------------    
//maximumOp     
//-------------------------------------------------    
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

//-------------------------------------------------    
//minimumOp
//-------------------------------------------------
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
