#ifndef NOVA_OPS
#define NOVA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "NovaDialect.td"

def Nova_ElementType : AnyTypeOf<[I1, I8, I16, I32, I64, F16, BF16, F32, F64, Complex<F32>, Complex<F64>]>;
def Nova_Tensor : TensorOf<[Nova_ElementType]>;

//------------------base class for binary--------------------------
// Base class for all binary elementwise tensor ops.
class Nova_BinaryOp<string mnemonic, list<Trait> commonTraits = []>: 
Op<Nova_Dialect, mnemonic,
         commonTraits # [InferTypeOpInterface,
                                   NoMemoryEffect,Elementwise ,
                                   SameOperandsAndResultElementType]> {

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results   = (outs Nova_Tensor:$result);

  let assemblyFormat =
    "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";

  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}



//----------------------add---------------------------------------

def Nova_AddOp : Nova_BinaryOp<"add", [Commutative]> {
  let summary = "Elementwise tensor addition";
  let description = [{
    Adds two tensors elementwise. Result type is automatically inferred.
    
    Example:
      %r = nova.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];
}
//-------------------------sub------------------------------------

def Nova_SubOp : Nova_BinaryOp<"sub",[] > {
  let summary = "Elementwise tensor subtraction";
  let description = [{
    Subtracts two tensors elementwise.

    Example:
      %r = nova.sub %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];
}

//-------------------------multiply------------------------------------

def Nova_MulOp : Nova_BinaryOp<"mul",[Commutative] > {
  let summary = "Elementwise tensor multipliucation";
  let description = [{
    multiplies two tensors elementwise.
    eg: [1,2,3] x [3,4,5] = [3,8,15]

    Example:
      %r = nova.mul %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];
}
//------------------------divide------------------------

def Nova_DivOp :Nova_BinaryOp<"div",[]>{
  let summary ="Elementwise tensor division";
  let description =[{ divides two tensor(dividend and divisor) elementwise.
    Example:
      %r = nova.div %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

}
//------------------------remainder------------------------

def Nova_RemOp :Nova_BinaryOp<"rem",[]>{
  let summary ="Elementwise tensor remainder finding";
  let description =[{Performs element-wise remainder of dividend lhs and divisor rhs. (dividend and divisor) elementwise.
  computation example :
  a=[2,7,10] b=[2,2,4] result= [0,1,2]
  
    Example:
      %r = nova.rem %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

}
//------------------------power------------------------

def Nova_PowOp :Nova_BinaryOp<"pow",[]>{
  let summary ="Elementwise tensor remainder finding";
  let description =[{Performs element-wise remainder of dividend lhs and divisor rhs. 
  computation example :
  a=[1,2,3] b=[2,2,4] result= [1,4,81]
  
  MLIR Example:
      %r = nova.pow %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

}
//------------------------max------------------------

def Nova_MaxOp :Nova_BinaryOp<"max",[Commutative]>{
  let summary ="Elementwise tensor remainder finding";
  let description =[{Performs element-wise max operation on tensors . 
  like 
  a=[1,2,3] b=[2,4,2] result= [2,4,3]
  
  Example:
      %r = nova.max %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

}
//------------------------min------------------------

def Nova_MinOp :Nova_BinaryOp<"min",[Commutative]>{
  let summary ="Elementwise tensor remainder finding";
  let description =[{Performs element-wise min operation on tensors 
  like 
  a=[1,2,3] b=[2,4,2] result= [1,2,2]
  
  Example:
      %r = nova.min %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

}
//-----------------------------------------------------
#endif // NOVA_OPS
