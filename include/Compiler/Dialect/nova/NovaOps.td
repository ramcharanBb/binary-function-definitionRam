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
  let hasVerifier =1;
}
//-------------------------sub------------------------------------

def Nova_AddOp : Nova_BinaryOp<"sub",[HLO_CompatibleOperandsAndResultType] > {
  let summary = "Elementwise tensor subtraction";
  let description = [{
    Subtracts two tensors elementwise. Result type is automatically inferred.

    Example:
      %r = nova.sub %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];
}



//-----------------------------------------------------
#endif // NOVA_OPS
