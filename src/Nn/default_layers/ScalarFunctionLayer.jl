"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a ScalarFunction layer in the network.
ScalarFunctionLayer applies the activation function directly to the output of
the previous layer (i.e., without passing through a weight matrix), but using an 
optional learnable parameter (an array) used as second argument, similarly to
[`VectorFunctionLayer`](@ref).
Differently from `VectorFunctionLayer`, the function is applied scalarwise to
each node. 
 
The number of nodes in input must be set to the same as in the previous layer


# Fields:
* `w`:   Weights (parameter) array passed as second argument to the activation
         function (if not empty)
* `n`:   Number of nodes in output (≡ number of nodes in input )
* `f`:   Activation function (vector)
* `dfx`: Derivative of the (vector) activation function with respect to the
         layer inputs (x)
* `dfw`: Derivative of the (vector) activation function with respect to the
         optional learnable weights (w)         

# Notes:
* The output `size` of this layer is the same as that of the previous layer.
"""
struct ScalarFunctionLayer{N, TF <: Function, TDFX <: Union{Nothing,Function}, TDFW <: Union{Nothing,Function}, WET <: Number, WT <: AbstractArray{WET,N}} <: AbstractLayer
     w::WT
     n::Int64
     f::TF
     dfx::TDFX
     dfw::TDFW
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new ScalarFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be the same as in the previous layer)
     # Keyword arguments:
     * `wsize`: A tuple or array specifying the size (number of elements) of the
       learnable parameter [def: empty array]
     * `w_eltype`: Eltype of the weights [def: `Float64`]
     * `w`:   Initial weights with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:  Activation function [def: `softmax`]
     * `dfx`: Derivative of the activation function with respect to the data  [default: try to match with well-known derivatives, resort to AD if `f` is unknown]
     * `dfw`: Derivative of the activation function with respect to the
              learnable parameter [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function ScalarFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],
              w_eltype = Float64,
              w  = xavier_init(sum(wsize)/2,sum(wsize)/2,rng=rng,Tuple(wsize),eltype=w_eltype),
              f=softmax,
              dfx=match_known_derivatives(f),
              dfw=nothing)
        nw = length(wsize) 
        w  = _dense(w,w_eltype)
        return new{nw,typeof(f),typeof(dfx),typeof(dfw),w_eltype,typeof(w)}(w,nₗ,f,dfx,dfw)
     end
     # Constructor from the individual fields (used e.g. to move the layer to a GPU)
     function ScalarFunctionLayer(w::AbstractArray,n::Integer,f,dfx,dfw)
        return new{ndims(w),typeof(f),typeof(dfx),typeof(dfw),eltype(w),typeof(w)}(w,n,f,dfx,dfw)
     end
end

"""
$(TYPEDSIGNATURES)

Create a weightless layer whose output is equal to the input. 
 
# Fields:
* `n`:   Number of nodes in output (≡ number of nodes in input )       

# Notes:
- The output `size` of this layer is the same as that of the previous layer.
- This is just an alias for a [`ScalarFunctionLayer`](@ref) with no weights and identity function.
"""
function ReplicatorLayer(n)
   return ScalarFunctionLayer(n,f=identity)
end

# The batch X has the records along the columns (n × n records).
# Without learnable weights the layer is fully vectorised, while with learnable weights the
# computation is performed record by record (on the CPU).

function forward_batch(layer::ScalarFunctionLayer{0},X)
   return layer.f.(X)
end
function forward_batch(layer::ScalarFunctionLayer,X)
   w = _tocpu(layer.w)
   return _recordwise(x -> layer.f.(x,w),X)
end

function backward_batch(layer::ScalarFunctionLayer{0},X,next_gradient)
   return _derivative(layer.f,layer.dfx).(X) .* next_gradient
end
function backward_batch(layer::ScalarFunctionLayer,X,next_gradient)
   w = _tocpu(layer.w)
   return _recordwise((x,g) -> _backward_record(layer,w,x,g),X,next_gradient)
end
function _backward_record(layer::ScalarFunctionLayer,w,x,next_gradient)
   if layer.dfx != nothing
      df_dx = layer.dfx.(x,Ref(w))
   else  # using AD
      df_dx = [gradient(xt -> layer.f(xt,w),xi)[1] for xi in x]
   end
   return df_dx .* next_gradient
end

function get_params(layer::ScalarFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function get_gradient_batch(layer::ScalarFunctionLayer{0},X,next_gradient)
   return Learnable(()) # parameterless layer
end
function get_gradient_batch(layer::ScalarFunctionLayer,X,next_gradient)
   w = _tocpu(layer.w)
   return _recordwise_sum((x,g) -> _get_gradient_record(layer,w,x,g),X,next_gradient,layer.w)
end
function _get_gradient_record(layer::ScalarFunctionLayer,w,x,next_gradient)
   if layer.dfw != nothing
      dfz = [layer.dfw(xi,wj) for wj in w, xi in x]
   else  # using AD
     tempfunction(w) = [layer.f(xi,w) for xi in x]
     dfz = (autojacobian(tempfunction,w; nY=layer.n))'
   end
   dϵ_dw = dfz * next_gradient
   return Learnable((dϵ_dw,))
end

function set_params!(layer::ScalarFunctionLayer{N},w) where {N}
   if N > 0
      layer.w .= w.data[1]
   end
   return nothing
end
function size(layer::ScalarFunctionLayer{N}) where {N}
    # Output size for the ScalarFunctionLayer is given by its activation function
    # We test its length with dummy values
    return ((layer.n,),(layer.n,))
end
