"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a VectorFunction layer in the network. Vector
function layer expects a vector activation function, i.e. a function taking the
whole output of the previous layer as input rather than working on a single node
as "normal" activation functions would do.
Useful for example with the SoftMax function in classification or with the
`pool1D` function to implement a "pool" layer in 1 dimension.
By default it is weightless, i.e. it doesn't apply any transformation to the
output coming from the previous layer except the activation function. However,
by passing the parameter `wsize` (a tuple or array - tested only 1D) you can
pass the learnable parameter to the activation function too. It is your
responsibility to be sure the activation function accepts only X or also this 
learnable array (as second argument).   
The number of nodes in input must be set to the same as in the previous layer
(and if you are using this for classification, to the number of classes, i.e.
the _previous_ layer must be set equal to the number of classes in the
predictions).

# Fields:
* `w`:   Weights (parameter) array passed as second argument to the activation
         function (if not empty)
* `nₗ`:  Number of nodes in input (i.e. length of previous layer)
* `n`:   Number of nodes in output (automatically inferred in the constructor)
* `f`:   Activation function (vector)
* `dfx`: Derivative of the (vector) activation function with respect to the
         layer inputs (x)
* `dfw`: Derivative of the (vector) activation function with respect to the
         optional learnable weights (w)         

# Notes:
* The output `size` of this layer is given by the size of the output function,
which is not necessarily the same as that of the previous layer.
"""
struct VectorFunctionLayer{N, TF <: Function, TDFX <: Union{Nothing,Function}, TDFW <: Union{Nothing,Function}, WET <: Number, WT <: AbstractArray{WET,N}} <: AbstractLayer
     w::WT
     nₗ::Int64
     n::Int64
     f::TF
     dfx::TDFX
     dfw::TDFW
     batched::Bool
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new VectorFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be the same as in the previous layer)
     # Keyword arguments:
     * `wsize`: A tuple or array specifying the size (number of elements) of the
       learnable parameter [def: empty array]
     * `w_eltype`: Eltype of the weights [def: `Float64`]
     * `w`:   Initial weights with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:  Activation function [def: `softmax`]
     * `dfx`: Derivative of the activation function with respect to the data
     [default: try to match with well-known derivatives, resort to AD if `f` is unknown]
     * `dfw`: Derivative of the activation function with respect to the
              learnable parameter [default: `nothing` (i.e. use AD)]
     * `batched`: Whether `f` can be applied directly to a whole batch of records, i.e. to a (nₗ × n records) matrix, returning the (n × n records) matrix of the outputs of each record (e.g. `f = X -> X ./ sum(X,dims=1)`) [def: `false`, i.e. `f` is applied record by record]
     * `dummyDataToTestOutputSize`: Dummy data to test the output size [def:
     `ones(nₗ)`]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian) and it is computed record by record. Otherwise the derivatives are computed by AD (with Zygote), on the whole batch if `batched=true`
     - With `batched=true` (and no derivative provided) the layer is computed with array-level operations on the whole batch, so it is much faster and it can run on GPUs (if `f` is written with array-level operations, e.g. broadcasting). Otherwise the computation is performed record by record on the CPU (except for `softmax`, that is always computed on the whole batch)
     - To avoid recomputing the activation function just to determine its output size,
       we compute the output size once here in the layer constructor by calling the
       activation function with `dummyDataToTestOutputSize`. Feel free to change
       it if it doesn't match with the activation function you are setting
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function VectorFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],
                w_eltype = Float64,
                w  = xavier_init(sum(wsize)/2,sum(wsize)/2,rng=rng,Tuple(wsize),eltype=w_eltype),
                f=softmax,
                dfx=match_known_derivatives(f),
                dfw=nothing,
                batched=false,
                dummyDataToTestOutputSize=ones(nₗ))
        nw = length(wsize) 
        x  = batched ? reshape(dummyDataToTestOutputSize,:,1) : dummyDataToTestOutputSize
        n  = size(nw == 0 ? f(x) : f(x,w),1)
        w = _dense(w,w_eltype)
        return new{nw,typeof(f),typeof(dfx),typeof(dfw),w_eltype,typeof(w)}(w,nₗ,n,f,dfx,dfw,batched)
     end
     # Constructor from the individual fields (used e.g. to move the layer to a GPU)
     function VectorFunctionLayer(w::AbstractArray,nₗ::Integer,n::Integer,f,dfx,dfw,batched::Bool)
        return new{ndims(w),typeof(f),typeof(dfx),typeof(dfw),eltype(w),typeof(w)}(w,nₗ,n,f,dfx,dfw,batched)
     end
end

# The batch X has the records along the columns (nₗ × n records).
# The softmax function (the most common vector activation function) and the functions declared
# as `batched` are computed on the whole batch, while the other functions are applied record by
# record (on the CPU). When not provided, the derivatives are computed by AD as the pullback of
# the function (i.e. the product of the transposed Jacobian with the next gradient).

_isbatchsoftmax(layer::VectorFunctionLayer{N}) where {N} = N == 0 && layer.f === softmax

"Softmax of each column of X"
function _softmax_batch(X)
   E = exp.(X .- maximum(X,dims=1))
   return E ./ sum(E,dims=1)
end

"Return the arguments of the activation function: `(x,)` or `(x,w)`"
_fargs(layer::VectorFunctionLayer{N},x,w) where {N} = N == 0 ? (x,) : (x,w)

"Pullback of the activation function with respect to all its arguments (records and weights)"
function _pullback(layer::VectorFunctionLayer,args,next_gradient)
   _, back = Zygote.pullback(layer.f,args...)
   grads   = back(next_gradient)
   return map((d,a) -> isnothing(d) ? zero(a) : d, grads, args) # AD returns `nothing` for args not used
end

function forward_batch(layer::VectorFunctionLayer,X)
   X = _asmatrix(X)
   _isbatchsoftmax(layer) && return _softmax_batch(X)
   layer.batched && return layer.f(_fargs(layer,X,layer.w)...)
   w = _tocpu(layer.w)
   return _recordwise(x -> layer.f(_fargs(layer,x,w)...),X)
end

function backward_batch(layer::VectorFunctionLayer,X,next_gradient)
   X = _asmatrix(X)
   next_gradient = _asmatrix(next_gradient)
   if _isbatchsoftmax(layer) && (layer.dfx === dsoftmax || isnothing(layer.dfx))
      # product of the softmax Jacobian, diag(s) - s*s', with the next gradient, for each record
      S = _softmax_batch(X)
      return S .* (next_gradient .- sum(S .* next_gradient,dims=1))
   end
   (layer.batched && isnothing(layer.dfx)) && return _pullback(layer,_fargs(layer,X,layer.w),next_gradient)[1]
   w = _tocpu(layer.w)
   return _recordwise((x,g) -> _backward_record(layer,w,x,g),X,next_gradient)
end
function _backward_record(layer::VectorFunctionLayer{N},w,x,next_gradient) where N
   isnothing(layer.dfx) && return _pullback(layer,_fargs(layer,x,w),next_gradient)[1]
   dfz = N == 0 ? layer.dfx(x)' : layer.dfx(x,w)'
   return dfz * next_gradient
end

function get_params(layer::VectorFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function get_gradient_batch(layer::VectorFunctionLayer{0},X,next_gradient)
   return Learnable(()) # parameterless layer
end
function get_gradient_batch(layer::VectorFunctionLayer,X,next_gradient)
   X = _asmatrix(X)
   next_gradient = _asmatrix(next_gradient)
   (layer.batched && isnothing(layer.dfw)) && return Learnable((_pullback(layer,(X,layer.w),next_gradient)[2],))
   w = _tocpu(layer.w)
   return _recordwise_sum((x,g) -> _get_gradient_record(layer,w,x,g),X,next_gradient,layer.w)
end
function _get_gradient_record(layer::VectorFunctionLayer,w,x,next_gradient)
   isnothing(layer.dfw) && return Learnable((_pullback(layer,(x,w),next_gradient)[2],))
   dfz = layer.dfw(x,w)'
   return Learnable((dfz * next_gradient,))
end

function backward_and_gradient_batch(layer::VectorFunctionLayer{N},X,next_gradient) where {N}
   if N > 0 && layer.batched && isnothing(layer.dfx) && isnothing(layer.dfw)
      # a single pullback gives both the derivatives with respect to the inputs and to the weights
      (dϵ_dX, dϵ_dw) = _pullback(layer,(_asmatrix(X),layer.w),_asmatrix(next_gradient))
      return (dϵ_dX,Learnable((dϵ_dw,)))
   end
   return (backward_batch(layer,X,next_gradient),get_gradient_batch(layer,X,next_gradient))
end

function set_params!(layer::VectorFunctionLayer{N},w) where {N}
   if N > 0
      layer.w .= w.data[1]
   end
   return nothing
end
function size(layer::VectorFunctionLayer{N}) where {N}
    # Output size for the VectorFunctionLayer is given by its activation function
    # We test its length with dummy values
    return ((layer.nₗ,),(layer.n,))
end
