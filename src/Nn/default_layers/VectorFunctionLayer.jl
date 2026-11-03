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
     * `dummyDataToTestOutputSize`: Dummy data to test the output size [def:
     `ones(nₗ)`]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
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
                dummyDataToTestOutputSize=ones(nₗ))
        nw = length(wsize) 
        if nw ==0 
          n  = length(f(dummyDataToTestOutputSize))
        else
         n  = length(f(dummyDataToTestOutputSize,w))
        end
        w = _dense(w,w_eltype)
        return new{nw,typeof(f),typeof(dfx),typeof(dfw),w_eltype,typeof(w)}(w,nₗ,n,f,dfx,dfw)
     end
     # Constructor from the individual fields (used e.g. to move the layer to a GPU)
     function VectorFunctionLayer(w::AbstractArray,nₗ::Integer,n::Integer,f,dfx,dfw)
        return new{ndims(w),typeof(f),typeof(dfx),typeof(dfw),eltype(w),typeof(w)}(w,nₗ,n,f,dfx,dfw)
     end
end

# The batch X has the records along the columns (nₗ × n records).
# The softmax function (the most common vector activation function) is vectorised over the
# batch, while other functions are applied record by record (on the CPU).

_isbatchsoftmax(layer::VectorFunctionLayer{N}) where {N} = N == 0 && layer.f === softmax

"Softmax of each column of X"
function _softmax_batch(X)
   E = exp.(X .- maximum(X,dims=1))
   return E ./ sum(E,dims=1)
end

function forward_batch(layer::VectorFunctionLayer{N},X) where {N}
   X = _asmatrix(X)
   _isbatchsoftmax(layer) && return _softmax_batch(X)
   w = _tocpu(layer.w)
   return _recordwise(x -> N == 0 ? layer.f(x) : layer.f(x,w),X)
end

function backward_batch(layer::VectorFunctionLayer,X,next_gradient)
   X = _asmatrix(X)
   next_gradient = _asmatrix(next_gradient)
   if _isbatchsoftmax(layer) && (layer.dfx === dsoftmax || isnothing(layer.dfx))
      # product of the softmax Jacobian, diag(s) - s*s', with the next gradient, for each record
      S = _softmax_batch(X)
      return S .* (next_gradient .- sum(S .* next_gradient,dims=1))
   end
   w = _tocpu(layer.w)
   return _recordwise((x,g) -> _backward_record(layer,w,x,g),X,next_gradient)
end
function _backward_record(layer::VectorFunctionLayer{N},w,x,next_gradient) where N
   if N == 0
      if layer.dfx != nothing
         dfz = layer.dfx(x)'
      else  # using AD
         dfz = (autojacobian(layer.f,x; nY=layer.n))'
      end
   else
      if layer.dfx != nothing
         dfz = layer.dfx(x,w)'
      else  # using AD
        tempfunction(x) = layer.f(x,w)
        dfz = (autojacobian(tempfunction,x; nY=layer.n))'
      end
   end
   return dfz * next_gradient
end

function get_params(layer::VectorFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function get_gradient_batch(layer::VectorFunctionLayer{0},X,next_gradient)
   return Learnable(()) # parameterless layer
end
function get_gradient_batch(layer::VectorFunctionLayer,X,next_gradient)
   w = _tocpu(layer.w)
   return _recordwise_sum((x,g) -> _get_gradient_record(layer,w,x,g),_asmatrix(X),_asmatrix(next_gradient),layer.w)
end
function _get_gradient_record(layer::VectorFunctionLayer,w,x,next_gradient)
   if layer.dfw != nothing
      dfz = layer.dfw(x,w)'
   else  # using AD
      dfz = (autojacobian(wt -> layer.f(x,wt),w; nY=layer.n))'
   end
   dϵ_dw = dfz * next_gradient
   return Learnable((dϵ_dw,))
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
