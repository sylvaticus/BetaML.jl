"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a ScalarFunction layer in the network.
ScalarFunctionLayer applies the activation function directly to the output of
the previous layer (i.e., without passing through a weight matrix), node by node,
optionally using a learnable weight as second argument: `yᵢ = f(xᵢ)` or `yᵢ = f(xᵢ,wᵢ)`.
The learnable weights can be either one per node, or a single one shared by all nodes.
Differently from [`VectorFunctionLayer`](@ref), the function is applied scalarwise to
each node. 
 
The number of nodes in input must be set to the same as in the previous layer


# Fields:
* `w`:   Weights (parameter) vector passed as second argument to the activation
         function (if not empty), either with one element per node or a single element
* `n`:   Number of nodes in output (≡ number of nodes in input )
* `f`:   Activation function (scalar)
* `dfx`: Derivative of the activation function with respect to the
         layer input of the node (x)
* `dfw`: Derivative of the activation function with respect to the
         learnable weight of the node (w)         

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
     * `wsize`: The size of the learnable weights, either `[nₗ]` (one weight per node) or `[1]` (a single weight shared by all nodes) [def: empty array, i.e. no learnable weights]
     * `w_eltype`: Eltype of the weights [def: `Float64`]
     * `w`:   Initial weights [default: Xavier initialisation, dims = `wsize`]
     * `f`:  Activation function. Without learnable weights it is called as `f(xᵢ)`, with learnable weights as `f(xᵢ,wᵢ)` [def: `softmax`]
     * `dfx`: Derivative of the activation function with respect to the data: `dfx(xᵢ)` or `dfx(xᵢ,wᵢ)` [default: try to match with well-known derivatives, resort to AD if `f` is unknown]
     * `dfw`: Derivative of the activation function with respect to the learnable weight: `dfw(xᵢ,wᵢ)` [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]
     # Notes:
     - The derivatives, if provided, are scalar functions (of the scalar input and, if any, of the scalar weight of the node)
     - With a single shared weight, `wᵢ` is the same for all nodes
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function ScalarFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],
              w_eltype = Float64,
              w  = xavier_init(sum(wsize)/2,sum(wsize)/2,rng=rng,Tuple(wsize),eltype=w_eltype),
              f=softmax,
              dfx=match_known_derivatives(f),
              dfw=nothing)
        nw = length(wsize) 
        if nw > 0
           (nw == 1 && wsize[1] in (1,nₗ)) || error("`wsize` must be either `[$(nₗ)]` (one weight per node) or `[1]` (a single weight shared by all nodes)")
           # with weights, a known derivative of `f(x)` is not a derivative of `f(x,w)`
           dfx == match_known_derivatives(f) && (dfx = nothing)
        end
        w  = _dense(w,w_eltype)
        size(w) == Tuple(wsize) || error("If manually provided, w should have size $(Tuple(wsize))")
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

# The batch X has the records along the columns (n × n records). Everything is computed node
# by node, so the layer is fully vectorised (the weight vector, of n or 1 elements, broadcasts
# over the batch).

"Derivatives of `f(x,w)` with respect to x and w (`dfx`/`dfw` if known, automatic otherwise)"
_derivative_x(f,dfx) = dfx
_derivative_x(f,::Nothing) = (x,w) -> ForwardDiff.derivative(t -> f(t,w),x)
_derivative_w(f,dfw) = dfw
_derivative_w(f,::Nothing) = (x,w) -> ForwardDiff.derivative(t -> f(x,t),w)

function forward_batch(layer::ScalarFunctionLayer{0},X)
   return layer.f.(X)
end
function forward_batch(layer::ScalarFunctionLayer,X)
   return layer.f.(_asmatrix(X),layer.w)
end

function backward_batch(layer::ScalarFunctionLayer{0},X,next_gradient)
   return _derivative(layer.f,layer.dfx).(X) .* next_gradient
end
function backward_batch(layer::ScalarFunctionLayer,X,next_gradient)
   dfx = _derivative_x(layer.f,layer.dfx)
   return dfx.(_asmatrix(X),layer.w) .* _asmatrix(next_gradient)
end

function get_params(layer::ScalarFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function get_gradient_batch(layer::ScalarFunctionLayer{0},X,next_gradient)
   return Learnable(()) # parameterless layer
end
function get_gradient_batch(layer::ScalarFunctionLayer,X,next_gradient)
   dfw   = _derivative_w(layer.f,layer.dfw)
   dϵ_dw = dfw.(_asmatrix(X),layer.w) .* _asmatrix(next_gradient)
   # sum over the records (and over the nodes for a shared weight)
   dϵ_dw = sum(dϵ_dw,dims = length(layer.w) == 1 ? (1,2) : 2)
   return Learnable((reshape(dϵ_dw,size(layer.w)),))
end

function set_params!(layer::ScalarFunctionLayer{N},w) where {N}
   if N > 0
      layer.w .= w.data[1]
   end
   return nothing
end
function size(layer::ScalarFunctionLayer{N}) where {N}
    return ((layer.n,),(layer.n,))
end
