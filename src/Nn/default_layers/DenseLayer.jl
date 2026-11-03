"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a layer in the network

# Fields:
* `w`:  Weights matrix with respect to the input from previous layer or data (n x n pr. layer)
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
struct DenseLayer{TF <: Function, TDF <: Union{Nothing,Function}, WET <: Number, WT <: AbstractMatrix{WET}, BT <: AbstractVector{WET}} <: AbstractLayer
     w::WT
     wb::BT
     f::TF
     df::TDF
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new DenseLayer

     # Positional arguments:
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     # Keyword arguments:
     * `w_eltype`: Eltype of the weights [def: `Float64`]
     * `w`:   Initial weights with respect to input [default: Xavier initialisation, dims = (n,nₗ)]
     * `wb`:  Initial weights with respect to bias [default: Xavier initialisation, dims = (n)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [default: try to match with well-known derivatives, resort to AD if `f` is unknown]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]

     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n))`
     - Specify `df=nothing` to explicitly use AD

     """
     function DenseLayer(nₗ,n;rng = Random.GLOBAL_RNG,
         w_eltype = Float64,
         w  = xavier_init(nₗ,n,rng=rng,eltype=w_eltype),
         wb = xavier_init(nₗ,n,rng=rng,n,eltype=w_eltype),
         f=identity,df=match_known_derivatives(f))
        size(w) == (n,nₗ) || error("If manually provided, w should have size (n,nₗ)")
        size(wb) == (n,) || error("If manually provided, wb should have size (n)")
        # To be sure w is a matrix and wb a column vector..
        w  = _dense(reshape(w,n,nₗ),w_eltype)
        wb = _dense(reshape(wb,n),w_eltype)
        return new{typeof(f),typeof(df),w_eltype,typeof(w),typeof(wb)}(w,wb,f,df)
     end
     # Constructor from the individual fields (used e.g. to move the layer to a GPU)
     function DenseLayer(w::AbstractMatrix,wb::AbstractVector,f,df)
        return new{typeof(f),typeof(df),eltype(w),typeof(w),typeof(wb)}(w,wb,f,df)
     end
end

# The batch X has the records along the columns (nₗ × n records)

"Compute the derivative of the loss with respect to the layer's linear outputs (z = w * x + wb)"
function _dϵ_dz(layer::DenseLayer,X,next_gradient)
   df = _derivative(layer.f,layer.df)
   return df.(layer.w * X .+ layer.wb) .* next_gradient
end

function forward_batch(layer::DenseLayer,X)
   return layer.f.(layer.w * _asmatrix(X) .+ layer.wb)
end

function backward_batch(layer::DenseLayer,X,next_gradient)
   dϵ_dz = _dϵ_dz(layer,_asmatrix(X),_asmatrix(next_gradient))
   return layer.w' * dϵ_dz
end

function get_gradient_batch(layer::DenseLayer,X,next_gradient)
   X     = _asmatrix(X)
   dϵ_dz = _dϵ_dz(layer,X,_asmatrix(next_gradient))
   return Learnable((dϵ_dz * X',vec(sum(dϵ_dz,dims=2))))
end

function backward_and_gradient_batch(layer::DenseLayer,X,next_gradient)
   X     = _asmatrix(X)
   dϵ_dz = _dϵ_dz(layer,X,_asmatrix(next_gradient))
   return (layer.w' * dϵ_dz, Learnable((dϵ_dz * X',vec(sum(dϵ_dz,dims=2)))))
end

function get_params(layer::DenseLayer)
  return Learnable((layer.w,layer.wb))
end

function set_params!(layer::DenseLayer,w)
   layer.w  .= w.data[1]
   layer.wb .= w.data[2]
end
function size(layer::DenseLayer)
   w_size =  size(layer.w')
   return ((w_size[1],),(w_size[2],))
end
