"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a layer without bias in the network

# Fields:
* `w`:  Weights matrix with respect to the input from previous layer or data (n x n pr. layer)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
struct DenseNoBiasLayer{TF <: Function, TDF <: Union{Nothing,Function}, WET <: Number, WT <: AbstractMatrix{WET}} <: AbstractLayer
     w::WT
     f::TF
     df::TDF
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new DenseNoBiasLayer

     # Positional arguments:
     * `nₗ`:  Number of nodes of the previous layer
     * `n`:   Number of nodes
     # Keyword arguments:
     * `w_eltype`: Eltype of the weights [def: `Float64`]
     * `w`:   Initial weights with respect to input [default: Xavier initialisation, dims = (n,nₗ)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [default: try to match with well-known derivatives, resort to AD if `f` is unknown]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]
     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n))`
     """
     function DenseNoBiasLayer(nₗ,n;rng = Random.GLOBAL_RNG,
           w_eltype = Float64,
           w  = xavier_init(nₗ,n,rng=rng,eltype=w_eltype),
           f=identity,df=match_known_derivatives(f))
      # To be sure w is a matrix and wb a column vector..
      w  = _dense(reshape(w,n,nₗ),w_eltype)
      return new{typeof(f),typeof(df),w_eltype,typeof(w)}(w,f,df)
     end
     # Constructor from the individual fields (used e.g. to move the layer to a GPU)
     function DenseNoBiasLayer(w::AbstractMatrix,f,df)
        return new{typeof(f),typeof(df),eltype(w),typeof(w)}(w,f,df)
     end
end

# The batch X has the records along the columns (nₗ × n records)

"Compute the derivative of the loss with respect to the layer's linear outputs (z = w * x)"
function _dϵ_dz(layer::DenseNoBiasLayer,X,next_gradient)
   df = _derivative(layer.f,layer.df)
   return df.(layer.w * X) .* next_gradient
end

function forward_batch(layer::DenseNoBiasLayer,X)
   return layer.f.(layer.w * _asmatrix(X))
end

function backward_batch(layer::DenseNoBiasLayer,X,next_gradient)
   dϵ_dz = _dϵ_dz(layer,_asmatrix(X),_asmatrix(next_gradient))
   return layer.w' * dϵ_dz
end

function get_gradient_batch(layer::DenseNoBiasLayer,X,next_gradient)
   X     = _asmatrix(X)
   dϵ_dz = _dϵ_dz(layer,X,_asmatrix(next_gradient))
   return Learnable((dϵ_dz * X',))
end

function backward_and_gradient_batch(layer::DenseNoBiasLayer,X,next_gradient)
   X     = _asmatrix(X)
   dϵ_dz = _dϵ_dz(layer,X,_asmatrix(next_gradient))
   return (layer.w' * dϵ_dz, Learnable((dϵ_dz * X',)))
end

function get_params(layer::DenseNoBiasLayer)
  return Learnable((layer.w,))
end

function set_params!(layer::DenseNoBiasLayer,w)
   layer.w .= w.data[1]
end

function size(layer::DenseNoBiasLayer)
   w_size =  size(layer.w')
   return ((w_size[1],),(w_size[2],))
end
