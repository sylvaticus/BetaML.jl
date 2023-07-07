"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a layer without bias in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct DenseNoBiasLayer <: AbstractLayer
     w::Array{Float64,2}
     f::Function
     df::Union{Function,Nothing}
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new DenseNoBiasLayer

     # Positional arguments:
     * `nₗ`:  Number of nodes of the previous layer
     * `n`:   Number of nodes
     # Keyword arguments:
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [def: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ,n))`
     """
     function DenseNoBiasLayer(nₗ,n;rng = Random.GLOBAL_RNG,w=rand(rng,Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),f=identity,df=match_known_derivatives(f))
         # To be sure w is a matrix and wb a column vector..
         w  = reshape(w,n,nₗ)
         return new(w,f,df)
     end
end

function _zComp(layer::DenseNoBiasLayer,x)
    w  = layer.w
    z  = zeros(eltype(x),size(w,1))
    @inbounds for n in axes(w,1)
        zn = zero(eltype(x))
        @turbo for nl in axes(x,1)
            zn += w[n,nl] * x[nl]
        end
        z[n]  = zn
    end
    return z
end

function forward(layer::DenseNoBiasLayer,x)
  z = _zComp(layer,x)
  return layer.f.(z)
end

function backward(layer::DenseNoBiasLayer,x,next_gradient)
   z = _zComp(layer,x)
   if layer.df != nothing
      dfz = layer.df.(z) 
   else
      dfz = layer.f'.(z)  # using AD
   end
   dϵ_dz = @turbo dfz .* next_gradient
   dϵ_dI = @turbo layer.w' * dϵ_dz
   return dϵ_dI
end

function get_params(layer::DenseNoBiasLayer)
  return Learnable((layer.w,))
end

function get_gradient(layer::DenseNoBiasLayer,x,next_gradient)
   z      = _zComp(layer,x)
   if layer.df != nothing
      dfz = layer.df.(z)
   else
      dfz = layer.f'.(z) # using AD
   end
   dϵ_dz = @turbo dfz .* next_gradient
   dϵ_dw = @turbo dϵ_dz * x'
   return Learnable((dϵ_dw,))
end

function set_params!(layer::DenseNoBiasLayer,w)
   layer.w = w.data[1]
end

function size(layer::DenseNoBiasLayer)
   w_size =  size(layer.w')
   return ((w_size[1],),(w_size[2],))
end
