"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a layer without bias in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
struct DenseNoBiasLayer{TF <: Function, TDF <: Union{Nothing,Function}, WET <: Number} <: AbstractLayer
     w::Array{WET,2}
     f::TF
     df::TDF
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new DenseNoBiasLayer

     # Positional arguments:
     * `nₗ`:  Number of nodes of the previous layer
     * `n`:   Number of nodes
     # Keyword arguments:
     * `w_eltype`: Eltype of the weigths [def: `Float64`]
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [default: try to match with well-known derivatives, resort to AD if `f` is unknown]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ,n))`
     """
     function DenseNoBiasLayer(nₗ,n;rng = Random.GLOBAL_RNG,
           w_eltype = Float64,
           w  = xavier_init(nₗ,n,rng=rng,eltype=w_eltype),
           f=identity,df=match_known_derivatives(f))
      # To be sure w is a matrix and wb a column vector..
      w  = reshape(w,n,nₗ)
      return new{typeof(f),typeof(df),w_eltype}(w,f,df)
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

function _zComp!(z,layer::DenseNoBiasLayer{TF,DTF,WET},x) where {TF, DTF, WET}
   @inbounds for n in axes(layer.w,1)
      zn = zero(WET)
      @turbo for nl in axes(x,1)
         zn += layer.w[n,nl] * x[nl]
      end
      z[n] += zn
   end
   return nothing
end

function forward(layer::DenseNoBiasLayer{TF,DTF,WET},x) where {TF, DTF, WET}
  z = zeros(WET,size(layer)[2][1])
  _zComp!(z,layer,x)
  return layer.f.(z)
end

function backward(layer::DenseNoBiasLayer{TF,DTF,WET},x,next_gradient) where {TF, DTF, WET}
   z = zeros(WET,size(layer)[2][1])
   _zComp!(z,layer,x)
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

function get_gradient(layer::DenseNoBiasLayer{TF,DTF,WET},x,next_gradient) where {TF, DTF, WET}
   z = zeros(WET,size(layer)[2][1])
   _zComp!(z,layer,x)
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
   layer.w .= w.data[1]
end

function size(layer::DenseNoBiasLayer)
   w_size =  size(layer.w')
   return ((w_size[1],),(w_size[2],))
end
