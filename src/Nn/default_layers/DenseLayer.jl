"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
   DenseLayer

Representation of a layer in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct DenseLayer <: AbstractLayer
     w::Array{Float64,2}
     wb::Array{Float64,1}
     f::Function
     df::Union{Function,Nothing}
     """
        DenseLayer(f,n,nₗ;w,wb,df)

     Instantiate a new DenseLayer

     # Positional arguments:
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     # Keyword arguments:
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (n,nₗ)]
     * `wb`:  Initial weigths with respect to bias [default: Xavier initialisation, dims = (n)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n))`

     """
     function DenseLayer(nₗ,n;rng = Random.GLOBAL_RNG,w=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),wb=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n),f=identity,df=match_known_derivatives(f))
         size(w) == (n,nₗ) || error("If manually provided, w should have size (n,nₗ)")
         size(wb) == (n,) || error("If manually provided, wb should have size (n)")
         # To be sure w is a matrix and wb a column vector..
         w  = reshape(w,n,nₗ)
         wb = reshape(wb,n)
         return new(w,wb,f,df)
     end
end

function _zComp(layer::DenseLayer,x)
    w  = layer.w
    wb = layer.wb
    z  = zeros(eltype(x),size(w,1))
    @inbounds for n in axes(w,1)
        zn = zero(eltype(x))
        @turbo for nl in axes(x,1)
            zn += w[n,nl] * x[nl]
        end
        zn   += wb[n]
        z[n]  = zn
    end
    return z
end


function forward(layer::DenseLayer,x)
  z =  _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) #layer.w * x + layer.wb # _zComp(layer,x) #   layer.w * x + layer.wb # testd @avx
  return layer.f.(z)
end

function backward(layer::DenseLayer,x,next_gradient)
   z = _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) # layer.w * x + layer.wb # _zComp(layer,x) # @avx layer.w * x + layer.wb               # tested @avx
   if layer.df != nothing
      dfz = layer.df.(z)
    else
      dfz = layer.f'.(z) # using AD
    end
   dϵ_dz = @turbo dfz .* next_gradient
   dϵ_dI = @turbo layer.w' * dϵ_dz # @avx
   return dϵ_dI
end

function get_params(layer::DenseLayer)
  return Learnable((layer.w,layer.wb))
end

function get_gradient(layer::DenseLayer,x,next_gradient)
   z      =  _zComp(layer,x) #@avx layer.w * x + layer.wb #  _zComp(layer,x) #layer.w * x + layer.wb # @avx
   if layer.df != nothing
      dfz = layer.df.(z)  
   else
      dfz =  layer.f'.(z) # using AD
   end
   dϵ_dz  = @turbo  dfz .* next_gradient
   dϵ_dw  = @turbo dϵ_dz * x' # @avx
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
end

function set_params!(layer::DenseLayer,w)
   layer.w  = w.data[1]
   layer.wb = w.data[2]
end
function size(layer::DenseLayer)
   w_size =  size(layer.w')
   return ((w_size[1],),(w_size[2],))
end