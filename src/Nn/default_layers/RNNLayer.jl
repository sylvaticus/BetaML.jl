"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Experimental


"""
   RNNLayer

Representation of a layer in the network

# Fields:
* `wx`: Weigths matrix with respect to the input from data (n by n_input)
* `ws`: Weigths matrix with respect to the layer state (n x n )
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
* `s` : State
"""
mutable struct RNNLayer <: RecursiveLayer
     wx::Array{Float64,2}
     ws::Array{Float64,2}
     wb::Array{Float64,1}
     s::Array{Float64,1}
     f::Function
     df::Union{Function,Nothing}
     """
        RNNLayer(nₗ,n;f,wx,ws,wb,df)

     Instantiate a new RNNLayer

     # Positional arguments:
     * `nₗ`: Number of nodes of the input
     * `n`:  Number of nodes of the state (and the output)
     # Keyword arguments:
     * `wx`:  Initial weigths with respect to input [default: Xavier initialisation, dims = (n,nₗ)]
     * `ws`:  Initial weigths with respect to input [default: Xavier initialisation, dims = (n,n)]
     * `wb`:  Initial weigths with respect to bias [default: Xavier initialisation, dims = (n)]
     * `s`:   Initial states [def: zeros(n)]
     * `f`:   Activation function [def: `relu`]
     * `df`:  Derivative of the activation function [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n))`

     """
     function RNNLayer(nₗ,n;rng = Random.GLOBAL_RNG,wx=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),ws=rand(rng, Uniform(-sqrt(6)/sqrt(n+n),sqrt(6)/sqrt(n+n)),n,n),wb=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n),s=zeros(n),f=relu,df=nothing)
         # To be sure w is a matrix and wb a column vector..
         wx  = reshape(w,n,nₗ)
         ws  = reshape(w,n,n)
         wb  = reshape(wb,n)
         s   = reshape(s,n)
         return new(w,wb,s,f,df)
     end
end
#=
function _zComp(layer::RNNLayer,x)
    w  = layer.w
    wb = layer.wb
    z  = zeros(eltype(x),size(w,1))
    @inbounds for n in axes(w,1)
        zn = zero(eltype(x))
        @simd for nl in axes(x,1)
            zn += w[n,nl] * x[nl]
        end
        zn   += wb[n]
        z[n]  = zn
    end
    return z
end
=#

function forward(layer::RNNLayer,x)
  #z =  _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) #layer.w * x + layer.wb # _zComp(layer,x) #   layer.w * x + layer.wb # testd @avx
  z =  layer.wb + layer.wx * x + layer.ws * s 
  return layer.f.(z)
end

function backward(layer::RNNLayer,x,next_gradient) #TODO
   z = _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) # layer.w * x + layer.wb # _zComp(layer,x) # @avx layer.w * x + layer.wb               # tested @avx
   if layer.df != nothing
      dfz = layer.df.(z) 
   else
      dfz = layer.f'.(z) # using AD
   end
   dϵ_dz = @turbo dfz .* next_gradient # tested @avx
   dϵ_dI =  layer.w' * dϵ_dz # @avx
   return dϵ_dI 
end

function get_params(layer::RNNLayer)
  return Learnable((layer.wb,layer.wx,layer.ws))
end

function get_gradient(layer::RNNLayer,x,next_gradient) #TODO
   z      =  _zComp(layer,x) #@avx layer.w * x + layer.wb #  _zComp(layer,x) #layer.w * x + layer.wb # @avx
   if layer.df != nothing
      dfz = layer.df.(z)
   else
      dfz = layer.f'.(z) # using AD
   end
   dϵ_dz  = @turbo dfz .* next_gradient 
   dϵ_dw  = @turbo dϵ_dz * x' # @avx
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
end

function set_params!(layer::RNNLayer,w)
   layer.wb = w.data[1]
   layer.wx = w.data[2]
   layer.ws = w.data[3]
end
function size(layer::RNNLayer)
    return size(layer.w')
end