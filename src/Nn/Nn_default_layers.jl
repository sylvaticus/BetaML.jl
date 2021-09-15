"""
# Provide implementation of default layers

Provided layers
- DenseLayer
- DenseNoBiasLayer
- VectorFunctionLayer
"""

#using Random, Zygote
#import ..Utils
#import Base.size

# ------------------------------------------------------------------------------
# DenseLayer layer

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
     function DenseLayer(nₗ,n;rng = Random.GLOBAL_RNG,w=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),wb=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n),f=identity,df=nothing)
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
        @simd for nl in axes(x,1)
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

function backward(layer::DenseLayer,x,nextGradient)
   z = _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) # layer.w * x + layer.wb # _zComp(layer,x) # @avx layer.w * x + layer.wb               # tested @avx
   if layer.df != nothing
       dϵ_dz =  layer.df.(z) .* nextGradient # tested @avx
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dI =  layer.w' * dϵ_dz # @avx
end

function getParams(layer::DenseLayer)
  return Learnable((layer.w,layer.wb))
end

function getGradient(layer::DenseLayer,x,nextGradient)
   z      =  _zComp(layer,x) #@avx layer.w * x + layer.wb #  _zComp(layer,x) #layer.w * x + layer.wb # @avx
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  = dϵ_dz * x' # @avx
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
end

function setParams!(layer::DenseLayer,w)
   layer.w  = w.data[1]
   layer.wb = w.data[2]
end
function size(layer::DenseLayer)
    return size(layer.w')
end


# ------------------------------------------------------------------------------
# DenseNoBiasLayer layer

"""
   DenseNoBiasLayer

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
     """
        DenseNoBiasLayer(f,nₗ,n;w,df)

     Instantiate a new DenseNoBiasLayer

        DenseNoBiasLayer(f,nₗ,n;w,df)

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
     function DenseNoBiasLayer(nₗ,n;rng = Random.GLOBAL_RNG,w=rand(rng,Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),f=identity,df=nothing)
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
        @simd for nl in axes(x,1)
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

function backward(layer::DenseNoBiasLayer,x,nextGradient)
   z = _zComp(layer,x)
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dI = layer.w' * dϵ_dz
end

function getParams(layer::DenseNoBiasLayer)
  return Learnable((layer.w,))
end

function getGradient(layer::DenseNoBiasLayer,x,nextGradient)
   z      = _zComp(layer,x)
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  =  dϵ_dz * x'
   return Learnable((dϵ_dw,))
end

function setParams!(layer::DenseNoBiasLayer,w)
   layer.w = w.data[1]
end

function size(layer::DenseNoBiasLayer)
    return size(layer.w')
end

# ------------------------------------------------------------------------------
# VectorFunctionLayer layer

"""
   VectorFunctionLayer

Representation of a VectorFunction layer in the network. Vector
function layer expects a vector activation function, i.e. a function taking the
whole output of the previous layer an input rather than working on a single node
as "normal" activation functions would do.
Useful for example with the SoftMax function in classification or with the
`pool1D` function to implement a "pool" layer in 1 dimensions.
By default it is weightless, i.e. it doesn't apply any transformation to the
output coming from the previous layer except the activation function. However,
by passing the parameter `wsize` (a touple or array - tested only 1D) you can
pass the learnable parameter to the activation function too. It is your
responsability to be sure the activation function accept only X or also this 
learnable array (as second argument).   
The number of nodes in input must be set to the same as in the previous layer
(and if you are using this for classification, to the number of classes, i.e.
the _previous_ layer must be set equal to the number of classes in the
predictions).

# Fields:
* `w`:   Weigths (parameter) array passes as second argument to the activation
         function (if not empty)
* `nₗ`:  Number of nodes in input (i.e. length of previous layer)
* `n`:   Number of nodes in output (automatically inferred in the constructor)
* `f`:   Activation function (vector)
* `dfx`: Derivative of the (vector) activation function with respect to the
         layer inputs (x)
* `dfw`: Derivative of the (vector) activation function with respect to the
         optional learnable weigths (w)         

# Notes:
* The output `size` of this layer is given by the size of the output function,
that not necessarily is the same as the previous layers.
"""
mutable struct VectorFunctionLayer{N} <: AbstractLayer
     w::Array{Float64,N}
     nₗ::Int64
     n::Int64
     f::Function
     dfx::Union{Function,Nothing}
     dfw::Union{Function,Nothing}
     """
        VectorFunctionLayer(n;rng,wsize,w,f,dfx,dfw,dummyDataToTestOutputSize)

     Instantiate a new VectorFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be same as in the previous layer)
     # Keyword arguments:
     * `wsize`: A tuple or array specifying the size (number of elements) of the
       learnable parameter [def: empty array]
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:  Activation function [def: `softMax`]
     * `dfx`: Derivative of the activation function with respect to the data
              [default: `nothing` (i.e. use AD)]
     * `dfw`: Derivative of the activation function with respect to the
              learnable parameter [default: `nothing` (i.e. use AD)]
     * `dummyDataToTestOutputSize`: Dummy data to test the output size [def:
     `ones(nₗ)`]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
     - To avoid recomputing the activation function just to determine its output size,
       we compute the output size once here in the layer constructor by calling the
       activation function with `dummyDataToTestOutputSize`. Feel free to change
       it if it doesn't match with the activation function you are setting
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function VectorFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],w=rand(rng,Uniform(-sqrt(6)/sqrt(sum(wsize)),sqrt(6)/sqrt(sum(wsize))),Tuple(wsize)), f=softMax,dfx=nothing,dfw=nothing,dummyDataToTestOutputSize=ones(nₗ))
        nw = length(wsize) 
        if nw ==0 
          n  = length(f(dummyDataToTestOutputSize))
        else
         n  = length(f(dummyDataToTestOutputSize,w))
        end
        return new{nw}(w,nₗ,n,f,dfx,dfw)
     end
end

function forward(layer::VectorFunctionLayer{N},x) where {N}
  return N == 0 ? layer.f(x) : layer.f(x,layer.w)
end

function backward(layer::VectorFunctionLayer{N},x,nextGradient) where N
   if N == 0
      if layer.dfx != nothing
         dϵ_dI = layer.dfx(x)' * nextGradient
      else  # using AD
         j = autoJacobian(layer.f,x; nY=layer.n)
         dϵ_dI = j' * nextGradient
      end
      return dϵ_dI
   else
      if layer.dfx != nothing
         dϵ_dI = layer.dfx(x,layer.w)' * nextGradient
      else  # using AD
        tempfunction(x) = layer.f(x,layer.w)
        nYl = layer.n
        j = autoJacobian(tempfunction,x; nY=nYl)
        dϵ_dI = j' * nextGradient
      end
     return dϵ_dI
   end
end

function getParams(layer::VectorFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function getGradient(layer::VectorFunctionLayer{N},x,nextGradient) where {N}
   if N == 0
     return Learnable(()) # parameterless layer
   else
      if layer.dfw != nothing
         dϵ_dw = layer.dfw(x,layer.w)' * nextGradient
      else  # using AD
        j = autoJacobian(wt -> layer.f(x,wt),layer.w; nY=layer.n)
        dϵ_dw = j' * nextGradient
      end
     return Learnable((dϵ_dw,))
   end
end

function setParams!(layer::VectorFunctionLayer{N},w) where {N}
   if N > 0
      layer.w = w.data[1]
   end
   return nothing
end
function size(layer::VectorFunctionLayer{N}) where {N}
    # Output size for the VectorFunctionLayer is given by its activation function
    # We test its length with dummy values
    return (layer.nₗ,layer.n)
end
