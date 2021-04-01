"""
# Provide implementation of default layers

Provided layers
- DenseLayer
- DenseNoBiasLayer
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
mutable struct DenseLayer <: Layer
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
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
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

function forward(layer::DenseLayer,x)
  return layer.f.(layer.w * x + layer.wb)
end

function backward(layer::DenseLayer,x,nextGradient)
   z = layer.w * x + layer.wb
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dI = layer.w' * dϵ_dz
end

function getParams(layer::DenseLayer)
  return Learnable((layer.w,layer.wb))
end

function getGradient(layer::DenseLayer,x,nextGradient)
   z      = layer.w * x + layer.wb
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  = dϵ_dz * x'
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
end

function setParams!(layer::DenseLayer,w)
   layer.w = w.data[1]
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
mutable struct DenseNoBiasLayer <: Layer
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

function forward(layer::DenseNoBiasLayer,x)
  return layer.f.(layer.w * x)
end

function backward(layer::DenseNoBiasLayer,x,nextGradient)
   z = layer.w * x
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
   z      = layer.w * x
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  = dϵ_dz * x'
   return Learnable((dϵ_dw,))
end

function setParams!(layer::DenseNoBiasLayer,w)
   layer.w = w.data[1]
end

function size(layer::DenseNoBiasLayer)
    return size(layer.w')
end

# ------------------------------------------------------------------------------
# VectorFunction layer

"""
   VectorFunctionLayer

Representation of a (weightless) VectorFunction layer in the network. Vector
function layer expects a vector activation function, i.e. a function taking the
whole output of the previous layer an input rather than working on a single node
as "normal" activation functions would do.
Useful for example with the SoftMax function in classification or with the `pool1D`
function to implement a "pool" layer in 1 dimensions.
As it is weightless, it doesn't apply any transformation to the output coming
from the previous layer. It means that the number of nodes must be set to the same
as in the previous layer (and if you are using this for classification, to the
number of classes, i.e. the _previous_ layer must be set equal to the number of
classes in the predictions).

# Fields:
* `nₗ`: Number of nodes in input (i.e. length of previous layer)
* `n`:  Number of nodes in output (automatically inferred in the constructor)
* `f`:  Activation function (vector)
* `df`: Derivative of the (vector) activation function

# Notes:
* The output `size` of this layer is given by the size of the output function,
that not necessarily is the same as the previous layers.
"""
mutable struct VectorFunctionLayer <: Layer
     nₗ::Int64
     n::Int64
     f::Function
     df::Union{Function,Nothing}
     """
        VectorFunctionLayer(f,n;df)

     Instantiate a new VectorFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be same as in the previous layer)
     # Keyword arguments:
     * `f`:  Activation function [def: `softMax`]
     * `df`: Derivative of the activation function [default: `nothing` (i.e. use AD)]
     * `dummyDataToTestOutputSize`: Dummy data to test the output size [def: `ones(nₗ)`]

     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
     - To avoid recomputing the activation function just to determine its output size,
       we compute the output size once here in the layer constructor by calling the
       activation function with `dummyDataToTestOutputSize`. Feel free to change
       it if it doesn't match with the activation function you are setting
     """
     function VectorFunctionLayer(nₗ;f=softMax,df=nothing,dummyDataToTestOutputSize=ones(nₗ))
         n = length(f(dummyDataToTestOutputSize))
         return new(nₗ,n,f,df)
     end
end

function forward(layer::VectorFunctionLayer,x)
  return layer.f(x)
end

function backward(layer::VectorFunctionLayer,x,nextGradient)
   if layer.df != nothing
       dϵ_dI = layer.df(x)' * nextGradient
    else  # using AD
      j = autoJacobian(layer.f,x; nY=layer.n)
      dϵ_dI = j' * nextGradient
    end
   return dϵ_dI
end

function getParams(layer::VectorFunctionLayer)
  return Learnable(())
end

function getGradient(layer::VectorFunctionLayer,x,nextGradient)
   return Learnable(()) # parameterless layer
end

function setParams!(layer::VectorFunctionLayer,w)
   return nothing
end
function size(layer::VectorFunctionLayer)
    # Output size for the VectorFunctionLayer is given by its activation function
    # We test its length with dummy values
    return (layer.nₗ,layer.n)
end
