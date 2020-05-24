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

     Positional arguments:
     * `f`:  Activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     Keyword arguments:
     * `w`:  Initial weigths with respect to input [default: `rand(n,nₗ)`]
     * `wb`: Initial weigths with respect to bias [default: `rand(n)`]
     * `df`: Derivative of the activation function [default: `nothing` (i.e. use AD)]

     """
     function DenseLayer(f,nₗ,n;w=rand(n,nₗ),wb=rand(n),df=nothing)
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
  return (layer.w,layer.wb)
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
   return (dϵ_dw,dϵ_dwb)
end

function setParams!(layer::DenseLayer,w)
   layer.w = w[1]
   layer.wb = w[2]
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

     Positional arguments:
     * `f`:  Activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     Keyword arguments:
     * `w`:  Initial weigths with respect to input [default: `rand(n,nₗ)`]
     * `df`: Derivative of the activation function [default: `nothing` (i.e. use AD)]
     """
     function DenseNoBiasLayer(f,nₗ,n;w=rand(n,nₗ),df=nothing)
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
  return (layer.w,)
end

function getGradient(layer::DenseNoBiasLayer,x,nextGradient)
   z      = layer.w * x
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  = dϵ_dz * x'
   return (dϵ_dw,)
end

function setParams!(layer::DenseNoBiasLayer,w)
   layer.w = w[1]
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
whole output of the previous layer in input rather than working on a single node
as "normal" activation functions.
Useful for example for the SoftMax function.

# Fields:
* `nₗ`: Number of nodes of the previous layer
* `n`:  Number of nodes in output
* `f`:  Activation function (vector)
* `df`: Derivative of the (vector) activation function
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
     * `f`:  Activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes [default: `n`]
     # Keyword arguments:
     * `df`: Derivative of the activation function [default: `nothing` (i.e. use AD)]

     # Notes:
     - If the derivative is provided, it should return the gradient as a (nₗ,n) matrix (i.e. the Jacobian)

     """
     function VectorFunctionLayer(f,nₗ,n=nₗ;df=nothing)
         return new(nₗ,n,f,df)
     end
end

function forward(layer::VectorFunctionLayer,x)
  return layer.f(layer.x)
end

function backward(layer::VectorFunctionLayer,x,nextGradient)

   if layer.df != nothing
       dϵ_dI = layer.df(x) * nextGradient
    else  # using AD
      T = eltype(nextGradient)
      j = Array{T, 2}(undef, nₗ, n)
      for i in 1:n
          j[:, i] .= gradient(x -> layer.f(x)[i], x)[1]
      end
       dϵ_dI = j * nextGradient
    end
   return dϵ_dI
end

function getParams(layer::VectorFunctionLayer)
  return ()
end

function getGradient(layer::VectorFunctionLayer,x,nextGradient)
   return () # parameterless layer
end

function setParams!(layer::VectorFunctionLayer,w)
   return nothing
end
function size(layer::VectorFunctionLayer)
    return (nₗ,n)
end
