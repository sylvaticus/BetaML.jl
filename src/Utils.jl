
###############################
# Shared utilities functions ##
###############################

module Utils

using LinearAlgebra

export reshape, makeColVector, makeRowVector, makeMatrix, relu, drelu, linearf,
       dlinearf, dtanh, sigmoid, dsigmoid, squaredCost, dSquaredCost, l1_distance,
       l2_distance, l2²_distance, cosine_distance, normalFixedSd, lse, sterling

# ------------------------------------------------------------------------------
# Various reshaping functions

import Base.reshape
""" reshape(myNumber, dims..) - Reshape a number as a n dimensional Array """
reshape(x::T, dims...) where {T <: Number} =   (x = [x]; reshape(x,dims) )
makeColVector(x::T) where {T <: Number} =  [x]
makeColVector(x::T) where {T <: AbstractArray} =  reshape(x,length(x))
makeRowVector(x::T) where {T <: Number} = return [x]'
makeRowVector(x::T) where {T <: AbstractArray} =  reshape(x,1,length(x))
"""Transform an Array{T,1} in an Array{T,2} and leave unchanged Array{T,2}."""
makeMatrix(x::Array) = ndims(x) == 1 ? reshape(x, (size(x)...,1)) : x

# ------------------------------------------------------------------------------
# Various neural network activation functions as well their derivatives

relu(x)     = max(0,x)
drelu(x)    = x <= 0 ? 0 : 1
linearf(x)  = x
dlinearf(x) = 1
#tanh(x) already in Julia base
dtanh(x)    = 1-tanh(x)^2
sigmoid(x)  = 1/(1+exp(-x))
dsigmoid(x) = exp(-x)*sigmoid(x)^2

# ------------------------------------------------------------------------------
# Various neural network loss functions as well their derivatives
squaredCost(ŷ,y)   = (1/2)*norm(y - ŷ)^2
dSquaredCost(ŷ,y)  = .- (y .- ŷ)

# ------------------------------------------------------------------------------
# Some common distance measures

"""L1 norm distance (aka "Manhattan Distance")"""
l1_distance(x,y)     = sum(abs.(x-y))
"""Euclidean (L2) distance"""
l2_distance(x,y)     = norm(x-y)
"""Squared Euclidean (L2) distance"""
l2²_distance(x,y)    = norm(x-y)^2
"""Cosine distance"""
cosine_distance(x,y) = dot(x,y)/(norm(x)*norm(y))

# ------------------------------------------------------------------------------
# Some common PDFs

""" PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
normalFixedSd(x,μ,σ²)    = (1/(2π*σ²)^(length(x)/2)) * exp(-1/(2σ²)*norm(x-μ)^2)
""" log-PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
logNormalFixedSd(x,μ,σ²) = - (length(x)/2) * log(2π*σ²)  -  norm(x-μ)^2/(2σ²)


# ------------------------------------------------------------------------------
# Other mathematical/computational functions

""" LogSumExp for efficiently computing log(sum(exp.(x))) """
lse(x) = maximum(x)+log(sum(exp.(x .- maximum(x))))
""" Sterling number: number of partitions of a set of n elements in k sets """
sterling(n::BigInt,k::BigInt) = (1/factorial(k)) * sum((-1)^i * binomial(k,i)* (k-i)^n for i in 0:k)
sterling(n::Int64,k::Int64)   = sterling(BigInt(n),BigInt(k))


end # end module
