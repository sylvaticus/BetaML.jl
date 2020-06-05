"""
  Utils.jl File

Machine Learning shared utility functions (Module Bmlt.Utils)

`?Bmlt.Utils` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/Bmlt.jl/blob/master/src/Utils.jl) - [Julia Package](https://github.com/sylvaticus/Utils.jl)
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    Utils module

Provide shared utility functions for various machine learning algorithms. You don't usually need to import from this module, as each other module (Nn, Perceptron, Clusters,...) reexport it.

"""
module Utils

using LinearAlgebra, Random, Statistics, Zygote

export reshape, makeColVector, makeRowVector, makeMatrix,
       oneHotEncoder, getScaleFactors, scale, scale!,batch,
       relu, drelu, didentity, dtanh, sigmoid, dsigmoid, softMax, dSoftMax,
       autoJacobian,
       squaredCost, dSquaredCost, l1_distance,
       error, accuracy,
       l2_distance, l2²_distance, cosine_distance, normalFixedSd, lse, sterling, logNormalFixedSd,
       radialKernel,polynomialKernel,
       Verbosity, NONE, LOW, STD, HIGH, FULL

#export @reexport



@enum Verbosity NONE=0 LOW=10 STD=20 HIGH=30 FULL=40

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
makeMatrix(x::AbstractArray) = ndims(x) == 1 ? reshape(x, (size(x)...,1)) : x


function oneHotEncoderRow(y,d;count = false)
    y = makeColVector(y)
    out = zeros(Int64,d)
    for j in y
        out[j] = count ? out[j] + 1 : 1
    end
    return out
end
"""
    oneHotEncoder(y,d;count)

Encode arrays (or arrays of arrays) of integer data as 0/1 matrices

# Parameters:
- `y`: The data to convert (integer, array or array of arrays of integers)
- `d`: The number of dimensions in the output matrik. [def: `maximum(maximum.(Y))`]
- `count`: Wether to count multiple instances on the same dimension/record or indicate just presence. [def: `false`]

"""
function oneHotEncoder(Y,d=maximum(maximum.(Y));count=false)
    n   = length(Y)
    if d < maximum(maximum.(Y))
        error("Trying to encode elements with indexes greater than the provided number of dimensions. Please increase d.")
    end
    out = zeros(Int64,n,d)
    for (i,y) in enumerate(Y)
        out[i,:] = oneHotEncoderRow(y,d;count = count)
    end
    return out
end

"""
  batch(n,bSize;sequential=false)

Return a vector of `bSize` indeces from `1` to `n`.
Randomly unless the optional parameter `sequential` is used.
"""
function batch(n::Integer,bSize::Integer;sequential=false)
    ridx = sequential ? collect(1:n) : shuffle(1:n)
    if bSize > n
        return [ridx]
    end
    nBatches = Int64(floor(n/bSize))
    batches = Array{Int64,1}[]
    for b in 1:nBatches
        push!(batches,ridx[b*bSize-bSize+1:b*bSize])
    end
    return batches
end

"""
    getScaleFactors(x;skip)

Return the scale factors (for each dimensions) in order to scale a matrix X (n,d)
such that each dimension has mean 0 and variance 1.

# Parameters
- `x`: the (n × d) dimension matrix to scale on each dimension d
- `skip`: an array of dimension index to skip the scaling [def: `[]`]

# Return
- A touple whose first elmement is the shift and the second the multiplicative
term to make the scale.
"""
function getScaleFactors(x;skip=[])
    μ  = mean(x,dims=1)
    σ² = var(x,corrected=false,dims=1)
    sfμ = - μ
    sfσ² = 1 ./ sqrt.(σ²)
    for i in skip
        sfμ[i] = 0
        sfσ²[i] = 1
    end
    return (sfμ,sfσ²)
end

"""
    scale(x;scalingFactors)

Perform a linear scaling of x using scaling factors `scalingFactors`.

# Parameters
- `x`: the (n × d) dimension matrix to scale on each dimension d
- `scalingFactors`: an tuple of the constant and multiplicative scaling factor
respectively [def: the scaling factors needed to scale x to mean 0 and variance 1]

# Return
- The scaled matrix

# Notes:
- also available `scale!(x;scalingFactors)` for in-place scaling.
"""
function scale(x,scaleFactors=(-mean(x,dims=1),1 ./ sqrt.(var(x,corrected=false,dims=1)) ))
    y = (x .+ scaleFactors[1]) .* scaleFactors[2]
    return y
end
function scale!(x,scaleFactors=(-mean(x,dims=1),1 ./ sqrt.(var(x,corrected=false,dims=1))))
    x .= (x .+ scaleFactors[1]) .* scaleFactors[2]
    return nothing
end

# ------------------------------------------------------------------------------
# Various neural network activation functions as well their derivatives

relu(x)     = max(0,x)
drelu(x)    = x <= 0 ? 0 : 1
#identity(x)  = x already in Julia base
didentity(x) = 1
#tanh(x) already in Julia base
dtanh(x)    = 1-tanh(x)^2
sigmoid(x)  = 1/(1+exp(-x))
dsigmoid(x) = exp(-x)*sigmoid(x)^2
softMax(x;β=1) = exp.((β .* x) .- lse(β .* x)) # efficient implementation of softMax(x)  = exp.(x) ./  sum(exp.(x))
""" dSoftMax(x;β) - Derivative of the softMax function """
function dSoftMax(x;β=1) # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    x = makeColVector(x)
    d = length(x)
    out = zeros(d,d)
    y = softMax(x,β=β)
    for i in 1:d
        smi = y[i]
        for j in 1:d
            if j == i
                out[i,j] = β*(smi-smi^2)
            else
                out[i,j] = - β*y[j]*smi
            end
        end
    end
    return out
end

"""
   autoJacobian(f,x;nY)

Evaluate the Jacobian using AD in the form of a (nY,nX) madrix of first derivatives

# Parameters:
- `f`: The function to compute the Jacobian
- `x`: The input to the function where the jacobian has to be computed
- `nY`: The number of outputs of the function `f` [def: `length(f(x))`]

# Return values:
- An `Array{Float64,2}` of the locally evaluated Jacobian

# Notes:
- The `nY` parameter is optional. If provided it avoids having to compute `f(x)`
"""
function autoJacobian(f,x;nY=length(f(x)))
    x = convert(Array{Float64,1},x)
    #j = Array{Float64, 2}(undef, size(x,1), nY)
    #for i in 1:nY
    #    j[:, i] .= gradient(x -> f(x)[i], x)[1]
    #end
    #return j'
    j = Array{Float64, 2}(undef, nY, size(x,1))
    for i in 1:nY
        j[i,:] = gradient(x -> f(x)[i], x)[1]'
    end
    return j
end


# ------------------------------------------------------------------------------
# Various error/accuracy measures
import Base.error
""" Categorical error """
error(x::Array{Int64,1},y::Array{Int64,1}) = sum(x .!= y)/length(x)
""" Categorical accuracy """
accuracy(x::Array{Int64,1},y::Array{Int64,1}) = sum(x .== y)/length(x)
"""
    accuracy(x,y;tol)
Categorical accuracy with probabilistic prediction of a single datapoint.

Use the parameter tol [def: `1`] to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.
"""
function accuracy(x::Array{T,1},y::Int64;tol=1) where {T <: Number}
    sIdx = sortperm(x)[end:-1:1]
    if x[y] in x[sIdx[1:min(tol,length(sIdx))]]
        return 1
    else
        return 0
    end
end
"""
   accuracy(x,y;tol)

Categorical accuracy with probabilistic predictions of a dataset.

Use the parameter tol [def: `1`] to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.
"""
function accuracy(x::Array{T,2},y::Array{Int64,1};tol=1) where {T <: Number}
    n = size(x,1)
    acc = sum([accuracy(x[i,:],y[i],tol=tol) for i in 1:n])/n
end
""" Categorical error with probabilistic prediction of a single datapoint. """
error(x::Array{T,1},y::Int64;tol=1) where {T <: Number} = 1 - accuracy(x,y;tol=tol)
""" Categorical error with probabilistic predictions of a dataset. """
error(x::Array{T,2},y::Array{Int64,1};tol=1) where {T <: Number} = 1 - accuracy(x,y;tol=tol)


# ------------------------------------------------------------------------------
# Various neural network loss functions as well their derivatives
squaredCost(ŷ,y)   = (1/2)*norm(y - ŷ)^2
dSquaredCost(ŷ,y)  = ( ŷ - y)

# ------------------------------------------------------------------------------
# Various kernel functions (e.g. for Perceptron)
"""Radial Kernel (aka _RBF kernel_) parametrised with γ=1/2. For other gammas γᵢ use
`K = (x,y) -> radialKernel(x,y,γ=γᵢ)` as kernel function in the supporting algorithms"""
radialKernel(x,y;γ=1/2) = exp(-γ*norm(x-y)^2)
"""Polynomial kernel parametrised with `c=0` and `d=2` (i.e. a quadratic kernel).
For other `cᵢ` and `dᵢ` use `K = (x,y) -> polynomialKernel(x,y,c=cᵢ,d=dᵢ)` as
kernel function in the supporting algorithms"""
polynomialKernel(x,y;c=0,d=2) = (dot(x,y)+c)^d

# ------------------------------------------------------------------------------
# Some common distance measures

"""L1 norm distance (aka _Manhattan Distance_)"""
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

#=
# https://github.com/simonster/Reexport.jl/blob/master/src/Reexport.jl
macro reexport(ex)
    isa(ex, Expr) && (ex.head == :module ||
                      ex.head == :using ||
                      (ex.head == :toplevel &&
                       all(e->isa(e, Expr) && e.head == :using, ex.args))) ||
        error("@reexport: syntax error")

    if ex.head == :module
        modules = Any[ex.args[2]]
        ex = Expr(:toplevel, ex, :(using .$(ex.args[2])))
    elseif ex.head == :using && all(e->isa(e, Symbol), ex.args)
        modules = Any[ex.args[end]]
    elseif ex.head == :using && ex.args[1].head == :(:)
        symbols = [e.args[end] for e in ex.args[1].args[2:end]]
        return esc(Expr(:toplevel, ex, :(eval(Expr(:export, $symbols...)))))
    else
        modules = Any[e.args[end] for e in ex.args]
    end

    esc(Expr(:toplevel, ex,
             [:(eval(Expr(:export, names($mod)...))) for mod in modules]...))
end
=#

#=

"""
    gradientDescentSingleUpdate(θ,▽,η)

Perform a single update of parameter θ using the gradient descent method with gradient ▽ and learning rate η.

# Notes:
- The parameter and the gradient can be either numbers, arrays or tuple of arrays.
"""
#- For Arrays and tuple of floats it is available also as inplace modification as
#  `gradientDescentSingleUpdate!(θ,▽,η)`
gradientDescentSingleUpdate(θ::Number,▽::Number,η) = θ .- (η .* ▽)
gradientDescentSingleUpdate(θ::AbstractArray,▽::AbstractArray,η) = gradientDescentSingleUpdate.(θ,▽,Ref(η))
gradientDescentSingleUpdate(θ::Tuple,▽::Tuple,η) = gradientDescentSingleUpdate.(θ,▽,Ref(η))
#gradientDescentSingleUpdate!(θ::AbstractArray{AbstractFloat},▽::AbstractArray{AbstractFloat},η) = (θ .= θ .- (η .* ▽))
#gradientDescentSingleUpdate!(θ::AbstractArray{AbstractFloat},▽::AbstractArray{Number},η) = (θ .= θ .- (η .* ▽))
#gradientDescentSingleUpdate!(θ::Tuple,▽::Tuple,η) = gradientDescentSingleUpdate!.(θ,▽,Ref(η))
=#

end # end module
