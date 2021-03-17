"""
  Utils.jl File

Machine Learning shared utility functions (Module BetaML.Utils)

`?BetaML.Utils` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Utils.jl) - [Julia Package](https://github.com/sylvaticus/Utils.jl)
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    Utils module

Provide shared utility functions for various machine learning algorithms. You don't usually need to import from this module, as each other module (Nn, Perceptron, Clusters,...) reexport it.

"""
module Utils

using LinearAlgebra, Random, Statistics, Future, Combinatorics, Zygote, CategoricalArrays

using ForceImport
@force using ..Api

export Verbosity, NONE, LOW, STD, HIGH, FULL,
       FIXEDSEED, FIXEDRNG, @codeLocation, generateParallelRngs,
       reshape, makeColVector, makeRowVector, makeMatrix, issortable, getPermutations,
       oneHotEncoder, integerEncoder, integerDecoder, colsWithMissing, getScaleFactors, scale, scale!, batch, partition, pca,
       didentity, relu, drelu, elu, delu, celu, dcelu, plu, dplu,  #identity and rectify units
       dtanh, sigmoid, dsigmoid, softmax, dsoftmax, softplus, dsoftplus, mish, dmish, # exp/trig based functions
       bic, aic,
       autoJacobian,
       squaredCost, dSquaredCost, crossEntropy, dCrossEntropy, classCounts, meanDicts, mode, gini, entropy, variance,
       error, accuracy, meanRelError,
       l1_distance,l2_distance, l2²_distance, cosine_distance, lse, sterling,
       #normalFixedSd, logNormalFixedSd,
       radialKernel, polynomialKernel


@enum Verbosity NONE=0 LOW=10 STD=20 HIGH=30 FULL=40

"""
    FIXEDSEED

Fixed seed to allow reproducible results.
This is the seed used to obtain the same results under unit tests.

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`             # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG)`        # always produce the same result (new rng object on each call)
"""
const FIXEDSEED = 123

"""
    FIXEDRNG

Fixed ring to allow reproducible results

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`         # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG))`   # always produce the same result (new rng object on each function call)

"""
const FIXEDRNG  = MersenneTwister(FIXEDSEED) #StableRNG(FIXEDSEED) Random.default_rng()

macro codeLocation()
    return quote
        st = stacktrace(backtrace())
        myf = ""
        for frm in st
            funcname = frm.func
            if frm.func != :backtrace && frm.func!= Symbol("macro expansion")
                myf = frm.func
                break
            end
        end
        println("Running function ", $("$(__module__)"),".$(myf) at ",$("$(__source__.file)"),":",$("$(__source__.line)"))
        println("Type `]dev BetaML` to modify the source code (this would change its location on disk)")
    end
end
"""
    generateParallelRngs(rng::AbstractRNG, n::Integer)

For multi-threaded models return n rings (one per thread) to be used in threaded comutations.

Note that each ring is a _copy_, at different state, of the original random ring.
This means that code that _use_ these RNGs will not change the original RNGF state.

Works only for the default Julia RNG.

Use it with `rngs = generateParallelRngs(rng,Threads.nthreads())`

"""
function generateParallelRngs(rng::AbstractRNG, n::Integer)
    step = rand(rng,big(10)^20:big(10)^40) # making the step random too !
    rngs = Vector{Union{MersenneTwister,Random._GLOBAL_RNG}}(undef, n) # Vector{typeof(rng)}(undef, n)
    rngs[1] = copy(rng)
    for i = 2:n
        rngs[i] = Future.randjump(rngs[i-1], step)
    end
    return rngs
end

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
#makeMatrix(x::AbstractArray{T,1}) where {T} =  reshape(x, (size(x)...,1))
#makeMatrix(x::AbstractArray{T}) where {T} = x

"""Return wheather an array is sortable, i.e. has methos issort defined"""
issortable(::AbstractArray{T,N})  where {T,N} = hasmethod(isless, Tuple{nonmissingtype(T),nonmissingtype(T)})

"""
    getPermutations(v::AbstractArray{T,1};keepStructure=false)

Return a vector of either (a) all possible permutations (uncollected) or (b) just those based on the unique values of the vector

Useful to measure accuracy where you don't care about the actual name of the labels, like in unsupervised classifications (e.g. clustering)

"""
function getPermutations(v::AbstractArray{T,1};keepStructure=false) where {T}
    if !keepStructure
        return Combinatorics.permutations(v)
    else
        classes       = unique(v)
        nCl           = length(classes)
        N             = size(v,1)
        pSet          = Combinatorics.permutations(1:nCl)
        nP            = length(pSet)
        vPermutations = fill(similar(v),nP)
        vOrigIdx      = [findfirst(x -> x == v[i] , classes) for i in 1:N]
        for (pIdx,perm) in enumerate(pSet)
            vPermutations[pIdx] = classes[perm[vOrigIdx]] # permuted specific version
        end
        return vPermutations
    end
end


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

import Base.findfirst, Base.findall
findfirst(el::T,cont::Array{T};returnTuple=true) where {T} = ndims(cont) > 1 && returnTuple ? Tuple(findfirst(x -> isequal(x,el),cont)) : findfirst(x -> isequal(x,el),cont)
findall(el::T, cont::Array{T};returnTuple=true) where {T} = ndims(cont) > 1 && returnTuple ? Tuple.(findall(x -> isequal(x,el),cont)) : findall(x -> isequal(x,el),cont)


"""
    integerEncoder(y;unique)

Encode an array of T to an array of integers using the their position in the unique vector if the input array

# Parameters:
- `y`: The vector to encode
- `unique`: Wether the vector to encode is already made of unique elements [def: `false`]
# Return:
- A vector of [1,length(Y)] integers corresponding to the position of each element in the "unique" version of the original input
# Note:
- Attention that while this function creates a ordered (and sortable) set, it is up to the user to be sure that this "property" is not indeed used in his code if the unencoded data is indeed unordered.
# Example:
```
julia> integerEncoder(["aa","cc","cc","bb","cc","aa"]) # out: [1, 2, 2, 3, 2, 1]
```
"""
function integerEncoder(Y::AbstractVector;unique=false)
    uniqueY = unique ? Y :  Base.unique(Y)
    nY      = length(Y)
    nU      = length(uniqueY)
    O       = findfirst.(Y,Ref(uniqueY))
    return O
end

"""
    integerDecoder(y,target::AbstractVector{T};unique)

Decode an array of integers to an array of T corresponding to the elements of `target`

# Parameters:
- `y`: The vector to decode
- `target`: The vector of elements to use for the encoding
- `unique`: Wether `target` is already made of unique elements [def: `true`]
# Return:
- A vector of length(y) elements corresponding to the (unique) target elements at the position y
# Example:
```
julia> integerDecoder([1, 2, 2, 3, 2, 1],["aa","cc","bb"]) # out: ["aa","cc","cc","bb","cc","aa"]
```
"""
function integerDecoder(Y,target::AbstractVector{T};unique=true) where{T}
    uniqueTarget =  unique ? target :  Base.unique(target)
    nY           =  length(Y)
    nU           =  length(uniqueTarget)
    return map(i -> uniqueTarget[i], Y )
end

"""
  batch(n,bSize;sequential=false,rng)

Return a vector of `bSize` vectors of indeces from `1` to `n`.
Randomly unless the optional parameter `sequential` is used.

# Example:
```julia
julia> Utils.batch(6,2,sequential=true)
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
 ```
"""
function batch(n::Integer,bSize::Integer;sequential=false,rng = Random.GLOBAL_RNG)
    ridx = sequential ? collect(1:n) : shuffle(rng,1:n)
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
    partition(data,parts;shuffle=true)

Partition (by rows) one or more matrices according to the shares in `parts`.

# Parameters
* `data`: A matrix/vector or a vector of matrices/vectors
* `parts`: A vector of the required shares (must sum to 1)
* `shufle`: Whether to randomly shuffle the matrices (preserving the relative order between matrices)
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Example:
```julia
julia> x = [1:10 11:20]
julia> y = collect(31:40)
julia> ((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3])
 ```
 """
function partition(data::AbstractArray{T,1},parts::AbstractArray{Float64,1};shuffle=true,rng = Random.GLOBAL_RNG) where T <: AbstractArray
        # the sets of vector/matrices
        n = size(data[1],1)
        if !all(size.(data,1) .== n)
            @error "All matrices passed to `partition` must have the same number of rows"
        end
        ridx = shuffle ? Random.shuffle(rng,1:n) : collect(1:n)
        return partition.(data,Ref(parts);shuffle=shuffle, fixedRIdx = ridx,rng=rng)
end

function partition(data::AbstractArray{T,N}, parts::AbstractArray{Float64,1};shuffle=true,fixedRIdx=Int64[],rng = Random.GLOBAL_RNG) where {T,N}
    # the individual vecto/matrix
    data = makeMatrix(data)
    n = size(data,1)
    nParts = size(parts)
    toReturn = []
    if !(sum(parts) ≈ 1)
        @error "The sum of `parts` in `partition` should total to 1."
    end
    ridx = fixedRIdx
    if (isempty(ridx))
       ridx = shuffle ? Random.shuffle(rng, 1:n) : collect(1:n)
    end
    current = 1
    cumPart = 0.0
    for (i,p) in enumerate(parts)
        cumPart += parts[i]
        final = i == nParts ? n : Int64(round(cumPart*n))
        if N == 1
            push!(toReturn,data[ridx[current:final]])
        else
            push!(toReturn,data[ridx[current:final],:])
        end
        current = (final +=1)
    end
    return toReturn
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
    scale(x,scaleFactors;rev)

Perform a linear scaling of x using scaling factors `scaleFactors`.

# Parameters
- `x`: The (n × d) dimension matrix to scale on each dimension d
- `scalingFactors`: A tuple of the constant and multiplicative scaling factor
respectively [def: the scaling factors needed to scale x to mean 0 and variance 1]
- `rev`: Whether to invert the scaling [def: `false`]

# Return
- The scaled matrix

# Notes:
- Also available `scale!(x,scaleFactors)` for in-place scaling.
- Retrieve the scale factors with the `getScaleFactors()` function
"""
function scale(x,scaleFactors=(-mean(x,dims=1),1 ./ sqrt.(var(x,corrected=false,dims=1))); rev=false )
    if (!rev)
      y = (x .+ scaleFactors[1]) .* scaleFactors[2]
    else
      y = (x ./ scaleFactors[2]) .- scaleFactors[1]
    end
    return y
end
function scale!(x,scaleFactors=(-mean(x,dims=1),1 ./ sqrt.(var(x,corrected=false,dims=1))); rev=false)
    if (!rev)
        x .= (x .+ scaleFactors[1]) .* scaleFactors[2]
    else
        x .= (x ./ scaleFactors[2]) .- scaleFactors[1]
    end
    return nothing
end

"""
pca(X;K,error)

Perform Principal Component Analysis returning the matrix reprojected among the dimensions of maximum variance.

# Parameters:
- `X` : The (N,D) data to reproject
- `K` : The number of dimensions to maintain (with K<=D) [def: `nothing`]
- `error`: The maximum approximation error that we are willing to accept [def: `0.05`]

# Return:
- A named tuple with:
  - `X`: The reprojected (NxK) matrix with the column dimensions organized in descending order of of the proportion of explained variance
  - `K`: The number of dimensions retieved
  - `error`: The actual proportion of variance not explained in the reprojected dimensions
  - `P`: The (D,K) matrix of the eigenvectors associated to the K-largest eigenvalues used to reproject the data matrix
  - `explVarByDim`: An array of dimensions D with the share of the cumulative variance explained by dimensions (the last element being always 1.0)

# Notes:
- If `K` is provided, the parameter `error` has no effect.
- If one doesn't know _a priori_ the error that she/he is willling to accept, nor the wished number of dimensions, he/she can run this pca function with `out = pca(X,K=size(X,2))` (i.e. with K=D), analise the proportions of explained cumulative variance by dimensions in `out.explVarByDim`, choose the number of dimensions K according to his/her needs and finally pick from the reprojected matrix only the number of dimensions needed, i.e. `out.X[:,1:K]`.

# Example:
```julia
julia> X = [1 10 100; 1.1 15 120; 0.95 23 90; 0.99 17 120; 1.05 8 90; 1.1 12 95]
6×3 Array{Float64,2}:
 1.0   10.0  100.0
 1.1   15.0  120.0
 0.95  23.0   90.0
 0.99  17.0  120.0
 1.05   8.0   90.0
 1.1   12.0   95.0
 julia> X = pca(X,error=0.05).X
6×2 Array{Float64,2}:
  3.1783   100.449
  6.80764  120.743
 16.8275    91.3551
  8.80372  120.878
  1.86179   90.3363
  5.51254   95.5965
```

"""
function pca(X;K=nothing,error=0.05)
    # debug
    #X = [1 10 100; 1.1 15 120; 0.95 23 90; 0.99 17 120; 1.05 8 90; 1.1 12 95]
    #K = nothing
    #error=0.05

    (N,D) = size(X)
    if !isnothing(K) && K > D
        @error("The parameter K must be ≤ D")
    end
    Σ = (1/N) * X'*(I-(1/N)*ones(N)*ones(N)')*X
    E = eigen(Σ) # eigenvalues are ordered from the smallest to the largest
    totVar  = sum(E.values)
    explVarByDim = [sum(E.values[D-k+1:end])/totVar for k in 1:D]
    propVarExplained = 0.0
    if K == nothing
        for k in 1:D
            if explVarByDim[k] >= (1 - error)
                propVarExplained  = explVarByDim[k]
                K                 = k
                break
            end
        end
    else
        propVarExplained = explVarByDim[K]
    end

    P = E.vectors[:,D-K+1:end]

    return (X=X*P,K=K,error=1-propVarExplained,P=P,explVarByDim=explVarByDim)
end


"""
    colsWithMissing(x)

Retuyrn an array with the ids of the columns where there is at least a missing value.
"""
function colsWithMissing(x)
    colsWithMissing = Int64[]
    (N,D) = size(x)
    for d in 1:D
        for n in 1:N
            if ismissing(x[n,d])
                push!(colsWithMissing,d)
                break
            end
        end
    end
    return colsWithMissing
end


# ------------------------------------------------------------------------------
# Various neural network loss functions as well their derivatives
"""
   squaredCost(ŷ,y)

Compute the squared costs between a vector of prediction and one of observations as (1/2)*norm(y - ŷ)^2.

Aside the 1/2 term correspond to the squared l-2 norm distance and when it is averaged on multiple datapoints corresponds to the Mean Squared Error.
It is mostly used for regression problems.
"""
squaredCost(ŷ,y)   = (1/2)*norm(y - ŷ)^2
dSquaredCost(ŷ,y)  = ( ŷ - y)
"""
   crossEntropy(ŷ, y; weight)

Compute the (weighted) cross-entropy between the predicted and the sampled probability distributions.

To be used in classification problems.

"""
crossEntropy(ŷ, y; weight = ones(eltype(y),length(y)))  = -sum(y .* log.(ŷ .+ 1e-15) .* weight)
dCrossEntropy(ŷ, y; weight = ones(eltype(y),length(y))) = - y .* weight ./ (ŷ .+ 1e-15)

# ------------------------------------------------------------------------------
# Various neural network activation functions as well their derivatives

#identity(x)          = x already in Julia base
didentity(x)          = one(x)
""" relu(x) \n\n Rectified Linear Unit \n\n https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf"""
relu(x)               = max(zero(x), x)
""" drelu(x) \n\n Rectified Linear Unit \n\n https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf"""
drelu(x)              = x <= zero(x) ? zero(x) : one(x)
"""elu(x; α=1) with α > 0 \n\n https://arxiv.org/pdf/1511.07289.pdf"""
elu(x; α=one(x))      = x > zero(x) ? x : α *(exp(x) - one(x))
"""delu(x; α=1) with α > 0 \n\n https://arxiv.org/pdf/1511.07289.pdf"""
delu(x; α=one(x))      = x > zero(x) ? one(x) : elu(x, α=α) + α
"""celu(x; α=1) \n\n https://arxiv.org/pdf/1704.07483.pdf"""
celu(x; α=one(x))     = max(zero(x),x)+ min(zero(x), α *(exp(x / α) - one(x) ))
#celu(x; α=one(x))    = if x >= zero(x) x/α else exp(x/α)-one(x) end
"""dcelu(x; α=1) \n\n https://arxiv.org/pdf/1704.07483.pdf"""
dcelu(x; α=one(x))    = x >= zero(x) ? one(x) : exp(x/α)
"""plu(x;α=0.1,c=1) \n\n Piecewise Linear Unit \n\n https://arxiv.org/pdf/1809.09534.pdf"""
plu(x;α=0.1,c=one(x)) = max(α*(x+c)-c,min(α*(x-c)+c,x)) # convert(eltype(x), α)
"""dplu(x;α=0.1,c=1) \n\n Piecewise Linear Unit derivative \n\n https://arxiv.org/pdf/1809.09534.pdf"""
dplu(x;α=0.1,c=one(x)) = ( ( x >= (α*(x+c)-c)  &&  x <= (α*(x+c)+c) ) ? one(x) : α ) # convert(eltype(x), α)

#tanh(x) already in Julia base
"""dtanh(x)"""
dtanh(x)              = sech(x)^2  # = 1-tanh(x)^2
"""sigmoid(x)"""
sigmoid(x)            = one(x)/(one(x)+exp(-x))
"""dsigmoid(x)"""
dsigmoid(x)           = exp(-x)*sigmoid(x)^2
"""softmax (x; β=1) \n\n The input x is a vector. Return a PMF"""
softmax(x; β=one.(x)) = exp.((β .* x) .- lse(β .* x))  # efficient implementation of softmax(x)  = exp.(x) ./  sum(exp.(x))
""" dsoftmax(x; β=1) \n\n Derivative of the softmax function \n\n https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/"""
function dsoftmax(x; β=one(x[1]))
    x = makeColVector(x)
    d = length(x)
    out = zeros(d,d)
    y = softmax(x,β=β)
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

"""softplus(x) \n\n https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus"""
softplus(x)           = log(one(x) + exp(x))
"""dsoftplus(x) \n\n https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus"""
dsoftplus(x)          = 1/(1+exp(-x))
""" mish(x) \n\n https://arxiv.org/pdf/1908.08681v1.pdf"""
mish(x)               = x*tanh(softplus(x))
""" dmish(x) \n\n https://arxiv.org/pdf/1908.08681v1.pdf"""
dmish(x) = x*(1 - tanh(log(exp(x) + 1))^2)*exp(x)/(exp(x) + 1) + tanh(log(exp(x) + 1))


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

""" accuracy(ŷ,y;ignoreLabels=false) - Categorical accuracy between two vectors (T vs T). If """
function accuracy(ŷ::AbstractArray{T,1},y::AbstractArray{T,1}; ignoreLabels=false)  where {T}
    # See here for better performances: https://discourse.julialang.org/t/permutations-of-a-vector-that-retain-the-vector-structure/56790/7
    if(!ignoreLabels)
        return sum(ŷ .== y)/length(ŷ)
    else
        classes  = unique(y)
        nCl      = length(classes)
        N        = size(y,1)
        pSet     =  collect(permutations(1:nCl))
        bestAcc  = -Inf
        yOrigIdx = [findfirst(x -> x == y[i] , classes) for i in 1:N]
        ŷOrigIdx = [findfirst(x -> x == ŷ[i] , classes) for i in 1:N]
        for perm in pSet
            py = perm[yOrigIdx] # permuted specific version
            acc = sum(ŷOrigIdx .== py)/N
            if acc > bestAcc
                bestAcc = acc
            end
        end
        return bestAcc
    end
end

""" error(ŷ,y;ignoreLabels=false) - Categorical error (T vs T)"""
error(ŷ::AbstractArray{T,1},y::AbstractArray{T,1}; ignoreLabels=false) where {T} = (1 - accuracy(ŷ,y;ignoreLabels=ignoreLabels) )


"""
    accuracy(ŷ,y;tol)
Categorical accuracy with probabilistic prediction of a single datapoint (PMF vs Int).

Use the parameter tol [def: `1`] to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.
"""
function accuracy(ŷ::Array{T,1},y::Int64;tol=1) where {T <: Number}
    sIdx = sortperm(ŷ)[end:-1:1]
    if ŷ[y] in ŷ[sIdx[1:min(tol,length(sIdx))]]
        return 1
    else
        return 0
    end
end

"""
    accuracy(ŷ,y;tol)

Categorical accuracy with probabilistic prediction of a single datapoint given in terms of a dictionary of probabilities (Dict{T,Float64} vs T).

# Parameters:
- `ŷ`: The returned probability mass function in terms of a Dictionary(Item1 => Prob1, Item2 => Prob2, ...)
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].
"""
function accuracy(ŷ::Dict{T,Float64},y::T;tol=1) where {T}
    if !(y in keys(ŷ)) return 0 end
    sIdx = sortperm(collect(values(ŷ)))[end:-1:1] # sort by decreasing values of the dictionary values
    sKeys = collect(keys(ŷ))[sIdx][1:min(tol,length(sIdx))]  # retrieve the corresponding keys
    if y in sKeys
        return 1
    else
        return 0
    end
end

@doc raw"""
   accuracy(ŷ,y;tol,ignoreLabels)

Categorical accuracy with probabilistic predictions of a dataset (PMF vs Int).

# Parameters:
- `ŷ`: An (N,K) matrix of probabilities that each ``\hat y_n`` record with ``n \in 1,....,N``  being of category ``k`` with $k \in 1,...,K$.
- `y`: The N array with the correct category for each point $n$.
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].
- `ignoreLabels`: Whether to ignore the specific label order in y. Useful for unsupervised learning algorithms where the specific label order don't make sense [def: false]

"""
function accuracy(ŷ::Array{T,2},y::Array{Int64,1};tol=1,ignoreLabels=false) where {T <: Number}
    (N,D) = size(ŷ)
    pSet = ignoreLabels ? collect(permutations(1:D)) : [collect(1:D)]
    bestAcc = -Inf
    for perm in pSet
        pŷ = hcat([ŷ[:,c] for c in perm]...)
        acc = sum([accuracy(pŷ[i,:],y[i],tol=tol) for i in 1:N])/N
        if acc > bestAcc
            bestAcc = acc
        end
    end
    return bestAcc
end

@doc raw"""
   accuracy(ŷ,y;tol)

Categorical accuracy with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (Dict{T,Float64} vs T).

# Parameters:
- `ŷ`: A narray where each item is the estimated probability mass function in terms of a Dictionary(Item1 => Prob1, Item2 => Prob2, ...)
- `y`: The N array with the correct category for each point $n$.
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].

"""
function accuracy(ŷ::Array{Dict{T,Float64},1},y::Array{T,1};tol=1) where {T}
    N = size(ŷ,1)
    acc = sum([accuracy(ŷ[i],y[i],tol=tol) for i in 1:N])/N
    return acc
end


""" error(ŷ,y) - Categorical error with probabilistic prediction of a single datapoint (PMF vs Int). """
error(ŷ::Array{T,1},y::Int64;tol=1) where {T <: Number} = 1 - accuracy(ŷ,y;tol=tol)
""" error(ŷ,y) - Categorical error with probabilistic predictions of a dataset (PMF vs Int). """
error(ŷ::Array{T,2},y::Array{Int64,1};tol=1) where {T <: Number} = 1 - accuracy(ŷ,y;tol=tol)
""" error(ŷ,y) - Categorical error with with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (Dict{T,Float64} vs T). """
error(ŷ::Array{Dict{T,Float64},1},y::Array{T,1};tol=1) where {T} = 1 - accuracy(ŷ,y;tol=tol)

"""
  meanRelError(ŷ,y;normDim=true,normRec=true,p=1)

Compute the mean relative error (l-1 based by default) between ŷ and y.

There are many ways to compute a mean relative error. In particular, if normRec (normDim) is set to true, the records (dimensions) are normalised, in the sense that it doesn't matter if a record (dimension) is bigger or smaller than the others, the relative error is first computed for each record (dimension) and then it is averaged.
With both `normDim` and `normRec` set to `false` the function returns the relative mean error; with both set to `true` (default) it returns the mean relative error (i.e. with p=1 the "[mean absolute percentage error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)")
The parameter `p` [def: `1`] controls the p-norm used to define the error.

"""
function meanRelError(ŷ,y;normDim=true,normRec=true,p=1)
    ŷ = makeMatrix(ŷ)
    y = makeMatrix(y)
    (n,d) = size(y)
    #ϵ = abs.(ŷ-y) .^ p
    if (!normDim && !normRec) # relative mean error
        avgϵRel = (sum(abs.(ŷ-y).^p)^(1/p) / (n*d)) / (sum( abs.(y) .^p)^(1/p) / (n*d)) # (avg error) / (avg y)
        # avgϵRel = (norm((ŷ-y),p)/(n*d)) / (norm(y,p) / (n*d))
    elseif (!normDim && normRec) # normalised by record (i.e. all records play the same weigth)
        avgϵRel_byRec = (sum(abs.(ŷ-y) .^ (1/p),dims=2).^(1/p) ./ d) ./   (sum(abs.(y) .^ (1/p) ,dims=2) ./d)
        avgϵRel = mean(avgϵRel_byRec)
    elseif (normDim && !normRec) # normalised by dimensions (i.e.  all dimensions play the same weigth)
        avgϵRel_byDim = (sum(abs.(ŷ-y) .^ (1/p),dims=1).^(1/p) ./ n) ./   (sum(abs.(y) .^ (1/p) ,dims=1) ./n)
        avgϵRel = mean(avgϵRel_byDim)
    else # mean relative error
        avgϵRel = sum(abs.((ŷ-y)./ y).^p)^(1/p)/(n*d) # avg(error/y)
        # avgϵRel = (norm((ŷ-y)./ y,p)/(n*d))
    end
    return avgϵRel
end

"""bic(lL,k,n) -  Bayesian information criterion (lower is better)"""
bic(lL,k,n) = k*log(n)-2*lL
"""aic(lL,k) -  Akaike information criterion (lower is better)"""
aic(lL,k)   = 2*k-2*lL

"""
   classCounts(x)

Return a dictionary that counts the number of each unique item (rows) in a dataset.

"""
function classCounts(x)
    dims = ndims(x)
    if dims == 1
        T = eltype(x)
    else
        T = Array{eltype(x),1}
    end
    counts = Dict{T,Int64}()  # a dictionary of label -> count.
    for i in 1:size(x,1)
        if dims == 1
            label = x[i]
        else
            label = x[i,:]
        end
        if !(label in keys(counts))
            counts[label] = 1
        else
            counts[label] += 1
        end
    end
    return counts
end

"""
  mode(dicts)

Given a vector of dictionaries representing probabilities it returns the mode of each element in terms of the key

Use it to return a unique value from a multiclass classifier returning probabilities.

"""
function mode(dicts::AbstractArray{Dict{T,Float64}}) where {T}
    return getindex.(findmax.(dicts),2)
end

"""
   meanDicts(dicts)

Compute the mean of the values of an array of dictionaries.

Given `dicts` an array of dictionaries, `meanDicts` first compute the union of the keys and then average the values.
If the original valueas are probabilities (non-negative items summing to 1), the result is also a probability distribution.

"""
function meanDicts(dicts; weights=ones(length(dicts)))
    T = eltype(keys(dicts[1]))
    allkeys = union([keys(i) for i in dicts]...)
    outDict = Dict{T,Float64}()
    ndicts = length(dicts)
    totWeights = sum(weights)
    for k in allkeys
        v = 0
        for (i,d) in enumerate(dicts)
            if k in keys(d)
                v += (d[k])*(weights[i]/totWeights)
            end
        end
        outDict[k] = v
    end

    return outDict
end

"""
   gini(x)

Calculate the Gini Impurity for a list of items (or rows).

See: https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain
"""
function gini(x)
    counts = classCounts(x)
    N = size(x,1)
    impurity = 1.0
    for k in keys(counts)
        probₖ = counts[k] / N
        impurity -= probₖ^2
    end
    return impurity
end

"""
   entropy(x)

Calculate the entropy for a list of items (or rows).

See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
"""
function entropy(x)
    counts = classCounts(x)
    N = size(x,1)
    entr = 0.0
    for k in keys(counts)
        probₖ = counts[k] / N
        entr -= probₖ * log2(probₖ)
    end
    return entr
end

"""variance(x) - population variance"""
variance(x) = var(x,corrected=false)

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

#=
# No longer used, as mixtures has been generalised
# ------------------------------------------------------------------------------
# Some common PDFs

""" PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
normalFixedSd(x,μ,σ²)    = (1/(2π*σ²)^(length(x)/2)) * exp(-1/(2σ²)*norm(x-μ)^2)
""" log-PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
logNormalFixedSd(x,μ,σ²) = - (length(x)/2) * log(2π*σ²)  -  norm(x-μ)^2/(2σ²)
=#

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
