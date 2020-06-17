"""
  Clustering.jl file

Clustering and collaborative filtering (via clustering) algorithms

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Clustering.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/Clustering.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FClustering.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 4](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2004%20-%20Unsupervised%20Learning/Unit%2004%20-%20Unsupervised%20Learning.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)
"""

"""
    Clustering module (WIP)

Provide clustering methods and collaborative filtering using clustering methods as backend.

The em algorithm is work in progress, as its API will likely change to account for different type of mixtures.

The module provide the following functions. Use `?[function]` to access their full signature and detailed documentation:

- [`initRepresentatives(X,K;initStrategy,Z₀)`](@ref initRepresentatives): Initialisation strategies for Kmean and Kmedoids
- [`kmeans(X,K;dist,initStrategy,Z₀)](@ref kmeans)`: Classical KMean algorithm
- [`kmedoids(X,K;dist,initStrategy,Z₀)](@ref kmedoids)`: Kmedoids algorithm
- [`emGMM(X,K;p₀,μ₀,σ²₀,tol,msgStep,minVariance,missingValue)](@ref emGMM)`: EM algorithm over GMM with fixed variance
- [`collFilteringGMM(X,K;p₀,μ₀,σ²₀,tol,msgStep,minVariance,missingValue)](@ref collFilteringGMM)`: Collaborative filtering using GMM
"""
module Clustering

using LinearAlgebra, Random, Statistics, Reexport
#using Distributions

@reexport using ..Utils

export initRepresentatives, kmeans, kmedoids, em, predictMissing

abstract type Mixture end
include("Mixtures.jl")

## Helper functions

"""
  initRepresentatives(X,K;initStrategy,Z₀)

Initialisate the representatives for a K-Mean or K-Medoids algorithm

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `initStrategy`: Wheter to select the initial representative vectors:
  * `random`: randomly in the X space
  * `grid`: using a grid approach [default]
  * `shuffle`: selecting randomly within the available points
  * `given`: using a provided set of initial representatives provided in the `Z₀` parameter
 * `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]

# Returns:
* A (K x D) matrix of initial representatives

# Example:
```julia
julia> Z₀ = initRepresentatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initStrategy="given",Z₀=[1.7 15; 3.6 40])
```
"""
function initRepresentatives(X,K;initStrategy="grid",Z₀=nothing)
    X  = makeMatrix(X)
    (N,D) = size(X)
    # Random choice of initial representative vectors (any point, not just in X!)
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    Z = zeros(K,D)
    if initStrategy == "random"
        for i in 1:K
            for j in 1:D
                Z[i,j] = rand(Uniform(minX[j],maxX[j]))
            end
        end
    elseif initStrategy == "grid"
        for d in 1:D
                Z[:,d] = collect(range(minX[d], stop=maxX[d], length=K))
        end
    elseif initStrategy == "given"
        if isnothing(Z₀) error("With the `given` strategy you need to provide the initial set of representatives in the Z₀ parameter.") end
        Z₀ = makeMatrix(Z₀)
        Z = Z₀
    elseif initStrategy == "shuffle"
        zIdx = shuffle(1:size(X)[1])[1:K]
        Z = X[zIdx, :]
    else
        error("initStrategy \"$initStrategy\" not implemented")
    end
    return Z
end


## Basic K-Means Algorithm (Lecture/segment 13.7 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

"""
  kmeans(X,K;dist,initStrategy,Z₀)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (see notes). Default to Euclidean distance.
* `initStrategy`: Wheter to select the initial representative vectors:
  * `random`: randomly in the X space
  * `grid`: using a grid approach [default]
  * `shuffle`: selecting randomly within the available points
  * `given`: using a provided set of initial representatives provided in the `Z₀` parameter
* `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]

# Returns:
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty
* The `dist` parameter can be:
  * Any user defined function accepting two vectors and returning a scalar
  * An anonymous function with the same characteristics (e.g. `dist = (x,y) -> norm(x-y)^2`)
  * One of the above predefined distances: `l1_distance`, `l2_distance`, `l2²_distance`, `cosine_distance`

# Example:
```julia
julia> (clIdx,Z) = kmeans([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3)
```
"""
function kmeans(X,K;dist=(x,y) -> norm(x-y),initStrategy="grid",Z₀=nothing)
    X  = makeMatrix(X)
    (N,D) = size(X)
    # Random choice of initial representative vectors (any point, not just in X!)
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    Z₀ = initRepresentatives(X,K,initStrategy=initStrategy,Z₀=Z₀)
    Z  = Z₀
    cIdx_prev = zeros(Int64,N)

    # Looping
    while true
        # Determining the constituency of each cluster
        cIdx      = zeros(Int64,N)
        for (i,x) in enumerate(eachrow(X))
            cost = Inf
            for (j,z) in enumerate(eachrow(Z))
               if (dist(x,z)  < cost)
                   cost    =  dist(x,z)
                   cIdx[i] = j
               end
            end
        end

        # Determining the new representative by each cluster
        # for (j,z) in enumerate(eachrow(Z))
        for j in  1:K
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            Z[j,:] = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
            #Z[j,:] = median(Cⱼ,dims=1) # for l1 distance
        end

        # Checking termination condition: clusters didn't move any more
        if cIdx == cIdx_prev
            return (cIdx,Z)
        else
            cIdx_prev = cIdx
        end

    end
end

## Basic K-Medoids Algorithm (Lecture/segment 14.3 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)
"""
  kmedoids(X,K;dist,initStrategy,Z₀)

Compute K-Medoids algorithm to identify K clusters of X using distance definition `dist`

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (see notes). Default to Euclidean distance.
* `initStrategy`: Wheter to select the initial representative vectors:
  * `random`: randomly in the X space
  * `grid`: using a grid approach
  * `shuffle`: selecting randomly within the available points [default]
  * `given`: using a provided set of initial representatives provided in the `Z₀` parameter
 * `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]

# Returns:
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty
* The `dist` parameter can be:
  * Any user defined function accepting two vectors and returning a scalar
  * An anonymous function with the same characteristics (e.g. `dist = (x,y) -> norm(x-y)^2`)
  * One of the above predefined distances: `l1_distance`, `l2_distance`, `l2²_distance`, `cosine_distance`

# Example:
```julia
julia> (clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,initStrategy="grid")
```
"""
function kmedoids(X,K;dist=(x,y) -> norm(x-y),initStrategy="shuffle",Z₀=nothing)
    X  = makeMatrix(X)
    (n,d) = size(X)
    # Random choice of initial representative vectors
    Z₀ = initRepresentatives(X,K,initStrategy=initStrategy,Z₀=Z₀)
    Z = Z₀
    cIdx_prev = zeros(Int64,n)

    # Looping
    while true
        # Determining the constituency of each cluster
        cIdx      = zeros(Int64,n)
        for (i,x) in enumerate(eachrow(X))
            cost = Inf
            for (j,z) in enumerate(eachrow(Z))
               if (dist(x,z) < cost)
                   cost =  dist(x,z)
                   cIdx[i] = j
               end
            end
        end

        # Determining the new representative by each cluster (within the points member)
        #for (j,z) in enumerate(eachrow(Z))
        for j in  1:K
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            nⱼ = size(Cⱼ)[1]     # Size of the cluster
            if nⱼ == 0 continue end # empty continuency. Let's not do anything. Stil in the next batch other representatives could move away and points could enter this cluster
            bestCost = Inf
            bestCIdx = 0
            for cIdx in 1:nⱼ      # candidate index
                 candidateCost = 0.0
                 for tIdx in 1:nⱼ # target index
                     candidateCost += dist(Cⱼ[cIdx,:],Cⱼ[tIdx,:])
                 end
                 if candidateCost < bestCost
                     bestCost = candidateCost
                     bestCIdx = cIdx
                 end
            end
            Z[j,:] = reshape(Cⱼ[bestCIdx,:],1,d)
        end

        # Checking termination condition: clusters didn't move any more
        if cIdx == cIdx_prev
            return (cIdx,Z)
        else
            cIdx_prev = cIdx
        end
    end

end


## The EM algorithm (Lecture/segment 16.5 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

"""
  em(X,K;p₀,μ₀,σ²₀,tol,msgStep,minVariance,missingValue)

Compute Expectation-Maximisation algorithm to identify K clusters of X data, i.e. employ a Generative Mixture Model as the underlying probabilistic model.

X can contain missing values in some or all of its dimensions. In such case the learning is done only with the available data.
Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :          A (n x d) data to clusterise
* `K`  :          Number of cluster wanted
* `p₀` :          Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
* `μ₀` :          Initial means (K x d) of the Gaussian [default: `nothing`]
* `σ²₀`:          Initial variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions [default: `nothing`]
* `tol`:          Tolerance to stop the algorithm [default: 10^(-6)]
* `msgStep` :     Iterations between update messages. Use 0 for no updates [default: 10]
* `minVariance`:  Minimum variance for the mixtures [default: 0.25]
* `missingValue`: Value to be considered as missing in the X [default: `missing`]

# Returns:
* A named touple of:
  * `pⱼₓ`: Matrix of size (N x K) of the probabilities of each point i to belong to cluster j
  * `pⱼ` : Probabilities of the categorical distribution (K x 1)
  * `μ`  : Means (K x d) of the Gaussian
  * `σ²` : Variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions
  * `ϵ`  : Vector of the discrepancy (matrix norm) between pⱼₓ and the lagged pⱼₓ at each iteration
  * `lL` : The log-likelihood (without considering the last mixture optimisation)
  * `BIC` : The Bayesian Information Criterion

# Example:
```julia
julia> clusters = emGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=1,missingValue=0)
```
"""

function em(X,K;p₀=nothing,mixtures=[SphericalGaussian() for i in 1:K],tol=10^(-6),msgStep=10,minVariance=0.25,initStrategy="grid")
    # debug:
    #X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
    #K = 3
    #p₀=nothing; tol=0.0001; msgStep=1; minVariance=0.25; initStrategy="grid"
    #mixtures = [SphericalGaussian() for i in 1:K]
    # ---------
    X     = makeMatrix(X)
    (N,D) = size(X)
    pₖ    = isnothing(p₀) ? fill(1/K,K) : p₀

    # Initialisation of the parameters of the mixtures
    initMixtures!(mixtures,X,minVariance=minVariance,initStrategy=initStrategy)

    pₙₖ = zeros(Float64,N,K) # The posteriors, i.e. the prob that item n belong to cluster k
    ϵ = Float64[]

    # Checking dimensions only once (but adding then inbounds doesn't change anything. Still good
    # to provide a nice informative message)
    if size(pₖ) != (K,) || length(mixtures) != K
        error("Error in the dimensions of the inputs. Please check them.")
    end

    # finding empty/non_empty values
    Xmask     =  .! ismissing.(X)
    #XdimCount = sum(Xmask, dims=2)

    lL = -Inf
    while(true)
        oldlL = lL
        # E Step: assigning the posterior prob p(j|xi) and computing the log-Likelihood of the parameters given the set of data
        # (this last one for informative purposes and terminating the algorithm)
        pₙₖlagged = copy(pₙₖ)
        logpₙₖ = log.(pₙₖ)
        lL = 0
        for n in 1:N
            if any(Xmask[n,:]) # if at least one true
                Xu = X[n,Xmask[n,:]]
                logpx = lse([log(pₖ[k] + 1e-16) + lpdf(mixtures[k],Xu,Xmask[n,:]) for k in 1:K])
                lL += logpx
                #px = sum([pⱼ[k]*normalFixedSd(Xu,μ[k,XMask[n,:]],σ²[k]) for k in 1:K])
                for k in 1:K
                    logpₙₖ[n,k] = log(pₖ[k] + 1e-16)+lpdf(mixtures[k],Xu,Xmask[n,:])-logpx
                end
            else
                logpₙₖ[n,:] = log.(pₖ)
            end
        end
        pₙₖ = exp.(logpₙₖ)

        push!(ϵ,norm(pₙₖlagged - pₙₖ))

        # M step: find parameters that maximise the likelihood
        # Updating the probabilities of the different mixtures
        nₖ = sum(pₙₖ,dims=1)'
        n  = sum(nₖ)
        pₖ = nₖ ./ n
        updateParameters!(mixtures, X, pₙₖ, Xmask; minVariance=minVariance)

        # Information. Note the likelihood is whitout accounting for the new mu, sigma
        if msgStep != 0 && (length(ϵ) % msgStep == 0 || length(ϵ) == 1)
            println("Iter. $(length(ϵ)):\tVar. of the post  $(ϵ[end]) \t  Log-likelihood $(lL)")
        end

        # Closing conditions. Note that the logLikelihood is those without considering the new mu,sigma
        if (lL - oldlL) <= (tol * abs(lL))
            npars = npar(mixtures) + (K-1)
            BIC  = lL - (1/2) * npars * log(N)
        #if (ϵ[end] < tol)
           return (pₙₖ=pₙₖ,pₖ=pₖ,mixtures=mixtures,ϵ=ϵ,lL=lL,BIC=BIC)
        end
    end # end while loop
end # end function

#using BenchmarkTools
#@benchmark clusters = emGMM([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,msgStep=0)
#@benchmark clusters = emGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
#@benchmark clusters = emGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
#@benchmark clusters = emGMM([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3,msgStep=0)
#@code_warntype em([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
#using Profile
#Juno.@profiler (for i = 1:1000 em([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0) end)
#Profile.clear()
#Profile.print()

"""
  predictMissing(X,K;p₀,μ₀,σ²₀,tol,msgStep,minVariance)

Fill missing entries in a sparse matrix assuming an underlying Gaussian Mixture probabilistic Model (GMM) and implementing
an Expectation-Maximisation algorithm.

Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :          A (N x D) sparse matrix of data to fill according to a GMM model
* `K`  :          Number of mixtures desired
* `p₀` :          Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
* `μ₀` :          Initial means (K x D) of the Gaussian [default: `nothing`]
* `σ²₀`:          Initial variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions [default: `nothing`]
* `tol`:          Tolerance to stop the algorithm [default: 10^(-6)]
* `msgStep` :     Iterations between update messages. Use 0 for no updates [default: 10]
* `minVariance`:  Minimum variance for the mixtures [default: 0.25]
* `missingValue`: Value to be considered as missing in the X [default: `missing`]

# Returns:
* A named touple of:
  * `̂X̂`    : The Filled Matrix of size (N x D)
  * `nFill`: The number of items filled
  * `lL`   : The log-likelihood (without considering the last mixture optimisation)
  * `BIC`  : The Bayesian Information Criterion

# Example:
```julia
julia>  cFOut = predictMissing([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3,msgStep=1)
```
"""
function predictMissing(X,K;mixtures=[SphericalGaussian() for i in 1:K],tol=10^(-6),msgStep=10,minVariance=0.25)
    emOut = em(X,K;mixtures=mixtures,tol=tol,msgStep=msgStep,minVariance=minVariance)
    (N,D) = size(X)
    #K = size(emOut.μ)[1]
    XMask = .! ismissing.(X)
    nFill = (N * D) - sum(XMask)
    X̂ = copy(X)
    for n in 1:N
        for d in 1:D
            if !XMask[n,d]
                 X̂[n,d] = sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K])
            end
        end
    end
    return (X̂=X̂,nFill=nFill,lL=emOut.lL,BIC=emOut.BIC)
end

end
