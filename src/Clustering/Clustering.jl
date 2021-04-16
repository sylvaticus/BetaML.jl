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

Provide clustering methods and missing values imputation / collaborative filtering / reccomendation systems using clustering methods as backend.

The module provides the following functions. Use `?[function]` to access their full signature and detailed documentation:

- [`initRepresentatives(X,K;initStrategy,Z₀)`](@ref initRepresentatives): Initialisation strategies for Kmean and Kmedoids
- [`kmeans(X,K;dist,initStrategy,Z₀)](@ref kmeans)`: Classical KMean algorithm
- [`kmedoids(X,K;dist,initStrategy,Z₀)](@ref kmedoids)`: Kmedoids algorithm
- [`gmm(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance,initStrategy)](@ref gmm)`: gmm algorithm over GMM
- [`predictMissing(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance)](@ref predictMissing)`: Fill mixing values / collaborative filtering using gmm as backbone

{Spherical|Diagonal|Full}Gaussian mixtures for `gmm` / `predictMissing` are already provided. User defined mixtures can be used defining a struct as subtype of `Mixture` and implementing for that mixture the following functions:
- `initMixtures!(mixtures, X; minVariance, minCovariance, initStrategy)`
- `lpdf(m,x,mask)` (for the e-step)
- `updateParameters!(mixtures, X, pₙₖ; minVariance, minCovariance)` (the m-step)

"""
module Clustering

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays
import Distributions

using  ForceImport
@force using ..Api
@force using ..Utils

export initRepresentatives, kmeans, kmedoids, gmm, predictMissing

abstract type AbstractMixture end
include("Mixtures.jl")

## Helper functions

"""
  initRepresentatives(X,K;initStrategy,Z₀)

Initialisate the representatives for a K-Mean or K-Medoids algorithm

# Parameters:
* `X`: a (N x D) data to clusterise
* `K`: Number of cluster wonted
* `initStrategy`: Whether to select the initial representative vectors:
  * `random`: randomly in the X space
  * `grid`: using a grid approach [default]
  * `shuffle`: selecting randomly within the available points
  * `given`: using a provided set of initial representatives provided in the `Z₀` parameter
 * `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]
 * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A (K x D) matrix of initial representatives

# Example:
```julia
julia> Z₀ = initRepresentatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initStrategy="given",Z₀=[1.7 15; 3.6 40])
```
"""
function initRepresentatives(X,K;initStrategy="grid",Z₀=nothing,rng = Random.GLOBAL_RNG)
    X  = makeMatrix(X)
    (N,D) = size(X)
    # Random choice of initial representative vectors (any point, not just in X!)
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    Z = zeros(K,D)
    if initStrategy == "random"
        for i in 1:K
            for j in 1:D
                Z[i,j] = rand(rng,Distributions.Uniform(minX[j],maxX[j]))
            end
        end
    elseif initStrategy == "grid"
        for d in 1:D
                # same "space" for each class on each dimension
                Z[:,d] = collect(range(minX[d] + (maxX[d]-minX[d])/(K*2) , stop=maxX[d] - (maxX[d]-minX[d])/(K*2)  , length=K))
                #ex: collect(range(minX[d], stop=maxX[d], length=K))
                #collect(range(s+(e-s)/(K*2), stop=e-(e-s)/(K*2), length=K))
        end
    elseif initStrategy == "given"
        if isnothing(Z₀) error("With the `given` strategy you need to provide the initial set of representatives in the Z₀ parameter.") end
        Z₀ = makeMatrix(Z₀)
        Z = Z₀
    elseif initStrategy == "shuffle"
        zIdx = shuffle(rng,1:size(X)[1])[1:K]
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
* `initStrategy`: Whether to select the initial representative vectors:
  * `random`: randomly in the X space
  * `grid`: using a grid approach [default]
  * `shuffle`: selecting randomly within the available points
  * `given`: using a provided set of initial representatives provided in the `Z₀` parameter
* `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

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
function kmeans(X,K;dist=(x,y) -> norm(x-y),initStrategy="grid",Z₀=nothing,rng = Random.GLOBAL_RNG)
    X  = makeMatrix(X)
    (N,D) = size(X)
    # Random choice of initial representative vectors (any point, not just in X!)
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    Z₀ = initRepresentatives(X,K,initStrategy=initStrategy,Z₀=Z₀,rng=rng)
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
            #debug  =  sum(Cⱼ,dims=1)
            #debug2 = size(Cⱼ)[1]
            if size(Cⱼ)[1] > 0
                Z[j,:] = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
            else
                # move toward the center if no costituency
                xAvg = mean(X,dims=1)'
                Z[j,:] = Z[j,:] .+ ((xAvg - Z[j,:]) .* 0.01)
                #debug  = sum(Cⱼ,dims=1)
                # debug2 = size(Cⱼ)[1]
            end
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
* `initStrategy`: Whether to select the initial representative vectors:
  * `random`: randomly in the X space
  * `grid`: using a grid approach
  * `shuffle`: selecting randomly within the available points [default]
  * `given`: using a provided set of initial representatives provided in the `Z₀` parameter
 * `Z₀`: Provided (K x D) matrix of initial representatives (used only together with the `given` initStrategy) [default: `nothing`]
 * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

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
function kmedoids(X,K;dist=(x,y) -> norm(x-y),initStrategy="grid",Z₀=nothing,rng = Random.GLOBAL_RNG)
    X  = makeMatrix(X)
    (n,d) = size(X)
    # Random choice of initial representative vectors
    Z₀ = initRepresentatives(X,K,initStrategy=initStrategy,Z₀=Z₀,rng=rng)
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


## The gmm algorithm (Lecture/segment 16.5 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

# no longer true with the numerical trick implemented
# - For mixtures with full covariance matrix (i.e. `FullGaussian(μ,σ²)`) the minCovariance should NOT be set equal to the minVariance, or if the covariance matrix goes too low, it will become singular and not invertible.
"""
  gmm(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance,initStrategy)

Compute Expectation-Maximisation algorithm to identify K clusters of X data, i.e. employ a Generative Mixture Model as the underlying probabilistic model.

X can contain missing values in some or all of its dimensions. In such case the learning is done only with the available data.
Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :           A (n x d) data to clusterise
* `K`  :           Number of cluster wanted
* `p₀` :           Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
* `mixtures`:      An array (of length K) of the mixture to employ (see notes) [def: `[DiagonalGaussian() for i in 1:K]`]
* `tol`:           Tolerance to stop the algorithm [default: 10^(-6)]
* `verbosity`:     A verbosity parameter regulating the information messages frequency [def: `STD`]
* `minVariance`:   Minimum variance for the mixtures [default: 0.05]
* `minCovariance`: Minimum covariance for the mixtures with full covariance matrix [default: 0]. This should be set different than minVariance (see notes).
* `initStrategy`:  Mixture initialisation algorithm [def: `kmeans`]
* `maxIter`:       Maximum number of iterations [def: `-1`, i.e. ∞]
* `rng`:           Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A named touple of:
  * `pₙₖ`:      Matrix of size (N x K) of the probabilities of each point i to belong to cluster j
  * `pₖ`:       Probabilities of the categorical distribution (K x 1)
  * `mixtures`: Vector (K x 1) of the estimated underlying distributions
  * `ϵ`:        Vector of the discrepancy (matrix norm) between pⱼₓ and the lagged pⱼₓ at each iteration
  * `lL`:       The log-likelihood (without considering the last mixture optimisation)
  * `BIC`:      The Bayesian Information Criterion (lower is better)
  * `AIC`:      The Akaike Information Criterion (lower is better)

 # Notes:
 - The mixtures currently implemented are `SphericalGaussian(μ,σ²)`,`DiagonalGaussian(μ,σ²)` and `FullGaussian(μ,σ²)`
 - Reasonable choices for the minVariance/Covariance depends on the mixture. For example 0.25 seems a reasonable value for the SphericalGaussian, 0.05 seems better for the DiagonalGaussian, and FullGaussian seems to prefer either very low values of variance/covariance (e.g. `(0.05,0.05)` ) or very big but similar ones (e.g. `(100,100)` ).
 - For `initStrategy`, look at the documentation of `initMixtures!` for the mixture you want. The provided gaussian mixtures support `grid`, `kmeans` or `given`. `grid` is faster (expecially if X contains missing values), but `kmeans` often provides better results.

 # Resources:
 - [Paper describing gmm with missing values](https://doi.org/10.1016/j.csda.2006.10.002)
 - [Class notes from MITx 6.86x (Sec 15.9)](https://stackedit.io/viewer#!url=https://github.com/sylvaticus/MITx_6.86x/raw/master/Unit 04 - Unsupervised Learning/Unit 04 - Unsupervised Learning.md)
 - [Limitations of gmm](https://www.r-craft.org/r-news/when-not-to-use-gaussian-mixture-model-gmm-clustering/)

# Example:
```julia
julia> clusters = gmm([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,verbosity=HIGH)
```
"""
function gmm(X,K;p₀=nothing,mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minVariance=0.05,minCovariance=0.0,initStrategy="kmeans",maxIter=-1,rng = Random.GLOBAL_RNG)
    if verbosity > STD
        @codeLocation
    end
    # debug:
    #X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
    #K = 3
    #p₀=nothing; tol=0.0001; msgStep=1; minVariance=0.25; initStrategy="grid"
    #mixtures = [SphericalGaussian() for i in 1:K]
    # ---------
    X     = makeMatrix(X)
    (N,D) = size(X)
    pₖ    = isnothing(p₀) ? fill(1/K,K) : p₀

    # no longer true with the numerical trick implemented
    #if (minVariance == minCovariance)
    #    @warn("Setting the minVariance equal to the minCovariance may lead to singularity problems for mixtures with full covariance matrix.")
    #end

    msgStepMap = Dict(NONE => 0, LOW=>100, STD=>20, HIGH=>5, FULL=>1)
    msgStep    = msgStepMap[verbosity]


    # Initialisation of the parameters of the mixtures
    initMixtures!(mixtures,X,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,rng=rng)

    pₙₖ = zeros(Float64,N,K) # The posteriors, i.e. the prob that item n belong to cluster k
    ϵ = Float64[]

    # Checking dimensions only once (but adding then inbounds doesn't change anything. Still good
    # to provide a nice informative message)
    if size(pₖ,1) != K || length(mixtures) != K
        error("Error in the dimensions of the inputs. Please check them.")
    end

    # finding empty/non_empty values
    Xmask     =  .! ismissing.(X)
    #XdimCount = sum(Xmask, dims=2)

    lL = -Inf
    iter = 1
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
                #=if (length(ϵ) == 2)
                    println("here I am")
                    for m in mixtures[3:end]
                        println(m.μ)
                        println(m.σ²)
                        println(Xu)
                        println(Xmask[n,:])
                        lpdf(m,Xu,Xmask[n,:])
                        println("here I am partially")
                    end
                    println("here I am dead")
                end=#
                logpx = lse([log(pₖ[k] + 1e-16) + lpdf(mixtures[k],Xu,Xmask[n,:]) for k in 1:K])
                lL += logpx
                #px = sum([pⱼ[k]*normalFixedSd(Xu,μ[k,XMask[n,:]],σ²[k]) for k in 1:K])
                #println(n)
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
        updateParameters!(mixtures, X, pₙₖ; minVariance=minVariance,minCovariance=minCovariance)

        # Information. Note the likelihood is whitout accounting for the new mu, sigma
        if msgStep != 0 && (length(ϵ) % msgStep == 0 || length(ϵ) == 1)
            println("Iter. $(length(ϵ)):\tVar. of the post  $(ϵ[end]) \t  Log-likelihood $(lL)")
        end

        # Closing conditions. Note that the logLikelihood is those without considering the new mu,sigma
        if ((lL - oldlL) <= (tol * abs(lL))) || (maxIter > 0 && iter == maxIter)
            npars = npar(mixtures) + (K-1)
            #BIC  = lL - (1/2) * npars * log(N)
            BICv = bic(lL,npars,N)
            AICv = aic(lL,npars)
        #if (ϵ[end] < tol)
           return (pₙₖ=pₙₖ,pₖ=pₖ,mixtures=mixtures,ϵ=ϵ,lL=lL,BIC=BICv,AIC=AICv)
       else
            iter += 1
       end
    end # end while loop
end # end function

#using BenchmarkTools
#@benchmark clusters = emGMM([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,msgStep=0)
#@benchmark clusters = emGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
#@benchmark clusters = emGMM([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
#@benchmark clusters = emGMM([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3,msgStep=0)
#@code_warntype gmm([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0)
#using Profile
#Juno.@profiler (for i = 1:1000 gmm([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,msgStep=0,missingValue=0) end)
#Profile.clear()
#Profile.print()

#  - For mixtures with full covariance matrix (i.e. `FullGaussian(μ,σ²)`) the minCovariance should NOT be set equal to the minVariance, or if the covariance matrix goes too low, it will become singular and not invertible.
"""
  predictMissing(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance)

Fill missing entries in a sparse matrix assuming an underlying Gaussian Mixture probabilistic Model (GMM) and implementing
an Expectation-Maximisation algorithm.

While the name of the function is `predictMissing`, the function can be used also for system reccomendation / collaborative filtering and GMM-based regressions.

Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :           A (N x D) sparse matrix of data to fill according to a GMM model
* `K`  :           Number of mixtures (latent classes) to consider [def: 3]
* `p₀` :           Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
* `mixtures`:      An array (of length K) of the mixture to employ (see notes) [def: `[DiagonalGaussian() for i in 1:K]`]
* `tol`:           Tolerance to stop the algorithm [default: 10^(-6)]
* `verbosity`:     A verbosity parameter regulating the information messages frequency [def: `STD`]
* `minVariance`:   Minimum variance for the mixtures [default: 0.05]
* `minCovariance`: Minimum covariance for the mixtures with full covariance matrix [default: 0]. This should be set different than minVariance (see notes).
* `initStrategy`:  Mixture initialisation algorithm [def: `grid`]
* `maxIter`:       Maximum number of iterations [def: `-1`, i.e. ∞]
* `rng`:           Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A named touple of:
  * `̂X̂`    : The Filled Matrix of size (N x D)
  * `nFill`: The number of items filled
  * `lL`   : The log-likelihood (without considering the last mixture optimisation)
  * `BIC` :  The Bayesian Information Criterion (lower is better)
  * `AIC` :  The Akaike Information Criterion (lower is better)

  # Notes:
  - The mixtures currently implemented are `SphericalGaussian(μ,σ²)`,`DiagonalGaussian(μ,σ²)` and `FullGaussian(μ,σ²)`
  - For `initStrategy`, look at the documentation of `initMixtures!` for the mixture you want. The provided gaussian mixtures support `grid`, `kmeans` or `given`. `grid` is faster, but `kmeans` often provides better results.
  - The algorithm requires to specify a number of "latent classes" (mlixtures) to divide the dataset into. If there isn't any prior domain specific knowledge on this point one can test sevaral `k` and verify which one minimise the `BIC` or `AIC` criteria.


# Example:
```julia
julia>  cFOut = predictMissing([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3)
```
"""
function predictMissing(X,K=3;p₀=nothing,mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minVariance=0.05,minCovariance=0.0,initStrategy="kmeans",maxIter=-1,rng = Random.GLOBAL_RNG)
    if verbosity > STD
        @codeLocation
    end
    emOut = gmm(X,K;p₀=p₀,mixtures=mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,maxIter=maxIter,rng=rng)
    (N,D) = size(X)
    nDim  = ndims(X)
    nmT   = nonmissingtype(eltype(X))
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
    X̂ = convert(Array{nmT,nDim},X̂)
    return (X̂=X̂,nFill=nFill,lL=emOut.lL,BIC=emOut.BIC,AIC=emOut.AIC)
end


# MLJ interface
include("Clustering_MLJ.jl")

end
