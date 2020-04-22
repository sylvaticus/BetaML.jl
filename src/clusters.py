
# Temptative Python porting (on the way)

using LinearAlgebra
using Random
using Distributions

""" Sterling number: number of partitions of a set of n elements in k sets """
sterling(n::BigInt,k::BigInt) = (1/factorial(k)) * sum((-1)^i * binomial(k,i)* (k-i)^n for i in 0:k)
sterling(n::Int64,k::Int64) = sterling(BigInt(n),BigInt(k))

# Some common distances
"""L1 norm distance (aka "Manhattan Distance")"""
l1_distance(x,y) = sum(abs.(x-y))
"""Euclidean (L2) distance"""
l2_distance(x,y) = norm(x-y)
"""Squared Euclidean (L2) distance"""
l2²_distance(x,y) = norm(x-y)^2
"""Cosine distance"""
cosine_distance(x,y) = dot(x,y)/(norm(x)*norm(y))

"""
  make_matrix(x)

Transform an Array{T,1} in an Array{T,2} and leave unchanged Array{T,2}.


"""
make_matrix(x::Array) = ndims(x) == 1 ? reshape(x, (size(x)...,1)) : x


"""
  initRepresentatives(X,K;initStrategy,Z₀))

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
    X  = make_matrix(X)
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
        Z₀ = make_matrix(Z₀)
        Z = Z₀
    elseif initStrategy == "shuffle"
        zIdx = shuffle(1:size(X)[1])[1:K]
        Z = X[zIdx, :]
    else
        error("initStrategy \"$initStrategy\" not implemented")
    end
    return Z
end


# Basic K-Means Algorithm (Lecture/segment 13.7 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

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
    X  = make_matrix(X)
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

# Basic K-Medoids Algorithm (Lecture/segment 14.3 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)
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
    X  = make_matrix(X)
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


# The EM algorithm (Lecture/segment 16.5 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

""" PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
normalFixedSd(x,μ,σ²) = (1/(2π*σ²)^(length(x)/2)) * exp(-1/(2σ²)*norm(x-μ)^2)

# 16.5 The E-M Algorithm
"""
  em(X,K;p₀,μ₀,σ²₀,tol,msgStep)

Compute Expectation-Maximisation algorithm to identify K clusters of X data assuming a Gaussian Mixture probabilistic Model.

# Parameters:
* `X`  :      A (n x d) data to clusterise
* `K`  :      Number of cluster wanted
* `p₀` :      Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
* `μ₀` :      Initial means (K x d) of the Gaussian [default: `nothing`]
* `σ²₀`:      Initial variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions [default: `nothing`]
* `tol`:      Initial tolerance to stop the algorithm [default: 0.0001]
* `msgStep` : Iterations between update messages. Use 0 for no updates [default: 10]

# Returns:
* A named touple of:
  * `pⱼₓ`: Matrix of size (N x K) of the probabilities of each point i to belong to cluster j
  * `pⱼ`  : Probabilities of the categorical distribution (K x 1)
  * `μ`  : Means (K x d) of the Gaussian
  * `σ²` : Variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions
  * `ϵ`  : Vector of the discrepancy (matrix norm) between pⱼₓ and the lagged pⱼₓ at each iteration

# Example:
```julia
julia> clusters = em([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,msgStep=3)
```
"""
function em(X,K;p₀=nothing,μ₀=nothing,σ²₀=nothing,tol=0.0001,msgStep=10)
    # debug:
    #X = [1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4]
    #K = 3
    #p₀=nothing; μ₀=nothing; σ²₀=nothing; tol=0.0001
    X     = make_matrix(X)
    (N,D) = size(X)

    # Initialisation of the parameters if not provided
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    varX = mean(var(X,dims=1))/K^2
    pⱼ = isnothing(p₀) ? fill(1/K,K) : p₀
    if !isnothing(μ₀)
        μ₀  = make_matrix(μ₀)
        μ = μ₀
    else
        μ = zeros(Float64,K,D)
        for d in 1:D
                μ[:,d] = collect(range(minX[d], stop=maxX[d], length=K))
        end
    end
    σ² = isnothing(σ²₀) ? fill(varX,K) : σ²₀
    pⱼₓ = zeros(Float64,N,K)

    ϵ = Float64[]
    while(true)
        # E Step: assigning the posterior prob p(j|xi)
        pⱼₓlagged = copy(pⱼₓ)
        for n in 1:N
            px = sum([pⱼ[j]*normalFixedSd(X[n,:],μ[j,:],σ²[j]) for j in 1:K])
            for k in 1:K
                pⱼₓ[n,k] = pⱼ[k]*normalFixedSd(X[n,:],μ[k,:],σ²[k])/px
            end
        end

        # Compute the log-Likelihood of the parameters given the set of data
        # Just for informaticve purposes, not needed for the algorithm
        lL = 0
        for n in 1:N
            lL += log(sum([pⱼ[j]*normalFixedSd(X[n,:],μ[j,:],σ²[j]) for j in 1:K]))
        end

        if msgStep != 0 && (length(ϵ) % msgStep == 0 || length(ϵ) == 1)
           println("Log likelihood on iter. $(length(ϵ))\t: $(lL)")
        end

        # M step: find parameters that maximise the likelihood
        nⱼ = sum(pⱼₓ,dims=1)'
        n  = sum(nⱼ)
        pⱼ = nⱼ ./ n
        μ  = (pⱼₓ' * X) ./ nⱼ
        σ² = [sum([pⱼₓ[n,j] * norm(X[n,:]-μ[j,:])^2 for n in 1:N]) for j in 1:K ] ./ (nⱼ .* D)

        push!(ϵ,norm(pⱼₓlagged - pⱼₓ))

        if msgStep != 0 && (length(ϵ) % msgStep == 0 || length(ϵ) == 1)
           println("Iter. $(length(ϵ))\t: $(ϵ[end])")
        end
        if (ϵ[end] < tol)
            return (pⱼₓ=pⱼₓ,pⱼ=pⱼ,μ=μ,σ²=σ²,ϵ=ϵ)
        end
    end
end
