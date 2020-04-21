using LinearAlgebra
using Random
using Distributions

""" Sterling number: number of partitions of a set of n elements in k sets """
sterling(n::BigInt,k::BigInt) = (1/factorial(k)) * sum((-1)^i * binomial(k,i)* (k-i)^n for i in 0:k)
sterling(n::Int64,k::Int64) = sterling(BigInt(n),BigInt(k))

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
        for d in 1:D
                Z = Z₀
        end
    elseif initStrategy == "shuffle"
        for d in 1:D
            zIdx = shuffle(1:size(X)[1])[1:K]
            Z = X[zIdx, :]
        end
    else
        error("initStrategy \"$initStrategy\" not implemented")
    end
    return Z
end


# Basic K-Means Algorithm (Lecture/segment 13.7 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

"""
  kmean(X,K,initStrategy)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

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
* A tuple of two items, the first one being a vector of size N of ids of the clusters associated to each point and the second one the (K x D) matrix of representatives

# Notes:
* Some returned clusters could be empty

# Example:
```julia
julia> (clIdx,Z) = kmean([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2)
```
"""
function kmean(X,K;initStrategy="grid",Z₀=nothing)
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
               if (norm(x-z)^2  < cost)
                   cost    =  norm(x-z)^2
                   cIdx[i] = j
               end
            end
        end

        # Determining the new representative by each cluster
        #for (j,z) in enumerate(eachrow(Z))
        for j in  1:K
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            Z[j,:] = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
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

"""Square Euclidean distance"""
square_euclidean(x,y) = norm(x-y)^2

"""Cosine distance"""
cos_distance(x,y) = dot(x,y)/(norm(x)*norm(y))


"""
  kmedoids(X,K;dist,initStrategy,Z₀)

Compute K-Medoids algorithm to identify K clusters of X using distance definition `dist`

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (must accept two vectors). Default to squared Euclidean.
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

# Example:
```julia
julia> (clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38],2,dist = (x,y) -> norm(x-y)^2,initStrategy="grid")
```
"""
function kmedoids(X,K;dist=(x,y) -> norm(x-y)^2,initStrategy="shuffle",Z₀=nothing)
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

(clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38],2,dist = (x,y) -> norm(x-y)^2,initStrategy="grid")

# The EM algorithm (Lecture/segment 16.5 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)


""" PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
normalFixedSd(x,μ,σ²) = (1/(2π*σ²)^(length(x)/2)) * exp(-1/(2σ²)*norm(x-μ)^2)
