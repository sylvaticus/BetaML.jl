using LinearAlgebra
using Random
using Distributions

""" Sterling number: number of partitions of a set of n elements in k sets """
sterling(n::BigInt,k::BigInt) = (1/factorial(k)) * sum((-1)^i * binomial(k,i)* (k-i)^n for i in 0:k)
sterling(n::Int64,k::Int64) = sterling(BigInt(n),BigInt(k))

# Basic K-Means Algorithm (Lecture/segment 13.7 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

"""
  kmean(X,K)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted

# Returns:
* A vector of size n of ids of the clusters associated to each point

# Notes:
* Some returned clusters could be empty

# Example:
```julia
julia> clIdx = kmean([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2)
```
"""
function kmean(X,K)
    (n,d) = size(X)
    # Random choice of initial representative vectors (any point, not just in X!)
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    Z₀ = zeros(K,d)
    for i in 1:K
        for j in 1:d
            Z₀[i,j] = rand(Uniform(minX[j],maxX[j]))
        end
    end
    Z = Z₀
    cIdx_prev = zeros(Int64,n)

    # Looping
    while true
        # Determining the constituency of each cluster
        cIdx      = zeros(Int64,n)
        for (i,x) in enumerate(eachrow(X))
            cost = Inf
            for (j,z) in enumerate(eachrow(Z))
               if (norm(x-z)^2  < cost)
                   cost    =  norm(x-z)^2
                   cIdx[i] = j
               end
            end
        end

        # Checking termination condition: clusters didn't move any more
        if cIdx == cIdx_prev
            return cIdx
        else
            cIdx_prev = cIdx
        end

        # Determining the new representative by each cluster
        for (j,z) in enumerate(eachrow(Z))
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            z = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
        end
    end
end

# Basic K-Medoids Algorithm (Lecture/segment 14.3 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)


"""Square Euclidean distance"""
square_euclidean(x,y) = norm(x-y)^2

"""Cosine distance"""
cos_distance(x,y) = dot(x,y)/(norm(x)*norm(y))


"""
  kmedoids(X,K,dist)

Compute K-Medoids algorithm to identify K clusters of X using distance definition `dist`

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (must accept two vectors). Default to squared Euclidean.

# Returns:
* A vector of size n of ids of the clusters associated to each point

# Notes:
* Some returned clusters could be empty

# Example:
```julia
julia> clIdx = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,dist = (x,y) -> norm(x-y)^2)
```
"""
function kmedoids(X,K;dist=(x,y) -> norm(x-y)^2)
    (n,d) = size(X)
    # Random choice of initial representative vectors
    zIdx = shuffle(1:size(X)[1])[1:K]
    Z₀ = X[zIdx, :]
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

        # Checking termination condition: clusters didn't move any more
        if cIdx == cIdx_prev
            return cIdx
        else
            cIdx_prev = cIdx
        end

        # Determining the new representative by each cluster (within the points member)
        for (j,z) in enumerate(eachrow(Z))
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            nⱼ = size(Cⱼ)[1]     # Size of the cluster
            println(nⱼ)
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
            z = reshape(Cⱼ[bestCIdx,:],1,d)
        end
    end
end

