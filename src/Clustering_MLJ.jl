# MLJ interface for clustering models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export KMeans

# ------------------------------------------------------------------------------
# Model Structure declarations..

mutable struct KMeans <: MMI.Unsupervised
   K::Int64
   dist::Function
   initStrategy::String
   Z₀::Union{Nothing,Matrix{Float64}}
end
KMeans(;
   K            = 3,
   dist         = dist=(x,y) -> norm(x-y),
   initStrategy = "grid",
   Z₀           = nothing
   ) = KMeans(K,dist,initStrategy,Z₀)

# ------------------------------------------------------------------------------
# Fit functions...
function MMI.fit(m::KMeans, verbosity, X)
    x                = MMI.matrix(X)                        # convert table to matrix
    (assignedClasses,representatives) = kmeans(x,m.K,dist=m.dist,initStrategy=m.initStrategy,Z₀=m.Z₀)
    cache=nothing
    report=nothing
    return ((assignedClasses,representatives,m.dist), cache, report)
end

# ------------------------------------------------------------------------------
# Transform functions...

""" fit(m::KMeans, fitResults, X) - Given a trained clustering model and some observations, return the distances to each centroids """
function MMI.transform(m::KMeans, fitResults, X)
    x                = MMI.matrix(X) # convert table to matrix
    (N,D) = size(x)
    nCl = size(fitResults[2],1)
    distances = Array{Float64,2}(undef,N,nCl)

    for n in 1:N
        for c in 1:nCl
            distances[n,c] = fitResults[3](x[n,:],fitResults[2][c,:])
        end
    end

    return MMI.table(distances)
end

""" predict(m::KMeans, fitResults, X) - Given a trained clustering model and some observations, predict the class of the observation"""
function MMI.predict(m::KMeans, fitResults, X)
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    nCl             = size(fitResults[2],1)
    distances       = MMI.matrix(MMI.transform(m, fitResults, X))
    mindist         = argmin(distances,dims=2)
    assignedClasses = [Tuple(mindist[n,1])[2]  for n in 1:N]
    return CategoricalArray(assignedClasses)
end
