# MLJ interface for hard clustering models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here

export KMeans, KMedoids, KMeansModel, KMedoidsModel

# ------------------------------------------------------------------------------
# Model Structure declarations..

mutable struct KMeans <: MMI.Unsupervised
   K::Int64
   dist::Function
   initStrategy::String
   Z₀::Union{Nothing,Matrix{Float64}}
   rng::AbstractRNG
end
KMeans(;
   K            = 3,
   dist         = dist=(x,y) -> norm(x-y),
   initStrategy = "shuffle",
   Z₀           = nothing,
   rng          = Random.GLOBAL_RNG,
 ) = KMeans(K,dist,initStrategy,Z₀,rng)

 mutable struct KMedoids <: MMI.Unsupervised
    K::Int64
    dist::Function
    initStrategy::String
    Z₀::Union{Nothing,Matrix{Float64}}
    rng::AbstractRNG
 end
 KMedoids(;
    K            = 3,
    dist         = dist=(x,y) -> norm(x-y),
    initStrategy = "shuffle",
    Z₀           = nothing,
    rng          = Random.GLOBAL_RNG,
  ) = KMedoids(K,dist,initStrategy,Z₀,rng)

# ------------------------------------------------------------------------------
# Fit functions...
function MMI.fit(m::Union{KMeans,KMedoids}, verbosity, X)
    x  = MMI.matrix(X)                        # convert table to matrix
    # Using low level API here. We could switch to APIV2...
    if typeof(m) == KMeans
        (assignedClasses,representatives) = kmeans(x,m.K,dist=m.dist,initStrategy=m.initStrategy,Z₀=m.Z₀,rng=m.rng)
    else
        (assignedClasses,representatives) = kmedoids(x,m.K,dist=m.dist,initStrategy=m.initStrategy,Z₀=m.Z₀,rng=m.rng)
    end
    cache=nothing
    report=nothing
    return ((classes=assignedClasses,centers=representatives,distanceFunction=m.dist), cache, report)
end
MMI.fitted_params(model::Union{KMeans,KMedoids}, fitresult) = (centers=fitesult[2], cluster_labels=CategoricalArrays.categorical(fitresults[1]))

# ------------------------------------------------------------------------------
# Transform functions...

""" fit(m::KMeans, fitResults, X) - Given a fitted clustering model and some observations, return the distances to each centroids """
function MMI.transform(m::Union{KMeans,KMedoids}, fitResults, X)
    x     = MMI.matrix(X) # convert table to matrix
    (N,D) = size(x)
    nCl   = size(fitResults.centers,1)
    distances = Array{Float64,2}(undef,N,nCl)
    for n in 1:N
        for c in 1:nCl
            distances[n,c] = fitResults.distanceFunction(x[n,:],fitResults[2][c,:])
        end
    end
    return MMI.table(distances)
end

# ------------------------------------------------------------------------------
# Predict functions...

""" predict(m::KMeans, fitResults, X) - Given a fitted clustering model and some observations, predict the class of the observation"""
function MMI.predict(m::Union{KMeans,KMedoids}, fitResults, X)
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    nCl             = size(fitResults.centers,1)
    distances       = MMI.matrix(MMI.transform(m, fitResults, X))
    mindist         = argmin(distances,dims=2)
    assignedClasses = [Tuple(mindist[n,1])[2]  for n in 1:N]
    return CategoricalArray(assignedClasses,levels=1:nCl)
end

# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(KMeans,
    input_scitype    = MMI.Table(MMI.Continuous),         # scitype of the inputs
    output_scitype   = MMI.Table(MMI.Continuous),         # scitype of the output of `transform`
    target_scitype   = AbstractArray{<:MMI.Multiclass},   # scitype of the output of `predict`
    supports_weights = false,                             # does the model support sample weights?
    descr            = "The classical KMeans clustering algorithm, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.KMeans"
)

MMI.metadata_model(KMedoids,
    input_scitype    = MMI.Table(MMI.Continuous),         # scitype of the inputs
    output_scitype   = MMI.Table(MMI.Continuous),         # scitype of the output of `transform`
    target_scitype   = AbstractArray{<:MMI.Multiclass},   # scitype of the output of `predict`
    supports_weights = false,                             # does the model support sample weights?
    descr            = "The K-medoids clustering algorithm with customisable distance function, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.KMedoids"
)