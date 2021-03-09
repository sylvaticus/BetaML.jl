# MLJ interface for clustering models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export KMeans, KMedoids, GMM

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

 mutable struct KMedoids <: MMI.Unsupervised
    K::Int64
    dist::Function
    initStrategy::String
    Z₀::Union{Nothing,Matrix{Float64}}
 end
 KMedoids(;
    K            = 3,
    dist         = dist=(x,y) -> norm(x-y),
    initStrategy = "grid",
    Z₀           = nothing
  ) = KMedoids(K,dist,initStrategy,Z₀)

# function gmm(X,K;p₀=nothing,mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minVariance=0.05,minCovariance=0.0,initStrategy="grid")
mutable struct GMM{TM <: AbstractMixture} <: MMI.Unsupervised
  K::Int64
  p₀::Union{Nothing,AbstractArray{Float64,1}}
  mixtures::AbstractArray{TM,1}
  minVariance::Float64
  minCovariance::Float64
  initStrategy::String
end
GMM(;
    K             = 3,
    p₀            = nothing,
    mixtures      = [DiagonalGaussian() for i in 1:K],
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
) = GMM(K,p₀,mixtures, minVariance, minCovariance,initStrategy)

# ------------------------------------------------------------------------------
# Fit functions...
function MMI.fit(m::Union{KMeans,KMedoids}, verbosity, X)
    x  = MMI.matrix(X)                        # convert table to matrix
    if typeof(m) == KMeans
        (assignedClasses,representatives) = kmeans(x,m.K,dist=m.dist,initStrategy=m.initStrategy,Z₀=m.Z₀)
    else
        (assignedClasses,representatives) = kmedoids(x,m.K,dist=m.dist,initStrategy=m.initStrategy,Z₀=m.Z₀)
    end
    cache=nothing
    report=nothing
    return ((assignedClasses,representatives,m.dist), cache, report)
end

function MMI.fit(m::GMM, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    res        = gmm(x,m.K,p₀=m.p₀,mixtures=m.mixtures, minVariance=m.minVariance, minCovariance=m.minCovariance,initStrategy=m.initStrategy,verbosity=NONE)
    fitResults = (res.pₙₖ,res.pₖ,res.mixtures)
    cache      = nothing
    report     = (res.ϵ,res.lL,res.BIC,res.AIC)
    return (fitResults, cache, report)
end

# ------------------------------------------------------------------------------
# Transform functions...

""" fit(m::KMeans, fitResults, X) - Given a trained clustering model and some observations, return the distances to each centroids """
function MMI.transform(m::Union{KMeans,KMedoids}, fitResults, X)
    x     = MMI.matrix(X) # convert table to matrix
    (N,D) = size(x)
    nCl   = size(fitResults[2],1)
    distances = Array{Float64,2}(undef,N,nCl)
    for n in 1:N
        for c in 1:nCl
            distances[n,c] = fitResults[3](x[n,:],fitResults[2][c,:])
        end
    end
    return MMI.table(distances)
end


""" predict(m::KMeans, fitResults, X) - Given a trained clustering model and some observations, predict the class of the observation"""
function MMI.predict(m::Union{KMeans,KMedoids}, fitResults, X)
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    nCl             = size(fitResults[2],1)
    distances       = MMI.matrix(MMI.transform(m, fitResults, X))
    mindist         = argmin(distances,dims=2)
    assignedClasses = [Tuple(mindist[n,1])[2]  for n in 1:N]
    return CategoricalArray(assignedClasses)
end

""" predict(m::GMM, fitResults, X) - Given a trained clustering model and some observations, predict the class of the observation"""
function MMI.predict(m::GMM, fitResults, X)
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    (pₙₖ,pₖ,mixtures) = fitResults
    nCl             = length(pₖ)
    prob            = Array{Float64,2}(undef,N,nCl)

    return prob
end

# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(KMeans,
    input_scitype    = MMI.Table(MMI.Continuous),
    output_scitype   = AbstractArray{<:MMI.Multiclass}, # for an unsupervised, what output?
    supports_weights = false,                           # does the model support sample weights?
    descr            = "The classical KMeans clustering algorithm, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.KMeans"
)

MMI.metadata_model(KMedoids,
    input_scitype    = MMI.Table(MMI.Continuous),
    output_scitype   = AbstractArray{<:MMI.Multiclass},     # for an unsupervised, what output?
    supports_weights = false,                               # does the model support sample weights?
    descr            = "The K-medoids clustering algorithm with customisable distance function, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.KMedoids"
)
