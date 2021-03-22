# MLJ interface for clustering models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export KMeans, KMedoids, GMM, MissingImputator

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

# function gmm(X,K;p₀=nothing,mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minVariance=0.05,minCovariance=0.0,initStrategy="grid")
mutable struct GMM{TM <: AbstractMixture} <: MMI.Probabilistic
  K::Int64
  p₀::Union{Nothing,AbstractArray{Float64,1}}
  mixtures::AbstractArray{TM,1}
  tol::Float64
  minVariance::Float64
  minCovariance::Float64
  initStrategy::String
  rng::AbstractRNG
end
GMM(;
    K             = 3,
    p₀            = nothing,
    mixtures      = [DiagonalGaussian() for i in 1:K],
    tol           = 10^(-6),
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
) = GMM(K,p₀,mixtures, tol, minVariance, minCovariance,initStrategy,rng)

mutable struct MissingImputator{TM <: AbstractMixture} <: MMI.Unsupervised
    K::Int64
    p₀::Union{Nothing,AbstractArray{Float64,1}}
    mixtures::AbstractArray{TM,1}
    tol::Float64
    minVariance::Float64
    minCovariance::Float64
    initStrategy::String
    rng::AbstractRNG
end
MissingImputator(;
    K             = 3,
    p₀            = nothing,
    mixtures      = [DiagonalGaussian() for i in 1:K],
    tol           = 10^(-6),
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
) = MissingImputator(K,p₀,mixtures, tol, minVariance, minCovariance,initStrategy,rng)


# ------------------------------------------------------------------------------
# Fit functions...
function MMI.fit(m::Union{KMeans,KMedoids}, verbosity, X)
    x  = MMI.matrix(X)                        # convert table to matrix
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


function MMI.fit(m::GMM, verbosity, X, y)
    # X is nothing, y is the data: https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Models-that-learn-a-probability-distribution-1
    y          = MMI.matrix(y) # convert table to matrix
    res        = gmm(y,m.K,p₀=m.p₀,mixtures=m.mixtures, minVariance=m.minVariance, minCovariance=m.minCovariance,initStrategy=m.initStrategy,verbosity=NONE,rng=m.rng)
    fitResults = (pₙₖ=res.pₙₖ,pₖ=res.pₖ,mixtures=res.mixtures) # res.pₙₖ
    cache      = nothing
    report     = (res.ϵ,res.lL,res.BIC,res.AIC)
    return (fitResults, cache, report)
end

function MMI.fit(m::MissingImputator, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    res        = gmm(x,m.K,p₀=m.p₀,mixtures=m.mixtures, minVariance=m.minVariance, minCovariance=m.minCovariance,initStrategy=m.initStrategy,verbosity=NONE,rng=m.rng)
    fitResults = (pₖ=res.pₖ,mixtures=res.mixtures) # pₙₖ=res.pₙₖ
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
    nCl   = size(fitResults.centers,1)
    distances = Array{Float64,2}(undef,N,nCl)
    for n in 1:N
        for c in 1:nCl
            distances[n,c] = fitResults.distanceFunction(x[n,:],fitResults[2][c,:])
        end
    end
    return MMI.table(distances)
end


""" predict(m::KMeans, fitResults, X) - Given a trained clustering model and some observations, predict the class of the observation"""
function MMI.predict(m::Union{KMeans,KMedoids}, fitResults, X)
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    nCl             = size(fitResults.centers,1)
    distances       = MMI.matrix(MMI.transform(m, fitResults, X))
    mindist         = argmin(distances,dims=2)
    assignedClasses = [Tuple(mindist[n,1])[2]  for n in 1:N]
    return CategoricalArray(assignedClasses,levels=1:nCl)
end

""" predict(m::GMM, fitResults, X) - Given a trained clustering model, predict the class of the observations used for training"""
function MMI.predict(m::GMM, fitResults, X)
    #=
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    (pₖ,mixtures)   = (fitResults.pₖ, fitResults.mixtures)
    nCl             = length(pₖ)
    # Compute the probabilities that maximise the likelihood given existing mistures and a single iteration (i.e. doesn't update the mixtures)
    thisOut         = gmm(x,nCl,p₀=pₖ,mixtures=mixtures,tol=m.tol,verbosity=NONE,minVariance=m.minVariance,minCovariance=m.minCovariance,initStrategy="given",maxIter=1,rng=m.rng)
    classes         = CategoricalArray(1:nCl)
    predictions     = MMI.UnivariateFinite(classes, thisOut.pₙₖ)
    =#
    (pₙₖ, pₖ)       = (fitResults.pₙₖ, pₖ=fitResults.pₖ)
    nCl             = length(pₖ)
    classes         = CategoricalArray(1:nCl)
    predictions     = MMI.UnivariateFinite(classes,pₙₖ)

    return predictions
end

""" transform(m::MissingImputator, fitResults, X) - Given a trained imputator model fill the missing data of some new observations"""
function MMI.transform(m::MissingImputator, fitResults, X)
    x             = MMI.matrix(X) # convert table to matrix
    (N,D)         = size(x)
    (pₖ,mixtures) = fitResults.pₖ, fitResults.mixtures   #
    nCl           = length(pₖ)
    # Fill the missing data of this "new X" using the mixtures computed in the fit stage
    xout          = predictMissing(x,nCl,p₀=pₖ,mixtures=mixtures,tol=m.tol,verbosity=NONE,minVariance=m.minVariance,minCovariance=m.minCovariance,initStrategy="given",maxIter=1,rng=m.rng)
    return MMI.table(xout.X̂)
end

""" transform(m::GMM, fitResults, X) - Given a trained clustering model and some observations, predict the class of the observation"""
function MMI.transform(m::GMM, fitResults, X)
    return MMI.predict(m::GMM, fitResults, X)
end

#""" transform(m::MissingImputator, X) - Given a matrix with missing value, impute them using an EM algorithm"""
#function MMI.transform(m::MissingImputator, X)
#    x    = MMI.matrix(X) # convert table to matrix
#    xout = predictMissing(x,m.K;p₀=m.p₀,mixtures=m.mixtures,tol=m.tol,verbosity=NONE,minVariance=m.minVariance,minCovariance=m.minCovariance,initStrategy=m.initStrategy)
#    return MMI.table(xout.X̂)
#end

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

MMI.metadata_model(GMM,
    input_scitype    = Nothing, # MMI.Table(MMI.Continuous,MMI.Missing),
    target_scitype   = MMI.Table(MMI.Continuous,MMI.Missing), #AbstractArray{<:MMI.Multiclass},
    supports_weights = false,                               # does the model support sample weights?
    descr            = "A Expectation-Maximisation clustering algorithm with customisable mixtures, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.GMM"
)

MMI.metadata_model(MissingImputator,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using an Expectation-Maximisation clustering algorithm, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.MissingImputator"
)
