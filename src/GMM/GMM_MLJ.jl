# MLJ interface for clustering models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here

export  GMMClusterer, MissingImputator

# ------------------------------------------------------------------------------
# Model Structure declarations..


mutable struct GMMClusterer <: MMI.Unsupervised
  K::Int64
  p₀::AbstractArray{Float64,1}
  mixtures::Symbol
  tol::Float64
  minVariance::Float64
  minCovariance::Float64
  initStrategy::String
  rng::AbstractRNG
end
GMMClusterer(;
    K             = 3,
    p₀            = Float64[],
    mixtures      = :diag_gaussian,
    tol           = 10^(-6),
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
) = GMMClusterer(K,p₀,mixtures, tol, minVariance, minCovariance,initStrategy,rng)

mutable struct MissingImputator <: MMI.Unsupervised
    K::Int64
    p₀::AbstractArray{Float64,1}
    mixtures::Symbol
    tol::Float64
    minVariance::Float64
    minCovariance::Float64
    initStrategy::String
    rng::AbstractRNG
end
MissingImputator(;
    K             = 3,
    p₀            = Float64[],
    mixtures      = :diag_gaussian,
    tol           = 10^(-6),
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
) = MissingImputator(K,p₀,mixtures, tol, minVariance, minCovariance,initStrategy,rng)


# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(m::GMMClusterer, verbosity, X)
    # X is nothing, y is the data: https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Models-that-learn-a-probability-distribution-1
    x          = MMI.matrix(X) # convert table to matrix
    if m.mixtures == :diag_gaussian
        mixtures = [DiagonalGaussian() for i in 1:m.K]
    elseif m.mixtures == :full_gaussian
        mixtures = [FullGaussian() for i in 1:m.K]
    elseif m.mixtures == :spherical_gaussian
        mixtures = [SphericalGaussian() for i in 1:m.K]
    else
        error("Usupported mixture. Supported mixtures are either `:diag_gaussian`, `:full_gaussian` or `:spherical_gaussian`.")
    end
    res        = gmm(x,m.K,p₀=deepcopy(m.p₀),mixtures=mixtures, minVariance=m.minVariance, minCovariance=m.minCovariance,initStrategy=m.initStrategy,verbosity=NONE,rng=m.rng)
    fitResults = (pₖ=res.pₖ,mixtures=res.mixtures) # res.pₙₖ
    cache      = nothing
    report     = (res.ϵ,res.lL,res.BIC,res.AIC)
    return (fitResults, cache, report)
end
MMI.fitted_params(model::GMMClusterer, fitresult) = (weights=fitesult.pₖ, mixtures=fitresult.mixtures)


function MMI.fit(m::MissingImputator, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    if m.mixtures == :diag_gaussian
        mixtures = [DiagonalGaussian() for i in 1:m.K]
    elseif m.mixtures == :full_gaussian
        mixtures = [FullGaussian() for i in 1:m.K]
    elseif m.mixtures == :spherical_gaussian
        mixtures = [SphericalGaussian() for i in 1:m.K]
    else
        error("Usupported mixture. Supported mixtures are either `:diag_gaussian`, `:full_gaussian` or `:spherical_gaussian`.")
    end
    res        = gmm(x,m.K,p₀=deepcopy(m.p₀),mixtures=mixtures, minVariance=m.minVariance, minCovariance=m.minCovariance,initStrategy=m.initStrategy,verbosity=NONE,rng=m.rng)
    fitResults = (pₖ=res.pₖ,mixtures=res.mixtures) # pₙₖ=res.pₙₖ
    cache      = nothing
    report     = (res.ϵ,res.lL,res.BIC,res.AIC)
    return (fitResults, cache, report)
end



# ------------------------------------------------------------------------------
# Transform functions...

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




# ------------------------------------------------------------------------------
# Predict functions...

function MMI.predict(m::GMMClusterer, fitResults, X)
    x               = MMI.matrix(X) # convert table to matrix
    (N,D)           = size(x)
    (pₖ,mixtures)   = (fitResults.pₖ, fitResults.mixtures)
    nCl             = length(pₖ)
    # Compute the probabilities that maximise the likelihood given existing mistures and a single iteration (i.e. doesn't update the mixtures)
    thisOut         = gmm(x,nCl,p₀=pₖ,mixtures=mixtures,tol=m.tol,verbosity=NONE,minVariance=m.minVariance,minCovariance=m.minCovariance,initStrategy="given",maxIter=1,rng=m.rng)
    classes         = CategoricalArray(1:nCl)
    predictions     = MMI.UnivariateFinite(classes, thisOut.pₙₖ)
    return predictions
end


# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(GMMClusterer,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = AbstractArray{<:MMI.Multiclass},       # scitype of the output of `transform`
    target_scitype   = AbstractArray{<:MMI.Multiclass},       # scitype of the output of `predict`
    #prediction_type  = :probabilistic,  # option not added to metadata_model function, need to do it separately
    supports_weights = false,                                 # does the model support sample weights?
    descr            = "A Expectation-Maximisation clustering algorithm with customisable mixtures, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.GMMClusterer"
)
MMI.prediction_type(::Type{<:GMMClusterer}) = :probabilistic

MMI.metadata_model(MissingImputator,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using an Expectation-Maximisation clustering algorithm, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Clustering.MissingImputator"
)
