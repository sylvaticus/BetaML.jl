# MLJ interface for imputers models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here

export MissingImputator, BetaMLMeanImputer,BetaMLGMMImputer, BetaMLRFImputer

# ------------------------------------------------------------------------------
# Model Structure declarations..

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

mutable struct BetaMLMeanImputer <: MMI.Unsupervised
    normaliseRecords::Bool
end
BetaMLMeanImputer(;
    normaliseRecords = false,
) = BetaMLMeanImputer(normaliseRecords)

mutable struct BetaMLGMMImputer <: MMI.Unsupervised
    K::Int64
    p₀::Union{Nothing,AbstractArray{Float64,1}}
    mixtures::Symbol
    tol::Float64
    minVariance::Float64
    minCovariance::Float64
    initStrategy::String
    rng::AbstractRNG
end
BetaMLGMMImputer(;
    K             = 3,
    p₀            = Int64[],
    mixtures      = :diag_gaussian,
    tol           = 10^(-6),
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
) = BetaMLGMMImputer(K,p₀,mixtures, tol, minVariance, minCovariance,initStrategy,rng)

mutable struct BetaMLRFImputer <: MMI.Unsupervised
    nTrees::Int64
    maxDepth::Int64
    minGain::Float64
    minRecords::Int64
    maxFeatures::Int64
    forcedCategoricalCols::Vector{Int64}
    splittingCriterion::Function
    verbosity::Verbosity
    recursivePassages::Int64                  
    multipleImputations::Int64 
    rng::AbstractRNG
end
BetaMLRFImputer(;
    nTrees                 = 30, 
    maxDepth               = typemax(Int64),
    minGain                = 0.0,
    minRecords             = 2,
    maxFeatures            = typemax(Int64),
    forcedCategoricalCols  = Int64[],
    splittingCriterion     = nothing,
    verbosity              = STD,
    recursivePassages      = 1,
    multipleImputations    = 1,
    rng                    = Random.GLOBAL_RNG,
) = BetaMLRFImputer(nTrees, maxDepth, minGain, minRecords, maxFeatures, forcedCategoricalCols, splittingCriterion, verbosity, recursivePassages, multipleImputations, rng)

# ------------------------------------------------------------------------------
# Fit functions...

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

function MMI.fit(m::BetaMLMeanImputer, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    mod = MeanImputer(
        normaliseRecords = m.normaliseRecords,
    )
    fit!(mod,x)
    fitResults = MMI.table(predict(mod))
    cache      = nothing
    report     = info(mod)
    return (fitResults, cache, report)
end

function MMI.fit(m::BetaMLGMMImputer, verbosity, X)
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

    mod = GMMImputer(
        K             = m.K,
        p₀            = m.p₀,
        mixtures      = mixtures,
        tol           = m.tol,
        minVariance   = m.minVariance,
        minCovariance = m.minCovariance,
        initStrategy  = m.initStrategy,
        verbosity     = NONE,
        rng           = m.rng
    )
    fit!(mod,x)
    fitResults = MMI.table(predict(mod))
    cache      = nothing
    report     = info(mod)

    return (fitResults, cache, report)  
end

function MMI.fit(m::BetaMLRFImputer, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    mod = RFImputer(
        nTrees                 = m.nTrees, 
        maxDepth               = m.maxDepth,
        minGain                = m.minGain,
        minRecords             = m.minRecords,
        maxFeatures            = m.maxFeatures,
        forcedCategoricalCols  = m.forcedCategoricalCols,
        splittingCriterion     = m.splittingCriterion,
        verbosity              = m.verbosity,
        recursivePassages      = m.recursivePassages,
        multipleImputations    = m.multipleImputations,
        rng                    = m.rng,
    )
    fit!(mod,x)
    if m.multipleImputations == 1
        fitResults = MMI.table(predict(mod))
    else
        fitResults = MMI.table.(predict(mod))
    end
    cache      = nothing
    report     = info(mod)
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
# Model metadata for registration in MLJ...

MMI.metadata_model(MissingImputator,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using an Expectation-Maximisation clustering algorithm, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Imputation.MissingImputator"
)

MMI.metadata_model(BetaMLMeanImputer,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using feature (column) mean, with optional record normalisation (using l-1 norms), from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaMLMeanImputer"
)

MMI.metadata_model(BetaMLGMMImputer,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using an Gaussian Mixture Models clustering, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaMLGMMImputer"
)
MMI.metadata_model(BetaMLRFImputer,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    output_scitype   = MMI.Table(MMI.Known),          # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using Random Forests, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaMLRFImputer"
)