# MLJ interface for imputers models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here

#export BetaMLMeanImputer,
export BetaMLGMMImputer
#BetaMLRFImputer

# ------------------------------------------------------------------------------
# Model Structure declarations..


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
    p₀            = nothing,
    mixtures      = :diag_gaussian,
    tol           = 10^(-6),
    minVariance   = 0.05,
    minCovariance = 0.0,
    initStrategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
) = BetaMLGMMImputer(K,p₀,mixtures, tol, minVariance, minCovariance,initStrategy,rng)


# ------------------------------------------------------------------------------
# Fit functions...

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



# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(BetaMLGMMImputer,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
    descr            = "Impute missing values using an Gaussian Mixture Models clustering, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaMLGMMImputer"
)
