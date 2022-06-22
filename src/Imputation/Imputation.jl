"""
    Imputation.jl file

Implement the BetaML.Imputation module

`?BetaML.Imputation` for documentation

- Go to [https://sylvaticus.github.io/BetaML.jl](https://sylvaticus.github.io/BetaML.jl/stable) for more general doc

"""

"""
    Imputation module

Provide various imputation methods for missing data. Note that the interpretation of "missing" can be very wide. Reccomendation systems / collaborative filtering (e.g. suggestion of the film to watch) may well be representated as missing data to impute.

Imputers that include a random component have normally a parameter to allow multiple imputation. Up to you then to use this information in the rest of your workflow. 

- [`MeanImputer`](@ref): Simple imputator using the feature or the recors (or both) means
- [`GMMImputer`](@ref): Impute data using a Generative (Gaussian) Mixture Model
- [`RFImputer`](@ref): Impute missing data using Random Forests

Imputations for all these models can be optained by running `impute([Imputator model],X)`. The function returns an `ImputerResult` that can be queried with `imputed(x::ImputerResult)`, `imputedValues(x::ImputerResult)` (for multiple imputations) and `info(x::ImputerResult)` to query further informations concerning the imputation.
"""
module Imputation

using Statistics, Random
using ForceImport
@force using ..Api
@force using ..Utils
@force using ..Clustering
@force using ..Trees

export Imputer, MeanImputer, GMMImputer, RFImputer,
       ImputerResult, MeanImputerResult, GMMImputerResult, RFImputerResult, 
       impute, imputed, imputedValues, info

abstract type Imputer end   
abstract type ImputerResult end

# ------------------------------------------------------------------------------
# MeanImputer
Base.@kwdef mutable struct MeanImputer <: Imputer
    recordCorrection::Float64 = 0.0
    meanIterations::Int64     = 1
end
struct MeanImputerResult <: ImputerResult
    imputed
    nImputedValues::Int64
end
function impute(imputer::MeanImputer,X)
    X̂ = copy(X)
    imp = imputer
    nR,nC = size(X)
    missingMask = ismissing.(X)
    for k in 1:imp.meanIterations
        cMeans    = [mean(skipmissing(X̂[:,i])) for i in 1:nC]
        rMeans    = [mean(skipmissing(X̂[i,:])) for i in 1:nR]
        X̂ = [missingMask[r,c] ? cMeans[c]*(1-imp.recordCorrection) + rMeans[r]*imp.recordCorrection : X̂[r,c] for r in 1:nR, c in 1:nC]
    end
    return MeanImputerResult(X̂,sum(missingMask))
end
imputed(r::MeanImputerResult) = r.imputed
imputedValues(r::MeanImputerResult) = [r.imputed]
info(r::MeanImputerResult) = (nImputedValues = r.nImputedValues,)

# ------------------------------------------------------------------------------
# GMMImputer
"""
    GMMImputer

Impute using Generated (Gaussian) mixture models.
Limitations:
- data must be numerical
- the resulted matrix is a Matrix{Float64}
- currently the Mixtures available do not support random initialisation, so there is no random component involved (i.e. no multiple imputations)    
"""
Base.@kwdef mutable struct GMMImputer <: Imputer
    K::Int64                           = 3
    p₀::Union{Nothing,Vector{Float64}} = nothing
    mixtures::Vector{AbstractMixture}  = [DiagonalGaussian() for i in 1:K]
    tol::Float64                       = 10^(-6)
    verbosity::Verbosity               = STD
    minVariance::Float64               = 0.05
    minCovariance::Float64             = 0.0
    initStrategy::String               = "kmeans"
    maxIter::Int64                     = -1
    multipleImputations::Int64         = 1
    rng::AbstractRNG                   = Random.GLOBAL_RNG
end

struct GMMImputerResult <: ImputerResult
    imputed
    nImputedValues::Int64
    lL::Vector{Float64}
    BIC::Vector{Float64}
    AIC::Vector{Float64}
end

function impute(imputer::GMMImputer,X)
    imp = imputer
    if imp.verbosity > STD
        @codeLocation
    end
    (N,D) = size(X)
    nDim  = ndims(X)
    nmT   = nonmissingtype(eltype(X))
    #K = size(emOut.μ)[1]
    XMask = .! ismissing.(X)
    nFill = (N * D) - sum(XMask)

    imputedValues = Array{Float64,nDim}[]
    nImputedValues = nFill
    lLs  = Float64[]
    BICs = Float64[]
    AICs = Float64[]

    for mi in 1:imp.multipleImputations
        emOut = gmm(X,imp.K;p₀=imp.p₀,mixtures=imp.mixtures,tol=imp.tol,verbosity=imp.verbosity,minVariance=imp.minVariance,minCovariance=imp.minCovariance,initStrategy=imp.initStrategy,maxIter=imp.maxIter,rng=imp.rng)
        #=
        X̂ = copy(X)
        for n in 1:N
            for d in 1:D
                if !XMask[n,d]
                    X̂[n,d] = sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:imp.K])
                end
            end
        end
        =#
        X̂ = [XMask[n,d] ? X[n,d] : sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:imp.K]) for n in 1:N, d in 1:D ]
        #X̂ = identity.(X̂)
        #X̂ = convert(Array{nmT,nDim},X̂)
        push!(imputedValues,X̂)
        push!(lLs,emOut.lL)
        push!(BICs,emOut.BIC)
        push!(AICs,emOut.AIC)
    end
    return GMMImputerResult(imputedValues,nImputedValues,lLs,BICs,AICs)
end

imputed(r::GMMImputerResult) = mean(r.imputed)
imputedValues(r::GMMImputerResult) = r.imputed
info(r::GMMImputerResult) = (nImputedValues = r.nImputedValues, lL=r.lL, BIC=r.BIC, AIC=r.AIC)

# ------------------------------------------------------------------------------
# RFImputer

Base.@kwdef mutable struct RFImputer <: Imputer
    nTrees::Int64                               = 30
    maxDepth::Int64                             = typemax(Int64)
    minGain::Float64                            = 0.0
    minRecords::Int64                           = 2
    maxFeatures::Int64                          = typemax(Int64)
    forcedCategoricalCols::Vector{Int64}        = Int64[] # like in RF, normally integers are considered ordinal
    splittingCriterion::Union{Function,Nothing} = nothing
    β::Float64                                  = 0.0
    oob::Bool                                   = false
    recursivePassages                           = 1
    multipleImputations::Int64                  = 1
    rng::AbstractRNG                            = Random.GLOBAL_RNG
end

struct RFImputerResult <: ImputerResult
    imputed
    nImputedValues::Int64
    oob::Vector{Float64}
end



function impute(imputer::RFImputer,X)
    nR,nC = size(X)
    if imputer.maxFeatures == typemax(Int64) && imputer.nTrees >1
      maxFeatures = Int(round(sqrt(size(X,2))))
    end
    maxFeatures   = min(nC,imputer.maxFeatures) 
    maxDepth      = min(nR,imputer.maxDepth)

    catCols = [! (nonmissingtype(eltype(identity.(X[:,c]))) <: Number ) || c in imputer.forcedCategoricalCols for c in 1:nC]

    forestModels  = Array{Forest,1}(undef,nC)
    missingMask = ismissing.(X) 
    sortedDims = reverse(sortperm(makeColVector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values

    return sortedDims
end

imputed(r::RFImputerResult) = r.imputed[1]
imputedValues(r::RFImputerResult) = r.imputed
info(r::RFImputerResult) = nothing

end # end Imputation module