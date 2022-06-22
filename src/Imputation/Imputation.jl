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


using ForceImport, Statistics
@force using ..Api
@force using ..Utils
@force using ..Clustering

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
function impute(X,imputer::MeanImputer)
    i = imputer
    nR,nC = size(X)
    missingMask = ismissing.(X)
    for k in 1:i.meanIterations
        cMeans    = [mean(skipmissing(X[:,i])) for i in 1:nC]
        rMeans    = [mean(skipmissing(X[i,:])) for i in 1:nR]
        [X[r,c] = cMeans[c]*(1-i.recordCorrection) + rMeans[r]*i.recordCorrection for c in 1:nC, r in 1:nR if missingMask[r,c] ]
    end
    return MeanImputerResult(X,sum(missingMask))
end
imputed(r::MeanImputerResult) = r.imputed
imputedValues(r::MeanImputerResult) = [r.imputed]
info(r::MeanImputerResult) = (nImputedValues = r.nImputedValues)

# ------------------------------------------------------------------------------
# GMMImputer
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
    rng::Random._GLOBAL_RNG            = Random.GLOBAL_RNG
end

struct GMMImputerResult <: ImputerResult
    imputed
    nImputedValues::Int64
    lL::Vector{Float64}
    BIC::Vector{Float64}
    AIC::Vector{Float64}
end

function impute(X,imputer::GMMImputer)
    i = imputer
    if i.verbosity > STD
        @codeLocation
    end
    (N,D) = size(X)
    nDim  = ndims(X)
    nmT   = nonmissingtype(eltype(X))
    #K = size(emOut.μ)[1]
    XMask = .! ismissing.(X)
    nFill = (N * D) - sum(XMask)

    imputedValues = Array{nmT,nDim}[]
    nImputedValues = nFill
    lLs  = Float64[]
    BICs = Float64[]
    AICs = Float64[]

    for mi in 1:i.multipleImputations
        emOut = gmm(X,i.K;p₀=i.p₀,mixtures=i.mixtures,tol=i.tol,verbosity=i.verbosity,minVariance=i.minVariance,minCovariance=i.minCovariance,initStrategy=i.initStrategy,maxIter=i.maxIter,rng=i.rng)

        X̂ = copy(X)
        for n in 1:N
            for d in 1:D
                if !XMask[n,d]
                    X̂[n,d] = sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K])
                end
            end
        end
        X̂ = convert(Array{nmT,nDim},X̂)
        push!(imputedValues,X̂)
        push!(lLs,emOut.lL)
        push!(BICs,emOut.BIC)
        push!(AICss,emOut.AIC)
    end
    return GMMImputerResult(imputedValues,nImputedValues,lLs,BICs,AICs)
end

imputed(r::MeanImputerResult) = mean(r.imputed)
imputedValues(r::MeanImputerResult) = r.imputed
info(r::MeanImputerResult) = (nImputedValues = r.nImputedValues, lL=r.lL, BIC=r.BIC, AIC=r.AIC)

# ------------------------------------------------------------------------------
# RFImputer

Base.@kwdef mutable struct RFImputer <: Imputer
    multipleImputations::Int64         = 1
    rng::Random._GLOBAL_RNG            = Random.GLOBAL_RNG
end

struct RFmputerResult <: ImputerResult
    imputed
    nImputedValues::Int64
end



function impute(X,imputer::RFImputer)
   return 
end


