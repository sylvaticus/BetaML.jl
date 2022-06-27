"""
    Imputation.jl file

Implement the BetaML.Imputation module

`?BetaML.Imputation` for documentation

- Go to [https://sylvaticus.github.io/BetaML.jl](https://sylvaticus.github.io/BetaML.jl/stable) for more general doc

"""

"""
    Imputation module

Provide various imputation methods for missing data. Note that the interpretation of "missing" can be very wide. Reccomendation systems / collaborative filtering (e.g. suggestion of the film to watch) may well be representated as missing data to impute.

- [`MeanImputer`](@ref): Simple imputator using the features or the records (or both) means (fastest)
- [`GMMImputer`](@ref): Impute data using a Generative (Gaussian) Mixture Model (good trade off)
- [`RFImputer`](@ref): Impute missing data using Random Forests, with optional replicable multiple imputations (most accurate).


Imputations for all these models can be optained by running `impute([Imputator model],X)`. The function returns an `ImputerResult` that can be queried with `imputed(x::ImputerResult)`, `imputedValues(x::ImputerResult)` (for multiple imputations) and `info(x::ImputerResult)` to query further informations concerning the imputation.
Note that if multiple imputations are run (for the supporting ipmutators) `impute()` just returns the last one. If you want the average, the mode or other functions of your differently imputed data use directly the output of `imputedValues()`.

## Example   

```julia
julia> X            = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ; 4 8 20; 1 2 5]
6×3 Matrix{Union{Missing, Int64}}:
    2      missing     10
 2000  4000          1000
 2000  4000         10000
    3     5            12
    4     8            20
    1     2             5
julia> mod          = RFImputer(multipleImputations=10, rng=copy(FIXEDRNG))
RFImputer(30, 9223372036854775807, 0.0, 2, 9223372036854775807, Int64[], nothing, 0.0, false, 1, 10, StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7))
julia> results      = impute(mod,X);
julia> vals         = imputedValues(results)
10-element Vector{Matrix{Union{Missing, Int64}}}:
 [2 3 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 4 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 4 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 136 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 137 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 4 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 4 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 4 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 137 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
 [2 137 10; 2000 4000 1000; … ; 4 8 20; 1 2 5]
julia> medianValues = [median([v[r,c] for v in vals]) for r in 1:nR, c in 1:nC]
6×3 Matrix{Float64}:
    2.0     4.0     10.0
 2000.0  4000.0   1000.0
 2000.0  4000.0  10000.0
    3.0     5.0     12.0
    4.0     8.0     20.0
    1.0     2.0      5.0
julia> infos        = info(results);
julia> infos.nImputedValues
1
```
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

"""
    MeanImputer

Impute using Generated (Gaussian) mixture models.
Limitations:
- data must be numerical
- the resulted matrix is a Matrix{Float64}
- currently the Mixtures available do not support random initialisation, so there is no random component involved (i.e. no multiple imputations)    
"""
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
    oob::Vector{Vector{Float64}}
end

function impute(imputer::RFImputer,X)
    nR,nC   = size(X)
    
    imputed = fill(similar(X),imputer.multipleImputations)
    if imputer.maxFeatures == typemax(Int64) && imputer.nTrees >1
      maxFeatures = Int(round(sqrt(size(X,2))))
    end
    maxFeatures   = min(nC,imputer.maxFeatures) 
    maxDepth      = min(nR,imputer.maxDepth)

    catCols = [! (nonmissingtype(eltype(identity.(X[:,c]))) <: Number ) || c in imputer.forcedCategoricalCols for c in 1:nC]

    #forestModels   = Array{Forest,1}(undef,nC)
    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    nImputedValues = sum(missingMask)
    oobErrors      = fill(fill(Inf,nC),imputer.multipleImputations) # by imputations and dimensions
    
    for imputation in 1:imputer.multipleImputations
        Xout    = copy(X)
        sortedDims     = reverse(sortperm(makeColVector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        oobErrorsImputation = fill(Inf,nC)
        for pass in 1:imputer.recursivePassages 
            if pass > 1
                shuffle!(imputer.rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                if imputer.splittingCriterion == nothing
                    splittingCriterion = catCols[d] ?  gini : variance
                else
                    splittingCriterion = imputer.splittingCriterion
                end
                nmy  = nonMissingMask[:,d]
                y    = X[nmy,d]
                ty   = nonmissingtype(eltype(y))
                y    = convert(Vector{ty},y)
                Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
                dfor = buildForest(Xd,y, # forest model specific for this dimension
                            imputer.nTrees,
                            maxDepth            = imputer.maxDepth,
                            minGain             = imputer.minGain,
                            minRecords          = imputer.minRecords,
                            maxFeatures         = imputer.maxFeatures,
                            splittingCriterion  = splittingCriterion,
                            β                   = imputer.β,
                            oob                 = false,
                            rng                 = imputer.rng,
                            forceClassification = catCols[d])
                # imputing missing values in d...
                for i in 1:nR
                    if ! missingMask[i,d]
                        continue
                    end
                    xrow = permutedims(Vector(Xout[i,[1:(d-1);(d+1):end]]))
                    yest = predict(dfor,xrow,rng=imputer.rng)[1]
                    
                    if ty <: Int 
                        if catCols[d]
                            yest = parse(ty,mode(yest))
                        else
                            yest = Int(round(yest))
                        end
                    elseif !(ty <: Number)
                        yest = mode(yest)
                    end
                    
                    Xout[i,d] = yest
                    #return Xout
                end
                # Compute oob errors on last passages if requested
                if pass == imputer.recursivePassages && imputer.oob
                    oobErrorsImputation[d] = Trees.oobError(dfor,Xd,y,rng=imputer.rng) # BetaML.Trees.oobError(dfor,Xd,y)
                end
            end # end dimension
        end # end recursive passage pass
        imputed[imputation]   = Xout
        oobErrors[imputation] = oobErrorsImputation
    end # end individual imputation
    return RFImputerResult(imputed,nImputedValues,oobErrors)
    
end

imputed(r::RFImputerResult) = r.imputed[1]
imputedValues(r::RFImputerResult) = r.imputed
info(r::RFImputerResult) = (nImputedValues=r.nImputedValues,oob=r.oob)

end # end Imputation module