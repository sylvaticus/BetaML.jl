"""
    Imputation.jl file

Implement the BetaML.Imputation module

`?BetaML.Imputation` for documentation

- Go to [https://sylvaticus.github.io/BetaML.jl](https://sylvaticus.github.io/BetaML.jl/stable) for more general doc

"""

"""
    Imputation module

Provide various imputation methods for missing data. Note that the interpretation of "missing" can be very wide.
For example, reccomendation systems / collaborative filtering (e.g. suggestion of the film to watch) can well be representated as a missing data to impute problem.

- [`MeanImputer`](@ref): Simple imputator using the features or the records means, with optional record normalisation (fastest)
- [`GMMImputer`](@ref): Impute data using a Generative (Gaussian) Mixture Model (good trade off)
- [`RFImputer`](@ref): Impute missing data using Random Forests, with optional replicable multiple imputations (most accurate).


Imputations for all these models can be optained by running `fit!([Imputator model],X)`. The data with the missing values imputed can then be obtained with `predict(m::Imputer)`. Use`info(m::Imputer)` to retrieve further information concerning the imputation.
Note that if multiple imputations are run (for the supporting imputators) `predict()` will return a vector of predictions rather than a single one`.

## Example   

```julia
julia> using Statistics, BetaML

julia> X            = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ; 4 8 20; 1 2 5]
6×3 Matrix{Union{Missing, Int64}}:
    2      missing     10
 2000  4000          1000
 2000  4000         10000
    3     5            12
    4     8            20
    1     2             5

julia> mod          = RFImputer(multipleImputations=10,  rng=copy(FIXEDRNG));

julia> fit!(mod,X)
true

julia> vals         = predict(mod)
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

julia> nR,nC        = size(vals[1])
(6, 3)

julia> medianValues = [median([v[r,c] for v in vals]) for r in 1:nR, c in 1:nC]
6×3 Matrix{Float64}:
    2.0     4.0     10.0
 2000.0  4000.0   1000.0
 2000.0  4000.0  10000.0
    3.0     5.0     12.0
    4.0     8.0     20.0
    1.0     2.0      5.0

julia> infos        = info(mod);

julia> infos.nImputedValues
1
```
"""
module Imputation

using Statistics, Random, LinearAlgebra
using ForceImport
@force using ..Api
@force using ..Utils
@force using ..Clustering
@force using ..GMM
@force using ..Trees

import ..GMM.estep

export predictMissing,
       Imputer, MeanImputer, GMMImputer, RFImputer,
       ImputerResult, RFImputerResult, 
       fit!, predict, info

abstract type Imputer <: BetaMLModel end   
abstract type ImputerResult end

# ------------------------------------------------------------------------------

"""
predictMissing(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance)

OLD API. Use [`GMMClusterer`](@ref) instead.

Fill missing entries in a sparse matrix (i.e. perform a "matrix completion") assuming an underlying Gaussian Mixture probabilistic Model (GMM) and implementing
an Expectation-Maximisation algorithm.

While the name of the function is `predictMissing`, the function can be also used for system reccomendation / collaborative filtering and GMM-based regressions. The advantage over traditional algorithms as k-nearest neighbors (KNN) is that GMM can "detect" the hidden structure of the observed data, where some observation can be similar to a certain pool of other observvations for a certain characteristic, but similar to an other pool of observations for other characteristics.

Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :           A (N x D) sparse matrix of data to fill according to a GMM model
* `K`  :           Number of mixtures (latent classes) to consider [def: 3]
* `p₀` :           Initial probabilities of the categorical distribution (K x 1) [default: `[]`]
* `mixtures`:      An array (of length K) of the mixture to employ (see notes) [def: `[DiagonalGaussian() for i in 1:K]`]
* `tol`:           Tolerance to stop the algorithm [default: 10^(-6)]
* `verbosity`:     A verbosity parameter regulating the information messages frequency [def: `STD`]
* `minVariance`:   Minimum variance for the mixtures [default: 0.05]
* `minCovariance`: Minimum covariance for the mixtures with full covariance matrix [default: 0]. This should be set different than minVariance (see notes).
* `initStrategy`:  Mixture initialisation algorithm [def: `grid`]
* `maxIter`:       Maximum number of iterations [def: `typemax(Int64)`, i.e. ∞]
* `rng`:           Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A named touple of:
* `̂X̂`    : The Filled Matrix of size (N x D)
* `nFill`: The number of items filled
* `lL`   : The log-likelihood (without considering the last mixture optimisation)
* `BIC` :  The Bayesian Information Criterion (lower is better)
* `AIC` :  The Akaike Information Criterion (lower is better)

# Notes:
- The mixtures currently implemented are `SphericalGaussian(μ,σ²)`,`DiagonalGaussian(μ,σ²)` and `FullGaussian(μ,σ²)`
- For `initStrategy`, look at the documentation of `initMixtures!` for the mixture you want. The provided gaussian mixtures support `grid`, `kmeans` or `given`. `grid` is faster, but `kmeans` often provides better results.
- The algorithm requires to specify a number of "latent classes" (mlixtures) to divide the dataset into. If there isn't any prior domain specific knowledge on this point one can test sevaral `k` and verify which one minimise the `BIC` or `AIC` criteria.


# Example:
```julia
julia>  cFOut = predictMissing([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3)
```
"""
function predictMissing(X,K=3;p₀=[],mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minVariance=0.05,minCovariance=0.0,initStrategy="kmeans",maxIter=typemax(Int64),rng = Random.GLOBAL_RNG)
 if verbosity > STD
     @codeLocation
 end
 emOut = gmm(X,K;p₀=p₀,mixtures=mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,maxIter=maxIter,rng=rng)

 (N,D) = size(X)
 nDim  = ndims(X)
 nmT   = nonmissingtype(eltype(X))
 #K = size(emOut.μ)[1]
 XMask = .! ismissing.(X)
 nFill = (N * D) - sum(XMask)
 #=
 X̂ = copy(X)
 for n in 1:N
     for d in 1:D
         if !XMask[n,d]
              X̂[n,d] = sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K])
         end
     end
 end
 =#
 X̂ = [XMask[n,d] ? X[n,d] : sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K]) for n in 1:N, d in 1:D ]
 X̂ = identity.(X̂)
 #X̂ = convert(Array{nmT,nDim},X̂)
 return (X̂=X̂,nFill=nFill,lL=emOut.lL,BIC=emOut.BIC,AIC=emOut.AIC)
end

# ------------------------------------------------------------------------------
# MeanImputer

Base.@kwdef mutable struct MeanImputerHyperParametersSet <: BetaMLHyperParametersSet
    norm::Union{Nothing,Int64}       = nothing
end
Base.@kwdef mutable struct MeanImputerLearnableParameters <: BetaMLLearnableParametersSet
    cMeans::Vector{Float64} = []
    norms::Vector{Float64}  = []
    imputedValues::Union{Nothing,Matrix{Float64}} = nothing
end

"""
    MeanImputer

Simple imputer using the feature (column) mean, optionally normalised by l-norms of the records (rows)

Parameters:
- `norm`: Normalise the feature mean by l-`norm` norm of the records [default: `nothing`]. Use it (e.g. `norm=1` to use the l-1 norm) if the records are highly heterogeneus (e.g. quantity exports of different countries).  

Limitations:
- data must be numerical
"""
mutable struct MeanImputer <: Imputer
    hpar::MeanImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,MeanImputerLearnableParameters}
    fitted::Bool
    info::Dict{Symbol,Any}
end

function MeanImputer(;kwargs...)
    m              = MeanImputer(MeanImputerHyperParametersSet(),BetaMLDefaultOptionsSet(),MeanImputerLearnableParameters(),false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
          end
        end
    end
    return m
end


"""
    fit!(imputer::MeanImputer,X)

Fit a matrix with missing data using [`MeanImputer`](@ref)
"""
function fit!(imputer::MeanImputer,X)
    (imputer.fitted == false ) || error("multiple training unsupported on this model")
    #X̂ = copy(X)
    nR,nC = size(X)
    missingMask = ismissing.(X)
    cMeans   = [mean(skipmissing(X[:,i])) for i in 1:nC]

    if imputer.hpar.norm == nothing
        adjNorms = []
        X̂ = [missingMask[r,c] ? cMeans[c] : X[r,c] for r in 1:nR, c in 1:nC]
    else
        adjNorms = [norm(collect(skipmissing(r)),imputer.hpar.norm) /   (nC - sum(ismissing.(r))) for r in eachrow(X)]
        X̂        = [missingMask[r,c] ? cMeans[c]*adjNorms[r]/sum(adjNorms) : X[r,c] for r in 1:nR, c in 1:nC]
    end
    imputer.par = MeanImputerLearnableParameters(cMeans,adjNorms,X̂)
    imputer.info[:nImputedValues] = sum(missingMask)
    imputer.fitted = true
    return true
end
"""
    predict(m::MeanImputer)

Return the data with the missing values replaced with the imputed ones using [`MeanImputer`](@ref).
"""
predict(m::MeanImputer) = m.par.imputedValues

"""
    predict(m::MeanImputer)

Return the data with the missing values replaced with the imputed ones using [`MeanImputer`](@ref).
"""
function predict(m::MeanImputer,X)
    nR,nC = size(X)
    m.fitted || error()
    nC == length(m.par.cMeans) || error()
    (m.hpar.norm == nothing || nR == length(m.par.norms)) || error()

    missingMask = ismissing.(X)
    if m.hpar.norm == nothing
        X̂ = [missingMask[r,c] ? m.par.cMeans[c] : X[r,c] for r in 1:nR, c in 1:nC]
    else
        X̂        = [missingMask[r,c] ? m.par.cMeans[c]*m.par.adjNorms[r]/sum(m.par.adjNorms) : X[r,c] for r in 1:nR, c in 1:nC]
    end
    return X̂
end

function show(io::IO, ::MIME"text/plain", m::MeanImputer)
    if m.fitted == false
        print(io,"MeanImputer - A simple feature-mean imputer (unfitted)")
    else
        print(io,"MeanImputer - A simple feature-mean imputer (fitted)")
    end
end

function show(io::IO, m::MeanImputer)
    if m.fitted == false
        print(io,"MeanImputer - A simple feature-mean imputer (unfitted)")
    else
        print(io,"MeanImputer - A simple feature-mean imputer (fitted)")
        println(io,m.info)
    end
end


# ------------------------------------------------------------------------------
# GMMImputer
Base.@kwdef mutable struct GMMImputerLearnableParameters <: BetaMLLearnableParametersSet
    mixtures::Vector{AbstractMixture}           = []
    probMixtures::Vector{Float64}               = []
    probRecords::Union{Nothing,Matrix{Float64}} = nothing
    imputedValues                               = nothing
end


"""
    GMMImputer

Missing data imputer that uses a Generated (Gaussian) Mixture Model.

For the parameters (`nClasses`,`mixtures`,..) see  [`GMMImputerLearnableParameters`](@ref).

Limitations:
- data must be numerical
- the resulted matrix is a Matrix{Float64}
- currently the Mixtures available do not support random initialisation for missing imputation, and the rest of the algorithm (we use the Expectation-Maximisation) is deterministic, so there is no random component involved (i.e. no multiple imputations)    
"""
mutable struct GMMImputer <: Imputer
    hpar::GMMClusterHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{GMMImputerLearnableParameters,Nothing}
    fitted::Bool
    info::Dict{Symbol,Any}    
end

function GMMImputer(;kwargs...)
    # ugly manual case...
    if (:nClasses in keys(kwargs) && ! (:mixtures in keys(kwargs)))
        nClasses = kwargs[:nClasses]
        hps = GMMClusterHyperParametersSet(nClasses = nClasses, mixtures = [DiagonalGaussian() for i in 1:nClasses])
    else 
        hps = GMMClusterHyperParametersSet()
    end
    m              = GMMImputer(hps,BetaMLDefaultOptionsSet(),GMMImputerLearnableParameters(),false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
          end
        end
    end
    return m
end


"""
    fit!(imputer::GMMImputer,X)

Fit a matrix with missing data using [`GMMImputer`](@ref)
"""
function fit!(m::GMMImputer,X)
    

    # Parameter alias..
    K             = m.hpar.nClasses
    p₀            = m.hpar.probMixtures
    mixtures      = m.hpar.mixtures
    tol           = m.hpar.tol
    minVariance   = m.hpar.minVariance
    minCovariance = m.hpar.minCovariance
    initStrategy  = m.hpar.initStrategy
    maxIter       = m.hpar.maxIter
    verbosity     = m.opt.verbosity
    rng           = m.opt.rng

    if m.opt.verbosity > STD
        @codeLocation
    end
    if m.fitted
        verbosity >= STD && @warn "Continuing training of a pre-fitted model"
        emOut = gmm(X,K;p₀=m.par.probMixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy="given",maxIter=maxIter,rng = rng)
    else
        emOut = gmm(X,K;p₀=p₀,mixtures=mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,maxIter=maxIter,rng = rng)
    end

    (N,D) = size(X)
    nDim  = ndims(X)
    nmT   = nonmissingtype(eltype(X))

    XMask = .! ismissing.(X)
    nFill = (N * D) - sum(XMask)

    nImputedValues = nFill

    X̂ = [XMask[n,d] ? X[n,d] : sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K]) for n in 1:N, d in 1:D ]
    
    m.par  = GMMImputerLearnableParameters(mixtures = emOut.mixtures, probMixtures=makeColVector(emOut.pₖ), probRecords = emOut.pₙₖ, imputedValues=X̂)

    m.info[:error]          = emOut.ϵ
    m.info[:lL]             = emOut.lL
    m.info[:BIC]            = emOut.BIC
    m.info[:AIC]            = emOut.AIC
    m.info[:fittedRecords] = get(m.info,:fittedRecords,0) + size(X,1)
    m.info[:dimensions]     = size(X,2)
    m.info[:nImputedValues]     = nImputedValues
    m.fitted=true
    return true
end

"""
    predict(m::GMMImputer)

Return the data with the missing values replaced with the imputed ones using [`GMMImputer`](@ref).
"""
predict(m::GMMImputer) = (! m.fitted) ? nothing : m.par.imputedValues 

function predict(m::GMMImputer,X)
    m.fitted || error("Trying to predict from an untrained model")
    X   = makeMatrix(X)
    N,D = size(X)
    XMask = .! ismissing.(X)
    mixtures = m.par.mixtures
    probMixtures = m.par.probMixtures
    probRecords, lL = estep(X,probMixtures,mixtures)

    X̂ = [XMask[n,d] ? X[n,d] : sum([mixtures[k].μ[d] * probRecords[n,k] for k in 1:m.hpar.nClasses]) for n in 1:N, d in 1:D ]
    
    return X̂
end

function show(io::IO, ::MIME"text/plain", m::GMMImputer)
    if m.fitted == false
        print(io,"GMMImputer - A Gaussian Mixture Model based imputer (unfitted)")
    else
        print(io,"GMMImputer - A Gaussian Mixture Model based imputer (fitted)")
    end
end

function show(io::IO, m::GMMImputer)
    if m.fitted == false
        print(io,"GMMImputer - A Gaussian Mixture Model based imputer (unfitted)")
    else
        print(io,"GMMImputer - A Gaussian Mixture Model based imputer (fitted)")
        println(io,m.info)
    end
end

# ------------------------------------------------------------------------------
# RFImputer
struct RFImputerResult <: ImputerResult
    imputedValues
    nImputedValues::Int64
    oob::Vector{Vector{Float64}}
end

"""
    RFImputer

Impute missing data using Random Forests, with optional replicable multiple imputations. 

For the underlying random forest algorithm parameters (`nTrees`,`maxDepth`,`minGain`,`minRecords`,`maxFeatures:`,`splittingCriterion`,`β`,`initStrategy`, `oob` and `rng`) see [`buildTree`](@ref) and [`buildForest`](@ref).

### Specific parameters:
- `forcedCategoricalCols`: specify the positions of the integer columns to treat as categorical instead of cardinal. [Default: empty vector (all numerical cols are treated as cardinal by default and the others as categorical)]
- `recursivePassages `: Define the times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`].
- `multipleImputations`: Determine the number of independent imputation of the whole dataset to make. Note that while independent, the imputations share the same random number generator (RNG).

### Notes:
- Given a certain RNG and its status (e.g. `RFImputer(...,rng=StableRNG(FIXEDSEED))`), the algorithm is completely deterministic, i.e. replicable. 
- The algorithm accepts virtually any kind of data, sortable or not
"""
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
    recursivePassages::Int64                    = 1
    multipleImputations::Int64                  = 1
    rng::AbstractRNG                            = Random.GLOBAL_RNG
    verbosity::Verbosity                        = STD
    fitResults::Union{RFImputerResult,Nothing}  = nothing
    fitted::Bool = false
end

"""
    fit!(imputer::RFImputer,X)

Fit a matrix with missing data using [`RFImputer`](@ref)
"""
function fit!(imputer::RFImputer,X)
    nR,nC   = size(X)
    
    imputed = fill(similar(X),imputer.multipleImputations)
    if imputer.maxFeatures == typemax(Int64) && imputer.nTrees >1
      maxFeatures = Int(round(sqrt(size(X,2))))
    end
    maxFeatures   = min(nC,imputer.maxFeatures) 
    maxDepth      = min(nR,imputer.maxDepth)

    catCols = [! (nonmissingtype(eltype(identity.(X[:,c]))) <: Number ) || c in imputer.forcedCategoricalCols for c in 1:nC]

    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    nImputedValues = sum(missingMask)
    oobErrors      = fill(fill(Inf,nC),imputer.multipleImputations) # by imputations and dimensions
    
    for imputation in 1:imputer.multipleImputations
        imputer.verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        sortedDims     = reverse(sortperm(makeColVector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        oobErrorsImputation = fill(Inf,nC)
        for pass in 1:imputer.recursivePassages 
            imputer.verbosity >= HIGH && println("- processing passage $pass")
            if pass > 1
                shuffle!(imputer.rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                imputer.verbosity >= FULL && println("  - processing dimension $d")
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
                            maxDepth            = maxDepth,
                            minGain             = imputer.minGain,
                            minRecords          = imputer.minRecords,
                            maxFeatures         = maxFeatures,
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
    imputer.fitResults = RFImputerResult(imputed,nImputedValues,oobErrors)
    imputer.fitted = true
    return true
end

"""
    predict(m::RFImputer)

Return the data with the missing values replaced with the imputed ones using [`RFImputer`](@ref). If `multipleImputations` was set >1 this is a vector of matrices (the individual imputations) instead of a single matrix.
"""
predict(m::RFImputer) =  (! m.fitted) ? nothing : (m.multipleImputations == 1 ? m.fitResults.imputedValues[1] : m.fitResults.imputedValues)

"""
    info(m::RFImputer)

Return wheter the model has been fitted, the number of imputed values and, if the option `oob` was set, the estimated _out-of-bag_ errors for each dimension (and individual imputation). The oob error reported is the mismatching error for classification and the relative mean error for regression.
"""
info(m::RFImputer) = m.fitted ? (fitted = true, nImputedValues = m.fitResults.nImputedValues, oob = m.fitResults.oob) : (fitted= false, nImputedValues = nothing, oob = nothing)


# MLJ interface
include("Imputation_MLJ.jl")

end # end Imputation module