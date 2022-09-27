"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
    Imputation.jl file

Implement the BetaML.Imputation module

`?BetaML.Imputation` for documentation

- Go to [https://sylvaticus.github.io/BetaML.jl](https://sylvaticus.github.io/BetaML.jl/stable) for more general doc

"""

"""
    Imputation module

Provide various imputation methods for missing data. Note that the interpretation of "missing" can be very wide.
For example, reccomendation systems / collaborative filtering (e.g. suggestion of the film to watch) can well be representated as a missing data to impute problem, often with better results than traditional algorithms as k-nearest neighbors (KNN)

Provided imputers:

- [`FeatureBasedImputer`](@ref): Impute data using the feature (column) mean, optionally normalised by l-norms of the records (rows) (fastest)
- [`GMMImputer`](@ref): Impute data using a Generative (Gaussian) Mixture Model (good trade off)
- [`RFImputer`](@ref): Impute missing data using Random Forests, with optional replicable multiple imputations (most accurate).
- [`UniversalImputer`](@ref): Impute missing data using a vector (one per column) of arbitrary learning models (classifiers/regressors) that implement `m = Model([options])`, `fit!(m,X,Y)` and `predict(m,X)`.


Imputations for all these models can be optained by running `mod = ImputatorModel([options])`, `fit!(mod,X)`. The data with the missing values imputed can then be obtained with `predict(mod)`. Use`info(m::Imputer)` to retrieve further information concerning the imputation.
Trained models can be also used to impute missing values in new data with `predict(mox,xNew)`.
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

julia> mod          = RFImputer(multiple_imputations=10,  rng=copy(FIXEDRNG));

julia> fit!(mod,X);

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

julia> infos["n_imputed_values"]
1
```
"""
module Imputation

using Statistics, Random, LinearAlgebra, StableRNGs, DocStringExtensions
using ForceImport
@force using ..Api
@force using ..Utils
@force using ..Clustering
@force using ..GMM
@force using ..Trees

import ..GMM.estep
import Base.print
import Base.show

export predictMissing,
       FeatureBasedImputerHyperParametersSet, RFImputerHyperParametersSet,UniversalImputerHyperParametersSet,
       Imputer, FeatureBasedImputer, GMMImputer, RFImputer, UniversalImputer,
       fit!, predict, info

abstract type Imputer <: BetaMLModel end   

# ------------------------------------------------------------------------------

"""
    predictMissing(X,K;initial_probmixtures,mixtures,tol,verbosity,minimum_variance,minimum_covariance)

!!! warning
    This function is deprecated and will possibly be removed in BetaML 0.9.
    Use the model [`GMMClusterer`](@ref) instead. 

Fill missing entries in a sparse matrix (i.e. perform a "matrix completion") assuming an underlying Gaussian Mixture probabilistic Model (GMM) fitted using an Expectation-Maximisation algorithm.

While the name of the function is `predictMissing`, the function can be also used for system reccomendation / collaborative filtering and GMM-based regressions. The advantage over traditional algorithms as k-nearest neighbors (KNN) is that GMM can "detect" the hidden structure of the observed data, where some observation can be similar to a certain pool of other observvations for a certain characteristic, but similar to an other pool of observations for other characteristics.

Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :           A (N x D) sparse matrix of data to fill according to a GMM model
* `K`  :           Number of mixtures (latent classes) to consider [def: 3]
* `initial_probmixtures` :           Initial probabilities of the categorical distribution (K x 1) [default: `[]`]
* `mixtures`:      An array (of length K) of the mixture to employ (see notes) [def: `[DiagonalGaussian() for i in 1:K]`]
* `tol`:           Tolerance to stop the algorithm [default: 10^(-6)]
* `verbosity`:     A verbosity parameter regulating the information messages frequency [def: `STD`]
* `minimum_variance`:   Minimum variance for the mixtures [default: 0.05]
* `minimum_covariance`: Minimum covariance for the mixtures with full covariance matrix [default: 0]. This should be set different than minimum_variance (see notes).
* `initialisation_strategy`:  Mixture initialisation algorithm [def: `grid`]
* `maximum_iterations`:       Maximum number of iterations [def: `typemax(Int64)`, i.e. ∞]
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
- For `initialisation_strategy`, look at the documentation of `init_mixtures!` for the mixture you want. The provided gaussian mixtures support `grid`, `kmeans` or `given`. `grid` is faster, but `kmeans` often provides better results.
- The algorithm requires to specify a number of "latent classes" (mlixtures) to divide the dataset into. If there isn't any prior domain specific knowledge on this point one can test sevaral `k` and verify which one minimise the `BIC` or `AIC` criteria.


# Example:
```julia
julia>  cFOut = predictMissing([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3)
```
"""
function predictMissing(X,K=3;initial_probmixtures=[],mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minimum_variance=0.05,minimum_covariance=0.0,initialisation_strategy="kmeans",maximum_iterations=typemax(Int64),rng = Random.GLOBAL_RNG)
 if verbosity > STD
     @codelocation
 end
 emOut = gmm(X,K;initial_probmixtures=initial_probmixtures,mixtures=mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy=initialisation_strategy,maximum_iterations=maximum_iterations,rng=rng)

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
# FeatureBasedImputer
"""
$(TYPEDEF)

Hyperparameters for the [`FeatureBasedImputer`](@ref) model

# Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct FeatureBasedImputerHyperParametersSet <: BetaMLHyperParametersSet
    "The descriptive statistic of the column (feature) to use as imputed value [def: `mean`]"
    statistic::Function                   = mean
    "Normalise the feature mean by l-`norm` norm of the records [default: `nothing`]. Use it (e.g. `norm=1` to use the l-1 norm) if the records are highly heterogeneus (e.g. quantity exports of different countries)."
    norm::Union{Nothing,Int64}       = nothing

end
Base.@kwdef mutable struct FeatureBasedImputerLearnableParameters <: BetaMLLearnableParametersSet
    cStats::Vector{Float64} = []
    norms::Vector{Float64}  = []
    #imputedValues::Union{Nothing,Matrix{Float64}} = nothing
end

"""
$(TYPEDEF)

Simple imputer using the missing data's feature (column) statistic (def: `mean`), optionally normalised by l-norms of the records (rows)

# Parameters:
- `statistics`: The descriptive statistic of the column (feature) to use as imputed value [def: `mean`]
- `norm`: Normalise the feature mean by l-`norm` norm of the records [default: `nothing`]. Use it (e.g. `norm=1` to use the l-1 norm) if the records are highly heterogeneus (e.g. quantity exports of different countries).  

# Limitations:
- data must be numerical
"""
mutable struct FeatureBasedImputer <: Imputer
    hpar::FeatureBasedImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,FeatureBasedImputerLearnableParameters}
    cres::Union{Nothing,Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}
end

function FeatureBasedImputer(;kwargs...)
    m              = FeatureBasedImputer(FeatureBasedImputerHyperParametersSet(),BetaMLDefaultOptionsSet(),FeatureBasedImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end


"""
$(TYPEDSIGNATURES)

Fit a matrix with missing data using [`FeatureBasedImputer`](@ref)
"""
function fit!(imputer::FeatureBasedImputer,X)
    (imputer.fitted == false ) || error("multiple training unsupported on this model")
    #X̂ = copy(X)
    nR,nC = size(X)
    cache       = imputer.opt.cache
    missingMask = ismissing.(X)
    overallStat = mean(skipmissing(X))
    statf       = imputer.hpar.statistic
    cStats      = [sum(ismissing.(X[:,i])) == nR ? overallStat : statf(skipmissing(X[:,i])) for i in 1:nC]

    if imputer.hpar.norm == nothing
        adjNorms = []
        X̂ = [missingMask[r,c] ? cStats[c] : X[r,c] for r in 1:nR, c in 1:nC]
    else
        adjNorms = [sum(ismissing.(r)) == nC ? missing : norm(collect(skipmissing(r)),imputer.hpar.norm) /   (nC - sum(ismissing.(r))) for r in eachrow(X)]
        adjNormsMean = mean(skipmissing(adjNorms))
        adjNorms[ismissing.(adjNorms)] .= adjNormsMean
        X̂        = [missingMask[r,c] ? cStats[c]*adjNorms[r]/sum(adjNorms) : X[r,c] for r in 1:nR, c in 1:nC]
    end
    imputer.par = FeatureBasedImputerLearnableParameters(cStats,adjNorms)
    imputer.cres = cache ? X̂ : nothing
    imputer.info["n_imputed_values"] = sum(missingMask)
    imputer.fitted = true
    return cache ? imputer.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Predict the missing data using the feature averages (eventually normalised) learned by fitting a [`FeatureBasedImputer`](@ref) model
"""
function predict(m::FeatureBasedImputer,X)
    nR,nC = size(X)
    m.fitted || error()
    nC == length(m.par.cStats) || error("`FeatureBasedImputer` can only predict missing values in matrices with the same number of columns as the matrice it has been trained with.")
    (m.hpar.norm == nothing || nR == length(m.par.norms)) || error("If norms are used, `FeatureBasedImputer` can predict only matrices with the same number of rows as the matrix it has been trained with.")

    missingMask = ismissing.(X)
    if m.hpar.norm == nothing
        X̂ = [missingMask[r,c] ? m.par.cStats[c] : X[r,c] for r in 1:nR, c in 1:nC]
    else
        X̂        = [missingMask[r,c] ? m.par.cStats[c]*m.par.norms[r]/sum(m.par.norms) : X[r,c] for r in 1:nR, c in 1:nC]
    end
    return X̂
end

function show(io::IO, ::MIME"text/plain", m::FeatureBasedImputer)
    if m.fitted == false
        print(io,"FeatureBasedImputer - A simple feature-stat based imputer (unfitted)")
    else
        print(io,"FeatureBasedImputer - A simple feature-stat based imputer (fitted)")
    end
end

function show(io::IO, m::FeatureBasedImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"FeatureBasedImputer - A simple feature-stat based imputer (unfitted)")
    else
        print(io,"FeatureBasedImputer - A simple feature-stat based imputer (fitted)")
        println(io,m.info)
    end
end


# ------------------------------------------------------------------------------
# GMMImputer
Base.@kwdef mutable struct GMMImputerLearnableParameters <: BetaMLLearnableParametersSet
    mixtures::Vector{AbstractMixture}           = []
    initial_probmixtures::Vector{Float64}               = []
    probRecords::Union{Nothing,Matrix{Float64}} = nothing
    #imputedValues                               = nothing
end


"""
$(TYPEDEF)

Missing data imputer that uses a Generative (Gaussian) Mixture Model.

For the parameters (`n_classes`,`mixtures`,..) see  [`GMMHyperParametersSet`](@ref).

# Limitations:
- data must be numerical
- the resulted matrix is a Matrix{Float64}
- currently the Mixtures available do not support random initialisation for missing imputation, and the rest of the algorithm (Expectation-Maximisation) is deterministic, so there is no random component involved (i.e. no multiple imputations)
"""
mutable struct GMMImputer <: Imputer
    hpar::GMMHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{GMMImputerLearnableParameters,Nothing}
    cres::Union{Nothing,Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}    
end

function GMMImputer(;kwargs...)
    # ugly manual case...
    if (:n_classes in keys(kwargs))
        n_classes = kwargs[:n_classes]
    else
        n_classes = 3
    end
    if ! (:mixtures in keys(kwargs))
        mixtures = [DiagonalGaussian() for i in 1:n_classes]
    elseif  typeof(kwargs[:mixtures]) <: UnionAll
        mixtures = [kwargs[:mixtures]() for i in 1:n_classes]
    else
        mixtures = kwargs[:mixtures]
    end
    hps = GMMHyperParametersSet(n_classes = n_classes, mixtures = mixtures)

    m   = GMMImputer(hps,BetaMLDefaultOptionsSet(),GMMImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              if kw == :mixtures
                found = true; continue
              end
              setproperty!(fobj,kw,kwv)
              found = true
          end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end


"""
$(TYPEDSIGNATURES)

Fit a matrix with missing data using [`GMMImputer`](@ref)
"""
function fit!(m::GMMImputer,X)
    

    # Parameter alias..
    K             = m.hpar.n_classes
    initial_probmixtures            = m.hpar.initial_probmixtures
    mixtures      = m.hpar.mixtures
    tol           = m.hpar.tol
    minimum_variance   = m.hpar.minimum_variance
    minimum_covariance = m.hpar.minimum_covariance
    initialisation_strategy  = m.hpar.initialisation_strategy
    maximum_iterations       = m.hpar.maximum_iterations
    cache         = m.opt.cache
    verbosity     = m.opt.verbosity
    rng           = m.opt.rng

    if m.opt.verbosity > STD
        @codelocation
    end
    if m.fitted
        verbosity >= STD && @warn "Continuing training of a pre-fitted model"
        emOut = gmm(X,K;initial_probmixtures=m.par.initial_probmixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy="given",maximum_iterations=maximum_iterations,rng = rng)
    else
        emOut = gmm(X,K;initial_probmixtures=initial_probmixtures,mixtures=mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy=initialisation_strategy,maximum_iterations=maximum_iterations,rng = rng)
    end

    (N,D) = size(X)
    nDim  = ndims(X)
    nmT   = nonmissingtype(eltype(X))

    XMask = .! ismissing.(X)
    nFill = (N * D) - sum(XMask)

    n_imputed_values = nFill

    m.par  = GMMImputerLearnableParameters(mixtures = emOut.mixtures, initial_probmixtures=makecolvector(emOut.pₖ), probRecords = emOut.pₙₖ)

    if cache
        X̂ = [XMask[n,d] ? X[n,d] : sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K]) for n in 1:N, d in 1:D ]
        m.cres = X̂
    end

    m.info["error"]          = emOut.ϵ
    m.info["lL"]             = emOut.lL
    m.info["BIC"]            = emOut.BIC
    m.info["AIC"]            = emOut.AIC
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(X,1)
    m.info["dimensions"]     = size(X,2)
    m.info["n_imputed_values"]     = n_imputed_values
    m.fitted=true
    return cache ? m.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Predict the missing data using the mixtures learned by fitting a [`GMMImputer`](@ref) model

"""
function predict(m::GMMImputer,X)
    m.fitted || error("Trying to predict from an untrained model")
    X   = makematrix(X)
    N,D = size(X)
    XMask = .! ismissing.(X)
    mixtures = m.par.mixtures
    initial_probmixtures = m.par.initial_probmixtures
    probRecords, lL = estep(X,initial_probmixtures,mixtures)

    X̂ = [XMask[n,d] ? X[n,d] : sum([mixtures[k].μ[d] * probRecords[n,k] for k in 1:m.hpar.n_classes]) for n in 1:N, d in 1:D ]
    
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
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GMMImputer - A Gaussian Mixture Model based imputer (unfitted)")
    else
        print(io,"GMMImputer - A Gaussian Mixture Model based imputer (fitted)")
        println(io,m.info)
    end
end

# ------------------------------------------------------------------------------
# RFImputer

"""
$(TYPEDEF)

Hyperparameters for [`RFImputer`](@ref)

# Parameters:
$(TYPEDFIELDS)

# Example:
```
julia>mod = RFImputer(n_trees=20,max_depth=10,recursive_passages=3)
```
"""
Base.@kwdef mutable struct RFImputerHyperParametersSet <: BetaMLHyperParametersSet
    "For the underlying random forest algorithm parameters (`n_trees`,`max_depth`,`min_gain`,`min_records`,`max_features:`,`splitting_criterion`,`β`,`initialisation_strategy`, `oob` and `rng`) see [`RFHyperParametersSet`](@ref) for the specific RF algorithm parameters"
    rfhpar                                      = RFHyperParametersSet()
    "Specify the positions of the integer columns to treat as categorical instead of cardinal. [Default: empty vector (all numerical cols are treated as cardinal by default and the others as categorical)]"
    forced_categorical_cols::Vector{Int64}                = Int64[] # like in RF, normally integers are considered ordinal
    "Define the times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`]."
    recursive_passages::Int64                    = 1
    "Determine the number of independent imputation of the whole dataset to make. Note that while independent, the imputations share the same random number generator (RNG)."
    multiple_imputations::Int64                  = 1
end

Base.@kwdef struct RFImputerLearnableParameters <: BetaMLLearnableParametersSet
    forests        = nothing
    #imputedValues  = nothing
    #n_imputed_values::Int64
    #oob::Vector{Vector{Float64}}
end

"""
$(TYPEDEF)

Impute missing data using Random Forests, with optional replicable multiple imputations. 

See [`RFImputerHyperParametersSet`](@ref), [`RFHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref) for the parameters.

# Notes:
- Given a certain RNG and its status (e.g. `RFImputer(...,rng=StableRNG(FIXEDSEED))`), the algorithm is completely deterministic, i.e. replicable. 
- The algorithm accepts virtually any kind of data, sortable or not
"""
mutable struct RFImputer <: Imputer
    hpar::RFImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{RFImputerLearnableParameters,Nothing}
    cres
    fitted::Bool
    info::Dict{String,Any}    
end

function RFImputer(;kwargs...)
    
    hps = RFImputerHyperParametersSet()
    m   = RFImputer(hps,BetaMLDefaultOptionsSet(),RFImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
          # Looking for the fields of the fields...
          thissubobjfields = fieldnames(nonmissingtype(typeof(fobj)))
          for f2 in thissubobjfields
            fobj2 = getproperty(fobj,f2)
            if kw in fieldnames(typeof(fobj2))
                setproperty!(fobj2,kw,kwv)
                found = true
            end
          end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end

"""
$(TYPEDSIGNATURES)

Fit a matrix with missing data using [`RFImputer`](@ref)
"""
function fit!(m::RFImputer,X)
    nR,nC   = size(X)

    if m.fitted
        @warn "This model has already been fitted and it doesn't support multiple training. This training will override the previous one(s)"
    end

    # Setting default parameters that depends from the data...
    max_depth    = m.hpar.rfhpar.max_depth    == nothing ?  size(X,1) : m.hpar.rfhpar.max_depth
    max_features = m.hpar.rfhpar.max_features == nothing ?  Int(round(sqrt(size(X,2)-1))) : m.hpar.rfhpar.max_features
    # Here only the hpar setting, later for each column
    #splitting_criterion = m.hpar.splitting_criterion == nothing ? ( (Ty <: Number && !m.hpar.force_classification) ? variance : gini) : m.hpar.splitting_criterion
    #splitting_criterion = m.hpar.rfhpar.splitting_criterion
    
    # Setting schortcuts to other hyperparameters/options....
    min_gain             = m.hpar.rfhpar.min_gain
    min_records          = m.hpar.rfhpar.min_records
    #force_classification = m.hpar.rfhpar.force_classification
    n_trees              = m.hpar.rfhpar.n_trees
    β                   = m.hpar.rfhpar.beta
    oob                 = m.hpar.rfhpar.oob
    cache               = m.opt.cache
    rng                 = m.opt.rng
    verbosity           = m.opt.verbosity
     
    forced_categorical_cols        = m.hpar.forced_categorical_cols
    recursive_passages    = m.hpar.recursive_passages
    multiple_imputations  = m.hpar.multiple_imputations

    imputed = fill(similar(X),multiple_imputations)
    if max_features == typemax(Int64) && n_trees >1
      max_features = Int(round(sqrt(size(X,2))))
    end
    max_features   = min(nC,max_features) 
    max_depth      = min(nR,max_depth)

    catCols = [! (nonmissingtype(eltype(identity.(X[:,c]))) <: Number ) || c in forced_categorical_cols for c in 1:nC]

    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    n_imputed_values = sum(missingMask)
    ooberrors      = fill(fill(Inf,nC),multiple_imputations) # by imputations and dimensions
    forests        = Array{Trees.Forest}(undef,multiple_imputations,nC)

    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        sortedDims     = reverse(sortperm(makecolvector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        ooberrorsImputation = fill(Inf,nC)
        for pass in 1:recursive_passages 
            m.opt.verbosity >= HIGH && println("- processing passage $pass")
            if pass > 1
                shuffle!(rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                verbosity >= FULL && println("  - processing dimension $d")
                if m.hpar.rfhpar.splitting_criterion == nothing
                    splitting_criterion = catCols[d] ?  gini : variance
                else
                    splitting_criterion = splitting_criterion
                end
                nmy  = nonMissingMask[:,d]
                y    = X[nmy,d]
                ty   = nonmissingtype(eltype(y))
                y    = convert(Vector{ty},y)
                Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
                dfor = buildForest(Xd,y, # forest model specific for this dimension
                            n_trees,
                            max_depth            = max_depth,
                            min_gain             = min_gain,
                            min_records          = min_records,
                            max_features         = max_features,
                            splitting_criterion  = splitting_criterion,
                            β                   = β,
                            oob                 = false,
                            rng                 = rng,
                            force_classification = catCols[d])
                # imputing missing values in d...
                for i in 1:nR
                    if ! missingMask[i,d]
                        continue
                    end
                    xrow = permutedims(Vector(Xout[i,[1:(d-1);(d+1):end]]))
                    yest = predict(dfor,xrow)[1]
                    
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
                # This is last passage: save the model and compute oob errors if requested
                if pass == recursive_passages 
                    forests[imputation,d] = dfor 
                    if oob
                        ooberrorsImputation[d] = Trees.ooberror(dfor,Xd,y,rng=rng) # BetaML.Trees.ooberror(dfor,Xd,y)
                    end
                end
            end # end dimension
        end # end recursive passage pass
        imputed[imputation]   = Xout

        ooberrors[imputation] = ooberrorsImputation
    end # end individual imputation
    m.par = RFImputerLearnableParameters(forests)
    if cache
        if multiple_imputations == 1
            m.cres = imputed[1]
        else
            m.cres = imputed
        end
    end 
    m.info["n_imputed_values"] = n_imputed_values
    m.info["oob_errors"] = ooberrors

    m.fitted = true
    return cache ? m.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Return the data with the missing values replaced with the imputed ones using the non-linear structure learned fitting a [`RFImputer`](@ref) model.

# Notes:
- If `multiple_imputations` was set > 1 this is a vector of matrices (the individual imputations) instead of a single matrix.
"""
function predict(m::RFImputer,X)
    nR,nC = size(X)
    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    multiple_imputations  = m.hpar.multiple_imputations
    rng = m.opt.rng
    forests = m.par.forests
    verbosity = m.opt.verbosity

    imputed = fill(similar(X),multiple_imputations)
    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        for d in 1:nC
            verbosity >= FULL && println("  - processing dimension $d")
            nmy  = nonMissingMask[:,d]
            y    = X[nmy,d]
            ty   = nonmissingtype(eltype(y))
            y    = convert(Vector{ty},y)
            Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
            dfor = forests[imputation,d]
            # imputing missing values in d...
            for i in 1:nR
                if ! missingMask[i,d]
                    continue
                end
                xrow = permutedims(Vector(Xout[i,[1:(d-1);(d+1):end]]))
                yest = predict(dfor,xrow)[1]
                
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

        end # end dimension
        imputed[imputation]   = Xout
    end # end individual imputation
    multiple_imputations == 1 ? (return imputed[1]) : return imputed
end

function show(io::IO, ::MIME"text/plain", m::RFImputer)
    if m.fitted == false
        print(io,"RFImputer - A Random-Forests based imputer (unfitted)")
    else
        print(io,"RFImputer - A Random-Forests based imputer (fitted)")
    end
end

function show(io::IO, m::RFImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"RFImputer - A Random-Forests based imputer (unfitted)")
    else
        print(io,"RFImputer - A Random-Forests based imputer (fitted)")
        println(io,m.info)
    end
end

# ------------------------------------------------------------------------------
# UniversalImputer

"""
$(TYPEDEF)

Hyperparameters for [`UniversalImputer`](@ref)

# Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct UniversalImputerHyperParametersSet <: BetaMLHyperParametersSet
    "Specify a regressor or classifier model (and its options/hyper-parameters) per each column of the matrix to impute. Default to random forests."
    estimators                        = nothing
    "Define the times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`]."
    recursive_passages::Int64      = 1
    "Determine the number of independent imputation of the whole dataset to make. Note that while independent, the imputations share the same random number generator (RNG)."
    multiple_imputations::Int64    = 1
end

Base.@kwdef struct UniversalImputerLearnableParameters <: BetaMLLearnableParametersSet
    fittedModels  = nothing
    #imputedValues  = nothing
end

"""
$(TYPEDEF)

Impute missing data using any regressor/classifier (not necessarily from BetaML) that implements `m=Model([options])`, `fit!(m,X,Y)` and `predict(m,X)`

See [`UniversalImputerHyperParametersSet`](@ref) for the hyper-parameters.

"""
mutable struct UniversalImputer <: Imputer
    hpar::UniversalImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{UniversalImputerLearnableParameters,Nothing}
    cres
    fitted::Bool
    info::Dict{String,Any}    
end

function UniversalImputer(;kwargs...)
    
    hps = UniversalImputerHyperParametersSet()
    m   = UniversalImputer(hps,BetaMLDefaultOptionsSet(),UniversalImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
          ## Looking for the fields of the fields...
          #thissubobjfields = fieldnames(nonmissingtype(typeof(fobj)))
          #for f2 in thissubobjfields
          #  fobj2 = getproperty(fobj,f2)
          #  if kw in fieldnames(typeof(fobj2))
          #      setproperty!(fobj2,kw,kwv)
          #  end
          #end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end

"""
$(TYPEDSIGNATURES)

Fit a matrix with missing data using [`UniversalImputer`](@ref)
"""
function fit!(m::UniversalImputer,X)
    nR,nC   = size(X)
    multiple_imputations  = m.hpar.multiple_imputations
    recursive_passages    = m.hpar.recursive_passages
    cache                = m.opt.cache
    verbosity            = m.opt.verbosity 
    rng                  = m.opt.rng
    # Setting `estimators`, a matrix of multiple_imputations x nC individual models...
    if ! m.fitted
        if m.hpar.estimators == nothing
            estimators = [RandomForestEstimator(rng = m.opt.rng, verbosity=verbosity) for i in 1:multiple_imputations, d in 1:nC]
        else
            estimators = vcat([permutedims(deepcopy(m.hpar.estimators)) for i in 1:multiple_imputations]...)
        end
    else
        m.opt.verbosity >= STD && @warn "This imputer has already been fitted. Not all learners support multiple training."
        estimators = m.par.fittedModels
    end

    
    imputed = fill(similar(X),multiple_imputations)

    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    n_imputed_values = sum(missingMask)

    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        sortedDims     = reverse(sortperm(makecolvector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        for pass in 1:recursive_passages 
            m.opt.verbosity >= HIGH && println("- processing passage $pass")
            if pass > 1
                shuffle!(rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                verbosity >= FULL && println("  - processing dimension $d")
                nmy  = nonMissingMask[:,d]
                y    = X[nmy,d]
                ty   = nonmissingtype(eltype(y))
                y    = convert(Vector{ty},y)
                Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
                dmodel = deepcopy(estimators[imputation,d])
                #println(dmodel)
                #println(Xd)
                #println(y)
                #println(dmodel.hpar)
                #println(dmodel.par)
                fit!(dmodel,Xd,y)
                # imputing missing values in d...
                for i in 1:nR
                    if ! missingMask[i,d]
                        continue
                    end
                    xrow = permutedims(Vector(Xout[i,[1:(d-1);(d+1):end]]))
                    yest = predict(dmodel,xrow)
                    # handling some particualr cases... 
                    if typeof(yest) <: AbstractMatrix
                        yest = yest[1,1]
                    elseif typeof(yest) <: AbstractVector
                        yest = yest[1]
                    end
                    if typeof(yest) <: AbstractVector{<:AbstractDict}
                        yest = mode(yest[1],rng=rng)
                    elseif typeof(yest) <: AbstractDict
                        yest = mode(yest,rng=rng)
                    end
                    
                    if ty <: Int 
                        if typeof(yest) <: AbstractString
                            yest = parse(ty,yest)
                        elseif typeof(yest) <: Number
                            yest = Int(round(yest))
                        else
                            error("I don't know how to convert this type $(typeof(yest)) to an integer!")
                        end
                    end

                    Xout[i,d] = yest
                    #return Xout
                end
                # This is last passage: save the model and compute oob errors if requested
                if pass == recursive_passages 
                    estimators[imputation,d] = dmodel 
                end
            end # end dimension
        end # end recursive passage pass
        imputed[imputation]   = Xout
    end # end individual imputation
    m.par = UniversalImputerLearnableParameters(estimators)
    if cache
        if multiple_imputations == 1
            m.cres = imputed[1]
        else
            m.cres = imputed
        end
    end 
    m.info["n_imputed_values"] = n_imputed_values
    m.fitted = true
    return cache ? m.cres : nothing
end


"""
$(TYPEDSIGNATURES)

Return the data with the missing values replaced with the imputed ones using the non-linear structure learned fitting a [`UniversalImputer`](@ref) model.

# Notes:
- If `multiple_imputations` was set > 1 this is a vector of matrices (the individual imputations) instead of a single matrix.
"""
function predict(m::UniversalImputer,X)
    nR,nC = size(X)
    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    multiple_imputations  = m.hpar.multiple_imputations
    rng = m.opt.rng
    estimators = m.par.fittedModels
    verbosity = m.opt.verbosity

    imputed = fill(similar(X),multiple_imputations)
    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        for d in 1:nC
            verbosity >= FULL && println("  - processing dimension $d")
            nmy  = nonMissingMask[:,d]
            y    = X[nmy,d]
            ty   = nonmissingtype(eltype(y))
            y    = convert(Vector{ty},y)
            Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
            dmod = estimators[imputation,d]
            # imputing missing values in d...
            for i in 1:nR
                if ! missingMask[i,d]
                    continue
                end
                xrow = permutedims(Vector(Xout[i,[1:(d-1);(d+1):end]]))
                yest = predict(dmod,xrow)
                # handling some particualr cases... 
                if typeof(yest) <: AbstractMatrix
                    yest = yest[1,1]
                elseif typeof(yest) <: AbstractVector
                    yest = yest[1]
                end
                if typeof(yest) <: AbstractVector{<:AbstractDict}
                    yest = mode(yest[1],rng=rng)
                elseif typeof(yest) <: AbstractDict
                    yest = mode(yest,rng=rng)
                end
                
                if ty <: Int 
                    if typeof(yest) <: AbstractString
                        yest = parse(ty,yest)
                    elseif typeof(yest) <: Number
                        yest = Int(round(yest))
                    else
                        error("I don't know how to convert this type $(typeof(yest)) to an integer!")
                    end
                end
                
                Xout[i,d] = yest
                #return Xout
            end

        end # end dimension
        imputed[imputation]   = Xout
    end # end individual imputation
    multiple_imputations == 1 ? (return imputed[1]) : return imputed
end

function show(io::IO, ::MIME"text/plain", m::UniversalImputer)
    if m.fitted == false
        print(io,"UniversalImputer - A imputer based on an arbitrary regressor/classifier(unfitted)")
    else
        print(io,"UniversalImputer - A imputer based on an arbitrary regressor/classifier(unfitted) (fitted)")
    end
end

function show(io::IO, m::UniversalImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"UniversalImputer - A imputer based on an arbitrary regressor/classifier(unfitted) (unfitted)")
    else
        print(io,"UniversalImputer - A imputer based on an arbitrary regressor/classifier(unfitted) (fitted)")
        println(io,m.info)
    end
end

# MLJ interface
include("Imputation_MLJ.jl")

end # end Imputation module