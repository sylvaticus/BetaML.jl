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

- [`SimpleImputer`](@ref): Impute data using the feature (column) mean, optionally normalised by l-norms of the records (rows) (fastest)
- [`GaussianMixtureImputer`](@ref): Impute data using a Generative (Gaussian) Mixture Model (good trade off)
- [`RandomForestImputer`](@ref): Impute missing data using Random Forests, with optional replicable multiple imputations (most accurate).
- [`GeneralImputer`](@ref): Impute missing data using a vector (one per column) of arbitrary learning models (classifiers/regressors) that implement `m = Model([options])`, `fit!(m,X,Y)` and `predict(m,X)` (not necessarily from `BetaML`).


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

julia> mod          = RandomForestImputer(multiple_imputations=10,  rng=copy(FIXEDRNG));

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

import ..Trees: buildForest
import ..GMM: gmm, estep
import Base.print
import Base.show

#export predictMissing,
export SimpleImputerHyperParametersSet, RandomForestImputerHyperParametersSet,GeneralImputerHyperParametersSet,
       Imputer, SimpleImputer, GaussianMixtureImputer, RandomForestImputer, GeneralImputer
#fit!, predict, info

abstract type Imputer <: BetaMLModel end   

# ------------------------------------------------------------------------------
# SimpleImputer
"""
$(TYPEDEF)

Hyperparameters for the [`SimpleImputer`](@ref) model

# Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SimpleImputerHyperParametersSet <: BetaMLHyperParametersSet
    "The descriptive statistic of the column (feature) to use as imputed value [def: `mean`]"
    statistic::Function                   = mean
    "Normalise the feature mean by l-`norm` norm of the records [default: `nothing`]. Use it (e.g. `norm=1` to use the l-1 norm) if the records are highly heterogeneus (e.g. quantity exports of different countries)."
    norm::Union{Nothing,Int64}       = nothing
end
Base.@kwdef mutable struct SimpleImputerLearnableParameters <: BetaMLLearnableParametersSet
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

# Example:
```julia
julia> using BetaML

julia> X = [2.0 missing 10; 20 40 100]
2×3 Matrix{Union{Missing, Float64}}:
  2.0    missing   10.0
 20.0  40.0       100.0

julia> mod = SimpleImputer(norm=1)
SimpleImputer - A simple feature-stat based imputer (unfitted)

julia> X_full = fit!(mod,X)
2×3 Matrix{Float64}:
  2.0   4.04494   10.0
 20.0  40.0      100.0

julia> info(mod)
Dict{String, Any} with 1 entry:
  "n_imputed_values" => 1

julia> parameters(mod)
BetaML.Imputation.SimpleImputerLearnableParameters (a BetaMLLearnableParametersSet struct)
- cStats: [11.0, 40.0, 55.0]
- norms: [6.0, 53.333333333333336]
```

"""
mutable struct SimpleImputer <: Imputer
    hpar::SimpleImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,SimpleImputerLearnableParameters}
    cres::Union{Nothing,Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}
end

function SimpleImputer(;kwargs...)
    m              = SimpleImputer(SimpleImputerHyperParametersSet(),BetaMLDefaultOptionsSet(),SimpleImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit a matrix with missing data using [`SimpleImputer`](@ref)
"""
function fit!(imputer::SimpleImputer,X)
    (imputer.fitted == false ) || error("Multiple training unsupported on this model")
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
    imputer.par = SimpleImputerLearnableParameters(cStats,adjNorms)
    imputer.cres = cache ? X̂ : nothing
    imputer.info["n_imputed_values"] = sum(missingMask)
    imputer.fitted = true
    return cache ? imputer.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Predict the missing data using the feature averages (eventually normalised) learned by fitting a [`SimpleImputer`](@ref) model
"""
function predict(m::SimpleImputer,X)
    nR,nC = size(X)
    m.fitted || error()
    nC == length(m.par.cStats) || error("`SimpleImputer` can only predict missing values in matrices with the same number of columns as the matrice it has been trained with.")
    (m.hpar.norm == nothing || nR == length(m.par.norms)) || error("If norms are used, `SimpleImputer` can predict only matrices with the same number of rows as the matrix it has been trained with.")

    missingMask = ismissing.(X)
    if m.hpar.norm == nothing
        X̂ = [missingMask[r,c] ? m.par.cStats[c] : X[r,c] for r in 1:nR, c in 1:nC]
    else
        X̂        = [missingMask[r,c] ? m.par.cStats[c]*m.par.norms[r]/sum(m.par.norms) : X[r,c] for r in 1:nR, c in 1:nC]
    end
    return X̂
end

function show(io::IO, ::MIME"text/plain", m::SimpleImputer)
    if m.fitted == false
        print(io,"SimpleImputer - A simple feature-stat based imputer (unfitted)")
    else
        print(io,"SimpleImputer - A simple feature-stat based imputer (fitted)")
    end
end

function show(io::IO, m::SimpleImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"SimpleImputer - A simple feature-stat based imputer (unfitted)")
    else
        print(io,"SimpleImputer - A simple feature-stat based imputer (fitted)")
        println(io,m.info)
    end
end


# ------------------------------------------------------------------------------
# GaussianMixtureImputer
Base.@kwdef mutable struct GaussianMixtureImputerLearnableParameters <: BetaMLLearnableParametersSet
    mixtures::Union{Type,Vector{<: AbstractMixture}}    = DiagonalGaussian[] # The type is only temporary, it should always be replaced by an actual mixture
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

# Example: 
```julia
julia> using BetaML

julia> X = [1 2.5; missing 20.5; 0.8 18; 12 22.8; 0.4 missing; 1.6 3.7];

julia> mod = GaussianMixtureImputer(mixtures=[SphericalGaussian() for i in 1:2])
GaussianMixtureImputer - A Gaussian Mixture Model based imputer (unfitted)

julia> X_full = fit!(mod,X)
Iter. 1:        Var. of the post  2.373498171519511       Log-likelihood -29.111866299189792
6×2 Matrix{Float64}:
  1.0       2.5
  6.14905  20.5
  0.8      18.0
 12.0      22.8
  0.4       4.61314
  1.6       3.7

julia> info(mod)
Dict{String, Any} with 7 entries:
  "xndims"           => 2
  "error"            => [2.3735, 0.17527, 0.0283747, 0.0053147, 0.000981885]
  "AIC"              => 57.798
  "fitted_records"   => 6
  "lL"               => -21.899
  "n_imputed_values" => 2
  "BIC"              => 56.3403

julia> parameters(mod)
BetaML.Imputation.GaussianMixtureImputerLearnableParameters (a BetaMLLearnableParametersSet struct)
- mixtures: AbstractMixture[SphericalGaussian{Float64}([1.0179819950570768, 3.0999990977255845], 0.2865287884295908), SphericalGaussian{Float64}([6.149053737674149, 20.43331198167713], 15.18664378248651)]
- initial_probmixtures: [0.48544987084082347, 0.5145501291591764]
- probRecords: [0.9999996039918224 3.9600817749531375e-7; 2.3866922376272767e-229 1.0; … ; 0.9127030246369684 0.08729697536303167; 0.9999965964161501 3.403583849794472e-6]
```
"""
mutable struct GaussianMixtureImputer <: Imputer
    hpar::GMMHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{GaussianMixtureImputerLearnableParameters,Nothing}
    cres::Union{Nothing,Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}    
end

function GaussianMixtureImputer(;kwargs...)
    m   = GaussianMixtureImputer(GMMHyperParametersSet(),BetaMLDefaultOptionsSet(),GaussianMixtureImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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
    # Special correction for GMMHyperParametersSet
    kwkeys = keys(kwargs) #in(2,[1,2,3])
    if !in(:mixtures,kwkeys) && !in(:n_classes,kwkeys)
        m.hpar.n_classes = 3
        m.hpar.mixtures = [DiagonalGaussian() for i in 1:3]
    elseif !in(:mixtures,kwkeys) && in(:n_classes,kwkeys)
        m.hpar.mixtures = [DiagonalGaussian() for i in 1:kwargs[:n_classes]]
    elseif typeof(kwargs[:mixtures]) <: UnionAll && !in(:n_classes,kwkeys)
        m.hpar.n_classes = 3
        m.hpar.mixtures = [kwargs[:mixtures]() for i in 1:3]
    elseif typeof(kwargs[:mixtures]) <: UnionAll && in(:n_classes,kwkeys)
        m.hpar.mixtures = [kwargs[:mixtures]() for i in 1:kwargs[:n_classes]]
    elseif typeof(kwargs[:mixtures]) <: AbstractVector && !in(:n_classes,kwkeys)
        m.hpar.n_classes = length(kwargs[:mixtures])
    elseif typeof(kwargs[:mixtures]) <: AbstractVector && in(:n_classes,kwkeys)
        kwargs[:n_classes] == length(kwargs[:mixtures]) || error("The length of the mixtures vector must be equal to the number of classes")
    end
    return m
end


"""
$(TYPEDSIGNATURES)

Fit a matrix with missing data using [`GaussianMixtureImputer`](@ref)
"""
function fit!(m::GaussianMixtureImputer,X)
    
    # Parameter alias..
    K             = m.hpar.n_classes
    initial_probmixtures            = m.hpar.initial_probmixtures
    mixtures      = m.hpar.mixtures
    if  typeof(mixtures) <: UnionAll
        mixtures = [mixtures() for i in 1:K]
    end
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

    m.par  = GaussianMixtureImputerLearnableParameters(mixtures = emOut.mixtures, initial_probmixtures=makecolvector(emOut.pₖ), probRecords = emOut.pₙₖ)

    if cache
        X̂ = [XMask[n,d] ? X[n,d] : sum([emOut.mixtures[k].μ[d] * emOut.pₙₖ[n,k] for k in 1:K]) for n in 1:N, d in 1:D ]
        m.cres = X̂
    end

    m.info["error"]          = emOut.ϵ
    m.info["lL"]             = emOut.lL
    m.info["BIC"]            = emOut.BIC
    m.info["AIC"]            = emOut.AIC
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(X,1)
    m.info["xndims"]     = size(X,2)
    m.info["n_imputed_values"]     = n_imputed_values
    m.fitted=true
    return cache ? m.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Predict the missing data using the mixtures learned by fitting a [`GaussianMixtureImputer`](@ref) model

"""
function predict(m::GaussianMixtureImputer,X)
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

function show(io::IO, ::MIME"text/plain", m::GaussianMixtureImputer)
    if m.fitted == false
        print(io,"GaussianMixtureImputer - A Gaussian Mixture Model based imputer (unfitted)")
    else
        print(io,"GaussianMixtureImputer - A Gaussian Mixture Model based imputer (fitted)")
    end
end

function show(io::IO, m::GaussianMixtureImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GaussianMixtureImputer - A Gaussian Mixture Model based imputer (unfitted)")
    else
        print(io,"GaussianMixtureImputer - A Gaussian Mixture Model based imputer (fitted)")
        println(io,m.info)
    end
end

# ------------------------------------------------------------------------------
# RandomForestImputer

"""
$(TYPEDEF)

Hyperparameters for [`RandomForestImputer`](@ref)

# Parameters:
$(TYPEDFIELDS)

# Example:
```
julia>mod = RandomForestImputer(n_trees=20,max_depth=10,recursive_passages=3)
```
"""
Base.@kwdef mutable struct RandomForestImputerHyperParametersSet <: BetaMLHyperParametersSet
    "For the underlying random forest algorithm parameters (`n_trees`,`max_depth`,`min_gain`,`min_records`,`max_features:`,`splitting_criterion`,`β`,`initialisation_strategy`, `oob` and `rng`) see [`RFHyperParametersSet`](@ref) for the specific RF algorithm parameters"
    rfhpar                                      = RFHyperParametersSet()
    "Specify the positions of the integer columns to treat as categorical instead of cardinal. [Default: empty vector (all numerical cols are treated as cardinal by default and the others as categorical)]"
    forced_categorical_cols::Vector{Int64}                = Int64[] # like in RF, normally integers are considered ordinal
    "Define the times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`]."
    recursive_passages::Int64                    = 1
    "Determine the number of independent imputation of the whole dataset to make. Note that while independent, the imputations share the same random number generator (RNG)."
    multiple_imputations::Int64                  = 1
    "Columns in the matrix for which to create an imputation model, i.e. to impute. It can be a vector of columns IDs (positions), or the keywords \"auto\" (default) or \"all\". With \"auto\" the model automatically detects the columns with missing data and impute only them. You may manually specify the columns or use \"auto\" if you want to create a imputation model for that columns during training even if all training data are non-missing to apply then the training model to further data with possibly missing values."
    cols_to_impute::Union{String,Vector{Int64}} = "auto"
end

Base.@kwdef struct RandomForestImputerLearnableParameters <: BetaMLLearnableParametersSet
    forests               = nothing
    cols_to_impute_actual = Int64[] 
    #imputedValues  = nothing
    #n_imputed_values::Int64
    #oob::Vector{Vector{Float64}}
end

"""
$(TYPEDEF)

Impute missing data using Random Forests, with optional replicable multiple imputations. 

See [`RandomForestImputerHyperParametersSet`](@ref), [`RFHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref) for the parameters.

# Notes:
- Given a certain RNG and its status (e.g. `RandomForestImputer(...,rng=StableRNG(FIXEDSEED))`), the algorithm is completely deterministic, i.e. replicable. 
- The algorithm accepts virtually any kind of data, sortable or not

# Example:
```julia
julia> using BetaML

julia> X = [1.4 2.5 "a"; missing 20.5 "b"; 0.6 18 missing; 0.7 22.8 "b"; 0.4 missing "b"; 1.6 3.7 "a"]
6×3 Matrix{Any}:
 1.4        2.5       "a"
  missing  20.5       "b"
 0.6       18         missing
 0.7       22.8       "b"
 0.4         missing  "b"
 1.6        3.7       "a"

julia> mod = RandomForestImputer(n_trees=20,max_depth=10,recursive_passages=2)
RandomForestImputer - A Random-Forests based imputer (unfitted)

julia> X_full = fit!(mod,X)
** Processing imputation 1
6×3 Matrix{Any}:
 1.4        2.5     "a"
 0.504167  20.5     "b"
 0.6       18       "b"
 0.7       22.8     "b"
 0.4       20.0837  "b"
 1.6        3.7     "a"
```

"""
mutable struct RandomForestImputer <: Imputer
    hpar::RandomForestImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{RandomForestImputerLearnableParameters,Nothing}
    cres
    fitted::Bool
    info::Dict{String,Any}    
end

function RandomForestImputer(;kwargs...)
    
    hps = RandomForestImputerHyperParametersSet()
    m   = RandomForestImputer(hps,BetaMLDefaultOptionsSet(),RandomForestImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit a matrix with missing data using [`RandomForestImputer`](@ref)
"""
function fit!(m::RandomForestImputer,X)
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

    # determining cols_to_impute...
    if m.hpar.cols_to_impute == "auto"
        cols2imp = findall(i -> i==true, [any(ismissing.(c)) for c in eachcol(X)]) #  ismissing.(sum.(eachcol(X))))
    elseif m.hpar.cols_to_impute == "all"
        cols2imp = collect(1:size(X,2))
    else
        cols2imp = m.hpar.cols_to_impute
    end

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
    ooberrors      = fill(convert(Vector{Union{Missing,Float64}},fill(missing,nC)),multiple_imputations) # by imputations and dimensions
    forests        = Array{Trees.Forest}(undef,multiple_imputations,nC)

    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        sortedDims     = reverse(sortperm(makecolvector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        ooberrorsImputation = convert(Vector{Union{Missing,Float64}},fill(missing,nC))
        for pass in 1:recursive_passages 
            m.opt.verbosity >= HIGH && println("- processing passage $pass")
            if pass > 1
                shuffle!(rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                !(d in cols2imp) && continue
                verbosity >= FULL && println("  - processing dimension $d")
                if m.hpar.rfhpar.splitting_criterion == nothing
                    splitting_criterion = catCols[d] ?  gini : variance
                else
                    splitting_criterion = splitting_criterion
                end
                nmy  = nonMissingMask[:,d]
                y    = catCols[d] ? X[nmy,d] : identity.(X[nmy,d]) # witout the identity it remains any and force always a classification
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
    m.par = RandomForestImputerLearnableParameters(forests,cols2imp)
    if cache
        if multiple_imputations == 1
            m.cres = Utils.disallowmissing(imputed[1])
        else
            m.cres = Utils.disallowmissing.(imputed)
        end
    end 
    m.info["n_imputed_values"] = n_imputed_values
    m.info["oob_errors"] = ooberrors

    m.fitted = true
    return cache ? m.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Return the data with the missing values replaced with the imputed ones using the non-linear structure learned fitting a [`RandomForestImputer`](@ref) model.

# Notes:
- If `multiple_imputations` was set > 1 this is a vector of matrices (the individual imputations) instead of a single matrix.
"""
function predict(m::RandomForestImputer,X)
    nR,nC = size(X)
    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    multiple_imputations  = m.hpar.multiple_imputations
    rng = m.opt.rng
    forests = m.par.forests
    verbosity = m.opt.verbosity
    cols2imp = m.par.cols_to_impute_actual

    imputed = fill(similar(X),multiple_imputations)
    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        for d in 1:nC
            !(d in cols2imp) && continue
            verbosity >= FULL && println("  - processing dimension $d")
            dfor = forests[imputation,d]
            is_regression = dfor.is_regression
            nmy  = nonMissingMask[:,d]
            y    = is_regression ? identity.(X[nmy,d]) : X[nmy,d]
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
                    if ! is_regression
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
    multiple_imputations == 1 ? (return Utils.disallowmissing(imputed[1])) : return Utils.disallowmissing.(imputed)
end

function show(io::IO, ::MIME"text/plain", m::RandomForestImputer)
    if m.fitted == false
        print(io,"RandomForestImputer - A Random-Forests based imputer (unfitted)")
    else
        print(io,"RandomForestImputer - A Random-Forests based imputer (fitted)")
    end
end

function show(io::IO, m::RandomForestImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"RandomForestImputer - A Random-Forests based imputer (unfitted)")
    else
        print(io,"RandomForestImputer - A Random-Forests based imputer (fitted)")
        println(io,m.info)
    end
end

# ------------------------------------------------------------------------------
# GeneralImputer

"""
$(TYPEDEF)

Hyperparameters for [`GeneralImputer`](@ref)

# Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct GeneralImputerHyperParametersSet <: BetaMLHyperParametersSet
    "Columns in the matrix for which to create an imputation model, i.e. to impute. It can be a vector of columns IDs (positions), or the keywords \"auto\" (default) or \"all\". With \"auto\" the model automatically detects the columns with missing data and impute only them. You may manually specify the columns or use \"all\" if you want to create a imputation model for that columns during training even if all training data are non-missing to apply then the training model to further data with possibly missing values."
    cols_to_impute::Union{String,Vector{Int64}} = "auto"
    "An entimator model (regressor or classifier), with eventually its options (hyper-parameters), to be used to impute the various columns of the matrix. It can also be a `cols_to_impute`-length vector of different estimators to consider a different estimator for each column (dimension) to impute, for example when some columns are categorical (and will hence require a classifier) and some others are numerical (hence requiring a regressor). [default: `nothing`, i.e. use BetaML random forests, handling classification and regression jobs automatically]."
    estimator                        = nothing
    "Wheter the estimator(s) used to predict the missing data support itself missing data in the training features (X). If not, when the model for a certain dimension is fitted, dimensions with missing data in the same rows of those where imputation is needed are dropped and then only non-missing rows in the other remaining dimensions are considered. It can be a vector of boolean values to specify this property for each individual estimator or a single booleann value to apply to all the estimators [default: `false`]"
    missing_supported::Union{Vector{Bool},Bool} = false
    "The function used by the estimator(s) to fit the model. It should take as fist argument the model itself, as second argument a matrix representing the features, and as third argument a vector representing the labels. This parameter is mandatory for non-BetaML estimators and can be a single value or a vector (one per estimator) in case of different estimator packages used. [default: `BetaML.fit!`]"
    fit_function::Union{Vector{Function},Function}     = fit!
    "The function used by the estimator(s) to predict the labels. It should take as fist argument the model itself and as second argument a matrix representing the features. This parameter is mandatory for non-BetaML estimators and can be a single value or a vector (one per estimator) in case of different estimator packages used. [default: `BetaML.predict`]"
    predict_function::Union{Vector{Function},Function} = predict
    "Define the number of times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`]."
    recursive_passages::Int64      = 1
    "Determine the number of independent imputation of the whole dataset to make. Note that while independent, the imputations share the same random number generator (RNG)."
    multiple_imputations::Int64    = 1
end

Base.@kwdef struct GeneralImputerLearnableParameters <: BetaMLLearnableParametersSet
    fittedModels          = nothing         # by cols_to_imute only
    cols_to_impute_actual = Int64[] 
    x_used_cols           = Vector{Int64}[] # by all columns
    #imputedValues  = nothing
end

"""
$(TYPEDEF)

Impute missing values using arbitrary learning models.

Impute missing values using any arbitrary learning model (classifier or regressor, not necessarily from BetaML) that implement an interface `m = Model([options])`, `train!(m,X,Y)` and `predict(m,X)`. For non-BetaML supervised models the actual training and predict functions must be specified in the `fit_function` and `predict_function` parameters respectively.
If needed (for example when some columns with missing data are categorical and some numerical) different models can be specified for each column.
Multiple imputations and multiple "passages" trought the various colums for a single imputation are supported. 

See [`GeneralImputerHyperParametersSet`](@ref) for all the hyper-parameters.

# Examples:

- *Using BetaML models*:

```julia
julia> using BetaML
julia> X = [1.4 2.5 "a"; missing 20.5 "b"; 0.6 18 missing; 0.7 22.8 "b"; 0.4 missing "b"; 1.6 3.7 "a"]
6×3 Matrix{Any}:
 1.4        2.5       "a"
  missing  20.5       "b"
 0.6       18         missing
 0.7       22.8       "b"
 0.4         missing  "b"
 1.6        3.7       "a"

 julia> mod = GeneralImputer(recursive_passages=2,multiple_imputations=2)
 GeneralImputer - A imputer based on an arbitrary regressor/classifier(unfitted)

 julia> mX_full = fit!(mod,X);
 ** Processing imputation 1
 ** Processing imputation 2

 julia> mX_full[1]
 6×3 Matrix{Any}:
  1.4        2.5     "a"
  0.546722  20.5     "b"
  0.6       18       "b"
  0.7       22.8     "b"
  0.4       19.8061  "b"
  1.6        3.7     "a"

 julia> mX_full[2]
 6×3 Matrix{Any}:
  1.4        2.5     "a"
  0.554167  20.5     "b"
  0.6       18       "b"
  0.7       22.8     "b"
  0.4       20.7551  "b"
  1.6        3.7     "a"
  
 julia> info(mod)
 Dict{String, Any} with 1 entry:
   "n_imputed_values" => 3
 
```

- *Using third party packages* (in this example `DecisionTree`):

```julia
julia> using BetaML
julia> import DecisionTree
julia> X = [1.4 2.5 "a"; missing 20.5 "b"; 0.6 18 missing; 0.7 22.8 "b"; 0.4 missing "b"; 1.6 3.7 "a"]
6×3 Matrix{Any}:
 1.4        2.5       "a"
  missing  20.5       "b"
 0.6       18         missing
 0.7       22.8       "b"
 0.4         missing  "b"
 1.6        3.7       "a"
julia> mod = GeneralImputer(estimator=[DecisionTree.DecisionTreeRegressor(),DecisionTree.DecisionTreeRegressor(),DecisionTree.DecisionTreeClassifier()], fit_function = DecisionTree.fit!, predict_function=DecisionTree.predict, recursive_passages=2)
GeneralImputer - A imputer based on an arbitrary regressor/classifier(unfitted)
julia> X_full = fit!(mod,X)
** Processing imputation 1
6×3 Matrix{Any}:
 1.4    2.5  "a"
 0.94  20.5  "b"
 0.6   18    "b"
 0.7   22.8  "b"
 0.4   13.5  "b"
 1.6    3.7  "a"
```
"""
mutable struct GeneralImputer <: Imputer
    hpar::GeneralImputerHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{GeneralImputerLearnableParameters,Nothing}
    cres
    fitted::Bool
    info::Dict{String,Any}    
end

function GeneralImputer(;kwargs...)
    
    hps = GeneralImputerHyperParametersSet()
    m   = GeneralImputer(hps,BetaMLDefaultOptionsSet(),GeneralImputerLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit a matrix with missing data using [`GeneralImputer`](@ref)
"""
function fit!(m::GeneralImputer,X)
    nR,nC   = size(X)
    multiple_imputations  = m.hpar.multiple_imputations
    recursive_passages    = m.hpar.recursive_passages
    cache                = m.opt.cache
    verbosity            = m.opt.verbosity 
    rng                  = m.opt.rng

    # determining cols_to_impute...
    if m.hpar.cols_to_impute == "auto"
        cols2imp = findall(i -> i==true, [any(ismissing.(c)) for c in eachcol(X)]) #  ismissing.(sum.(eachcol(X))))
    elseif m.hpar.cols_to_impute == "all"
        cols2imp = collect(1:size(X,2))
    else
        cols2imp = m.hpar.cols_to_impute
    end

    nD2Imp = length(cols2imp)

    # Setting `estimators`, a matrix of multiple_imputations x nD2Imp individual models...
    if ! m.fitted
        if m.hpar.estimator == nothing
            estimators = [RandomForestEstimator(rng = m.opt.rng, verbosity=verbosity) for i in 1:multiple_imputations, d in 1:nD2Imp]
        elseif typeof(m.hpar.estimator) <: AbstractVector
            length(m.hpar.estimator) == nD2Imp || error("I can't use $(length(m.hpar.estimator)) estimators to impute $(nD2Imp) columns.")
            estimators = vcat([permutedims(deepcopy(m.hpar.estimator)) for i in 1:multiple_imputations]...)
        else # single estimator
            estimators = [deepcopy(m.hpar.estimator) for i in 1:multiple_imputations, j in 1:nD2Imp]
        end
    else
        m.opt.verbosity >= STD && @warn "This imputer has already been fitted. Not all learners support multiple training."
        estimators = m.par.fittedModels
    end

    missing_supported = typeof(m.hpar.missing_supported) <: AbstractArray ? m.hpar.missing_supported : fill(m.hpar.missing_supported,nD2Imp) 
    fit_functions = typeof(m.hpar.fit_function) <: AbstractArray ? m.hpar.fit_function : fill(m.hpar.fit_function,nD2Imp) 
    predict_functions = typeof(m.hpar.predict_function) <: AbstractArray ? m.hpar.predict_function : fill(m.hpar.predict_function,nD2Imp) 


    imputed = fill(similar(X),multiple_imputations)

    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    n_imputed_values = sum(missingMask)
    x_used_cols = [Int64[] for d in 1:size(X,2)]

    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout           = copy(X)
        sortedDims     = reverse(sortperm(makecolvector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        for pass in 1:recursive_passages
            Xout_passage = copy(Xout)
            m.opt.verbosity >= HIGH && println("- processing passage $pass")
            if pass > 1
                shuffle!(rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                !(d in cols2imp) && continue
                dIdx = findfirst(x -> x == d, cols2imp)
                verbosity >= FULL && println("  - processing dimension $d")
                msup = missing_supported[dIdx]
                if msup # missing is support, I consider all non-missing y rows and all dimensions..
                    nmy  = nonMissingMask[:,d]
                    y    = identity.(X[nmy,d]) # otherwise for some models it remains a classification
                    ty   = nonmissingtype(eltype(y))
                    y    = convert(Vector{ty},y)
                    Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
                    x_used_cols[d] = setdiff(collect(1:nC),d)
                else # missing is NOT supported, I consider only cols with nonmissing data in rows to impute and full rows in the remaining cols
                    nmy  = nonMissingMask[:,d]
                    # Step 1 removing cols with missing values in the rows that we will need to impute (i.e. that are also missing in the the y col)..
                    # I need to remove col and not row, as I need to impute this value, I can't just skip the row
                    candidates_d = setdiff(collect(1:nC),d)
                    for (ri,r) in enumerate(eachrow(Xout))
                        !nmy[ri] || continue # we want to look only where y is missing to remove cols 
                        for dc in candidates_d
                            if ismissing(r[dc])
                                candidates_d = setdiff(candidates_d,dc)
                            end
                        end
                    end
                    x_used_cols[d] = candidates_d
                    Xd = Xout[:,candidates_d]
                    # Step 2: for training, consider only the rows where not-dropped cols values are all nonmissing
                    nmxrows = [all(.! ismissing.(r)) for r in eachrow(Xd)]
                    nmrows = nmxrows .& nmy # non missing both in Y and remained X rows

                    y    = identity.(X[nmrows,d]) # otherwise for some models it remains a classification
                    ty   = nonmissingtype(eltype(y))
                    y    = convert(Vector{ty},y)
                    tX   = nonmissingtype(eltype(Xd))
                    Xd   = convert(Matrix{tX},Matrix(Xd[nmrows,:]))

                end
                dmodel = deepcopy(estimators[imputation,dIdx])
                fit_functions[dIdx](dmodel,Xd,y)

                # imputing missing values in d...
                for i in 1:nR
                    if ! missingMask[i,d]
                        continue
                    end
                    xrow = Vector(Xout[i,x_used_cols[d]])
                    if !msup # no missing supported, the row shoudn't contain missing values
                        xrow = Utils.disallowmissing(xrow)
                    end
                    yest = predict_functions[dIdx](dmodel,xrow)
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

                    Xout_passage[i,d] = yest
                    #return Xout
                end
                # This is last passage: save the model
                if pass == recursive_passages 
                    estimators[imputation,dIdx] = dmodel 
                end
            end # end dimension
            Xout = copy(Xout_passage)
        end # end recursive passage pass
        imputed[imputation]   = Xout
    end # end individual imputation
    m.par = GeneralImputerLearnableParameters(estimators,cols2imp,x_used_cols)
    if cache
        if multiple_imputations == 1
            m.cres = Utils.disallowmissing(imputed[1])
        else
            m.cres = Utils.disallowmissing.(imputed)
        end
    end 
    m.info["n_imputed_values"] = n_imputed_values
    m.fitted = true
    return cache ? m.cres : nothing
end


"""
$(TYPEDSIGNATURES)

Return the data with the missing values replaced with the imputed ones using the non-linear structure learned fitting a [`GeneralImputer`](@ref) model.

# Notes:
- if `multiple_imputations` was set > 1 this is a vector of matrices (the individual imputations) instead of a single matrix.
- due to the fact that the final models are fitted with already imputed values when multiple passages are emploied, these models can not be used to impute "new" matrices if they do not support themselves missing values. In this case, use `X̂new = fit!(m::GeneralImputer,Xnew)` instad of `fit!(m::GeneralImputer,X); X̂new = predict(m,Xnew)`.  
"""
function predict(m::GeneralImputer,X)
    cols2imp              = m.par.cols_to_impute_actual
    nD2Imp = length(cols2imp)
    missing_supported = typeof(m.hpar.missing_supported) <: AbstractArray ? m.hpar.missing_supported : fill(m.hpar.missing_supported,nD2Imp) 

    m.hpar.recursive_passages == 1 || all(missing_supported) || error("`predict(m::GeneralImputer,Xnew)` can not be used with multiple recursive passages in models that don't support missing values. Fit a new model for `Xnew` instead.")
    nR,nC                 = size(X)
    missingMask           = ismissing.(X)
    nonMissingMask        = .! missingMask 
    multiple_imputations  = m.hpar.multiple_imputations
    rng                   = m.opt.rng
    estimators            = m.par.fittedModels
    verbosity             = m.opt.verbosity
    
    x_used_cols           = m.par.x_used_cols

    

    predict_functions = typeof(m.hpar.predict_function) <: AbstractArray ? m.hpar.predict_function : fill(m.hpar.predict_function,nD2Imp) 

    imputed               = fill(similar(X),multiple_imputations)
    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout    = copy(X)
        for d in 1:nC
            !(d in cols2imp) && continue
            verbosity >= FULL && println("  - processing dimension $d")
            dIdx = findfirst(x -> x == d, cols2imp)
            msup = missing_supported[dIdx]
            nmy  = nonMissingMask[:,d]
            y    = X[nmy,d]
            ty   = nonmissingtype(eltype(y))
            y    = convert(Vector{ty},y)
            #Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
            dmod = estimators[imputation,dIdx]
            # imputing missing values in d...
            for i in 1:nR
                if ! missingMask[i,d]
                    continue
                end
                xrow = Vector(Xout[i,x_used_cols[d]])
                if !msup # no missing supported, the row shoudn't contain missing values
                    xrow = Utils.disallowmissing(xrow)
                end
                yest = predict_functions[dIdx](dmod,xrow)
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
    multiple_imputations == 1 ? (return Utils.disallowmissing(imputed[1])) : return Utils.disallowmissing.(imputed)
end

function show(io::IO, ::MIME"text/plain", m::GeneralImputer)
    if m.fitted == false
        print(io,"GeneralImputer - A imputer based on an arbitrary regressor/classifier(unfitted)")
    else
        print(io,"GeneralImputer - A imputer based on an arbitrary regressor/classifier(unfitted) (fitted)")
    end
end

function show(io::IO, m::GeneralImputer)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GeneralImputer - A imputer based on an arbitrary regressor/classifier(unfitted) (unfitted)")
    else
        print(io,"GeneralImputer - A imputer based on an arbitrary regressor/classifier(unfitted) (fitted)")
        println(io,m.info)
    end
end


end # end Imputation module