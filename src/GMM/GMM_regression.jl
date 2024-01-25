"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

import BetaML.Utils.allowmissing

# ------------------------------------------------------------------------------
# GaussianMixtureRegressor2 

Base.@kwdef mutable struct GaussianMixtureRegressor2_lp <: BetaMLLearnableParametersSet
    mixtures::Union{Type,Vector{<: AbstractMixture}}    = DiagonalGaussian[] # The type is only temporary, it should always be replaced by an actual mixture
    initial_probmixtures::Vector{Float64}                  = []
    #probRecords::Union{Nothing,Matrix{Float64}}    = nothing
    meanYByMixture::Union{Nothing,Matrix{Float64}} = nothing
end

"""
$(TYPEDEF)

A multi-dimensional, missing data friendly non-linear regressor based on Generative (Gaussian) Mixture Model (strategy "1").

The training data is used to fit a probabilistic model with latent mixtures (Gaussian distributions with different covariances are already implemented) and then predictions of new data is obtained by fitting the new data to the mixtures.

For hyperparameters see [`GaussianMixture_hp`](@ref) and [`BML_options`](@ref).

This strategy (`GaussianMixtureRegressor2`) works by fitting the EM algorithm on the feature matrix X.
Once the data has been probabilistically assigned to the various classes, a mean value of fitting values Y is computed for each cluster (using the probabilities as weigths).
At predict time, the new data is first fitted to the learned mixtures using the e-step part of the EM algorithm to obtain the probabilistic assignment of each record to the various mixtures. Then these probabilities are multiplied to the mixture averages for the Y dimensions learned at training time to obtain the predicted value(s) for each record. 

# Notes:
- Predicted values are always a matrix, even when a single variable is predicted (use `dropdims(ŷ,dims=2)` to get a single vector).

# Example:
```julia
julia> using BetaML

julia> X = [1.1 10.1; 0.9 9.8; 10.0 1.1; 12.1 0.8; 0.8 9.8];

julia> Y = X[:,1] .* 2 - X[:,2]
5-element Vector{Float64}:
 -7.8999999999999995
 -8.0
 18.9
 23.4
 -8.200000000000001

julia> mod = GaussianMixtureRegressor2(n_classes=2)
GaussianMixtureRegressor2 - A regressor based on Generative Mixture Model (unfitted)

julia> ŷ = fit!(mod,X,Y)
Iter. 1:        Var. of the post  2.15612140465882        Log-likelihood -29.06452054772657
5×1 Matrix{Float64}:
 -8.033333333333333
 -8.033333333333333
 21.15
 21.15
 -8.033333333333333

julia> new_probs = predict(mod,[11 0.9])
1×1 Matrix{Float64}:
 21.15

julia> info(mod)
Dict{String, Any} with 6 entries:
  "xndims"         => 2
  "error"          => [2.15612, 0.118848, 4.19495e-7, 0.0, 0.0]
  "AIC"            => 32.7605
  "fitted_records" => 5
  "lL"             => -7.38023
  "BIC"            => 29.2454
```

"""
mutable struct GaussianMixtureRegressor2 <: BetaMLUnsupervisedModel
    hpar::GaussianMixture_hp
    opt::BML_options
    par::Union{Nothing,GaussianMixtureRegressor2_lp}
    cres::Union{Nothing,Matrix{Float64}} 
    fitted::Bool
    info::Dict{String,Any}
end

function GaussianMixtureRegressor2(;kwargs...)
     m = GaussianMixtureRegressor2(GaussianMixture_hp(),BML_options(),GaussianMixtureRegressor2_lp(),nothing,false,Dict{Symbol,Any}())
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

    # Special correction for GaussianMixture_hp
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

Fit the [`GaussianMixtureRegressor2`](@ref) model to data

# Notes:
- re-fitting is a new complete fitting but starting with mixtures computed in the previous fitting(s)
"""
function fit!(m::GaussianMixtureRegressor2,x,y)

    m.fitted || autotune!(m,(x,y))
    
    x = makematrix(x)
    y = makematrix(y)

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

    if m.fitted
        verbosity >= STD && @warn "Continuing training of a pre-fitted model"
        gmmOut = gmm(x,K;initial_probmixtures=m.par.initial_probmixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy="given",maximum_iterations=maximum_iterations,rng = rng)
    else
        gmmOut = gmm(x,K;initial_probmixtures=initial_probmixtures,mixtures=mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy=initialisation_strategy,maximum_iterations=maximum_iterations,rng = rng)
    end

    probRecords    = gmmOut.pₙₖ
    sumProbrecords = sum(probRecords,dims=1)
    ysum           = probRecords' * y
    ymean          = vcat(transpose([ysum[r,:] / sumProbrecords[1,r] for r in 1:size(ysum,1)])...)

    m.par  = GaussianMixtureRegressor2_lp(mixtures = gmmOut.mixtures, initial_probmixtures=makecolvector(gmmOut.pₖ), meanYByMixture = ymean)
    m.cres = cache ? probRecords  * ymean : nothing


    m.info["error"]          = gmmOut.ϵ
    m.info["lL"]             = gmmOut.lL
    m.info["BIC"]            = gmmOut.BIC
    m.info["AIC"]            = gmmOut.AIC
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(x,1)
    m.info["xndims"]     = size(x,2)
    m.fitted=true
    return cache ? m.cres : nothing
end  

"""
$(TYPEDSIGNATURES)

Predict the classes probabilities associated to new data assuming the mixtures and average values per class computed in fitting a [`GaussianMixtureRegressor2`](@ref) model.

"""
function predict(m::GaussianMixtureRegressor2,X)
    X    = makematrix(X)
    N,DX = size(X)
    mixtures = m.par.mixtures
    yByMixture = m.par.meanYByMixture
    initial_probmixtures = m.par.initial_probmixtures
    probRecords, lL = estep(X,initial_probmixtures,mixtures)
    return probRecords * yByMixture
end

function show(io::IO, ::MIME"text/plain", m::GaussianMixtureRegressor2)
    if m.fitted == false
        print(io,"GaussianMixtureRegressor2 - A regressor based on Generative Mixture Model (unfitted)")
    else
        print(io,"GaussianMixtureRegressor2 - A regressor based on Generative Mixture Model (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, m::GaussianMixtureRegressor2)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GaussianMixtureRegressor2 - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, unfitted)")
    else
        print(io,"GaussianMixtureRegressor2 - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, fitted on $(m.info["fitted_records"]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.initial_probmixtures)
    end
end

# ------------------------------------------------------------------------------
# GaussianMixtureRegressor
"""
$(TYPEDEF)

A multi-dimensional, missing data friendly non-linear regressor based on Generative (Gaussian) Mixture Model.

The training data is used to fit a probabilistic model with latent mixtures (Gaussian distributions with different covariances are already implemented) and then predictions of new data is obtained by fitting the new data to the mixtures.

For hyperparameters see [`GaussianMixture_hp`](@ref) and [`BML_options`](@ref).

Thsi strategy (`GaussianMixtureRegressor`) works by training the EM algorithm on a combined (hcat) matrix of X and Y.
At predict time, the new data is first fitted to the learned mixtures using the e-step part of the EM algorithm (and using missing values for the dimensions belonging to Y) to obtain the probabilistic assignment of each record to the various mixtures. Then these probabilities are multiplied to the mixture averages for the Y dimensions to obtain the predicted value(s) for each record. 

# Example:
```julia
julia> using BetaML

julia> X = [1.1 10.1; 0.9 9.8; 10.0 1.1; 12.1 0.8; 0.8 9.8];

julia> Y = X[:,1] .* 2 - X[:,2]
5-element Vector{Float64}:
 -7.8999999999999995
 -8.0
 18.9
 23.4
 -8.200000000000001

julia> mod = GaussianMixtureRegressor(n_classes=2)
GaussianMixtureRegressor - A regressor based on Generative Mixture Model (unfitted)

julia> ŷ = fit!(mod,X,Y)
Iter. 1:        Var. of the post  2.2191120060614065      Log-likelihood -47.70971887023561
5×1 Matrix{Float64}:
 -8.033333333333333
 -8.033333333333333
 21.15
 21.15
 -8.033333333333333

julia> new_probs = predict(mod,[11 0.9])
1×1 Matrix{Float64}:
 21.15

julia> info(mod)
Dict{String, Any} with 6 entries:
  "xndims"         => 3
  "error"          => [2.21911, 0.0260833, 3.19141e-39, 0.0]
  "AIC"            => 60.0684
  "fitted_records" => 5
  "lL"             => -17.0342
  "BIC"            => 54.9911

julia> parameters(mod)
BetaML.GMM.GMMCluster_lp (a BetaMLLearnableParametersSet struct)
- mixtures: DiagonalGaussian{Float64}[DiagonalGaussian{Float64}([0.9333333333333332, 9.9, -8.033333333333333], [1.1024999999999996, 0.05, 5.0625]), DiagonalGaussian{Float64}([11.05, 0.9500000000000001, 21.15], [1.1024999999999996, 0.05, 5.0625])]
- initial_probmixtures: [0.6, 0.4]
```
"""
mutable struct GaussianMixtureRegressor <: BetaMLUnsupervisedModel
    hpar::GaussianMixture_hp
    opt::BML_options
    par::Union{Nothing,GMMCluster_lp}
    cres::Union{Nothing,Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}
end

function GaussianMixtureRegressor(;kwargs...)
    m = GaussianMixtureRegressor(GaussianMixture_hp(),BML_options(),GMMCluster_lp(),nothing,false,Dict{Symbol,Any}())
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
    # Special correction for GaussianMixture_hp
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

Fit the [`GaussianMixtureRegressor`](@ref) model to data

# Notes:
- re-fitting is a new complete fitting but starting with mixtures computed in the previous fitting(s)
"""
function fit!(m::GaussianMixtureRegressor,x,y)

    m.fitted || autotune!(m,(x,y))

    x = makematrix(x)
    N,DX = size(x)
    y = makematrix(y)
    x = hcat(x,y)
    DFull = size(x,2)
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

    if m.fitted
        verbosity >= HIGH && @info "Continuing training of a pre-fitted model"
        gmmOut = gmm(x,K;initial_probmixtures=m.par.initial_probmixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy="given",maximum_iterations=maximum_iterations,rng = rng)
    else
        gmmOut = gmm(x,K;initial_probmixtures=initial_probmixtures,mixtures=mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy=initialisation_strategy,maximum_iterations=maximum_iterations,rng = rng)
    end
    probRecords = gmmOut.pₙₖ
    m.par  = GMMCluster_lp(mixtures = gmmOut.mixtures, initial_probmixtures=makecolvector(gmmOut.pₖ))
    m.cres = cache ?  probRecords  * [gmmOut.mixtures[k].μ[d] for k in 1:K, d in DX+1:DFull]  : nothing

    m.info["error"]          = gmmOut.ϵ
    m.info["lL"]             = gmmOut.lL
    m.info["BIC"]            = gmmOut.BIC
    m.info["AIC"]            = gmmOut.AIC
    m.info["fitted_records"] = get(m.info,"fitted_records",0) + size(x,1)
    m.info["xndims"]     = size(x,2)
    m.fitted=true
    return cache ? m.cres : nothing
end    

"""
$(TYPEDSIGNATURES)

Predict the classes probabilities associated to new data assuming the mixtures computed fitting a [`GaussianMixtureRegressor`](@ref) model on a merged X and Y matrix
"""
function predict(m::GaussianMixtureRegressor,X)
    X    = makematrix(X)
    X    = allowmissing(X)
    N,DX = size(X)
    mixtures = m.par.mixtures
    DFull    = length(mixtures[1].μ)
    K        = length(mixtures)
    X        = hcat(X,fill(missing,N,DFull-DX))
    yByMixture = [mixtures[k].μ[d] for k in 1:K, d in DX+1:DFull]
    initial_probmixtures = m.par.initial_probmixtures
    probRecords, lL = estep(X,initial_probmixtures,mixtures)
    return probRecords * yByMixture
end

function show(io::IO, ::MIME"text/plain", m::GaussianMixtureRegressor)
    if m.fitted == false
        print(io,"GaussianMixtureRegressor - A regressor based on Generative Mixture Model (unfitted)")
    else
        print(io,"GaussianMixtureRegressor - A regressor based on Generative Mixture Model (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, m::GaussianMixtureRegressor)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GaussianMixtureRegressor - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, unfitted)")
    else
        print(io,"GaussianMixtureRegressor - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, fitted on $(m.info["fitted_records"]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.initial_probmixtures)
    end
end