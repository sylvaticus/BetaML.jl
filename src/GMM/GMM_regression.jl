"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

import BetaML.Utils.allowmissing!

# ------------------------------------------------------------------------------
# GMMRegressor1 

Base.@kwdef mutable struct GMMRegressor1LearnableParameters <: BetaMLLearnableParametersSet
    mixtures::Vector{AbstractMixture}              = []
    initial_probmixtures::Vector{Float64}                  = []
    #probRecords::Union{Nothing,Matrix{Float64}}    = nothing
    meanYByMixture::Union{Nothing,Matrix{Float64}} = nothing
end

"""
$(TYPEDEF)

A multi-dimensional, missing data friendly non-linear regressor based on Generative (Gaussian) Mixture Model (strategy "1").

The training data is used to fit a probabilistic model with latent mixtures (Gaussian distributions with different covariances are already implemented) and then predictions of new data is obtained by fitting the new data to the mixtures.

For hyperparameters see [`GMMHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref).

This strategy (`GMMRegressor1`) works by fitting the EM algorithm on the feature matrix X.
Once the data has been probabilistically assigned to the various classes, a mean value of fitting values Y is computed for each cluster (using the probabilities as weigths).
At predict time, the new data is first fitted to the learned mixtures using the e-step part of the EM algorithm to obtain the probabilistic assignment of each record to the various mixtures. Then these probabilities are multiplied to the mixture averages for the Y dimensions learned at training time to obtain the predicted value(s) for each record. 

"""
mutable struct GMMRegressor1 <: BetaMLUnsupervisedModel
    hpar::GMMHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,GMMRegressor1LearnableParameters}
    cres::Union{Nothing,Matrix{Float64}} 
    fitted::Bool
    info::Dict{Symbol,Any}
end

function GMMRegressor1(;kwargs...)
    # ugly manual case...
    if (:n_classes in keys(kwargs) && ! (:mixtures in keys(kwargs)))
        n_classes = kwargs[:n_classes]
        hps = GMMHyperParametersSet(n_classes = n_classes, mixtures = [DiagonalGaussian() for i in 1:n_classes])
    else 
        hps = GMMHyperParametersSet()
    end
    m = GMMRegressor1(hps,BetaMLDefaultOptionsSet(),GMMRegressor1LearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit the [`GMMRegressor1`](@ref) model to data

# Notes:
- re-fitting is a new complete fitting but starting with mixtures computed in the previous fitting(s)
"""
function fit!(m::GMMRegressor1,x,y)

    x = makeMatrix(x)
    y = makeMatrix(y)

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

    m.par  = GMMRegressor1LearnableParameters(mixtures = gmmOut.mixtures, initial_probmixtures=makeColVector(gmmOut.pₖ), meanYByMixture = ymean)
    m.cres = cache ? probRecords  * ymean : nothing


    m.info[:error]          = gmmOut.ϵ
    m.info[:lL]             = gmmOut.lL
    m.info[:BIC]            = gmmOut.BIC
    m.info[:AIC]            = gmmOut.AIC
    m.info[:fitted_records] = get(m.info,:fitted_records,0) + size(x,1)
    m.info[:dimensions]     = size(x,2)
    m.fitted=true
    return cache ? m.cres : nothing
end  

"""
$(TYPEDSIGNATURES)

Predict the classes probabilities associated to new data assuming the mixtures and average values per class computed in fitting a [`GMMRegressor1`](@ref) model.

"""
function predict(m::GMMRegressor1,X)
    X    = makeMatrix(X)
    N,DX = size(X)
    mixtures = m.par.mixtures
    yByMixture = m.par.meanYByMixture
    initial_probmixtures = m.par.initial_probmixtures
    probRecords, lL = estep(X,initial_probmixtures,mixtures)
    return probRecords * yByMixture
end

function show(io::IO, ::MIME"text/plain", m::GMMRegressor1)
    if m.fitted == false
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model (unfitted)")
    else
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model (fitted on $(m.info[:fitted_records]) records)")
    end
end

function show(io::IO, m::GMMRegressor1)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, unfitted)")
    else
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, fitted on $(m.info[:fitted_records]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.initial_probmixtures)
    end
end

# ------------------------------------------------------------------------------
# GMMRegressor2
"""
$(TYPEDEF)

A multi-dimensional, missing data friendly non-linear regressor based on Generative (Gaussian) Mixture Model.

The training data is used to fit a probabilistic model with latent mixtures (Gaussian distributions with different covariances are already implemented) and then predictions of new data is obtained by fitting the new data to the mixtures.

For hyperparameters see [`GMMHyperParametersSet`](@ref) and [`GMMClusterOptionsSet`](@ref).

Thsi strategy (`GMMRegressor2`) works by training the EM algorithm on a combined (hcat) matrix of X and Y.
At predict time, the new data is first fitted to the learned mixtures using the e-step part of the EM algorithm (and using missing values for the dimensions belonging to Y) to obtain the probabilistic assignment of each record to the various mixtures. Then these probabilities are multiplied to the mixture averages for the Y dimensions to obtain the predicted value(s) for each record. 
"""
mutable struct GMMRegressor2 <: BetaMLUnsupervisedModel
    hpar::GMMHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,GMMClusterLearnableParameters}
    cres::Union{Nothing,Matrix{Float64}}
    fitted::Bool
    info::Dict{Symbol,Any}
end

function GMMRegressor2(;kwargs...)
    # ugly manual case...
    if (:n_classes in keys(kwargs) && ! (:mixtures in keys(kwargs)))
        n_classes = kwargs[:n_classes]
        hps = GMMHyperParametersSet(n_classes = n_classes, mixtures = [DiagonalGaussian() for i in 1:n_classes])
    else 
        hps = GMMHyperParametersSet()
    end
    m = GMMRegressor2(hps,BetaMLDefaultOptionsSet(),GMMClusterLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit the [`GMMRegressor2`](@ref) model to data

# Notes:
- re-fitting is a new complete fitting but starting with mixtures computed in the previous fitting(s)
"""
function fit!(m::GMMRegressor2,x,y)

    x = makeMatrix(x)
    N,DX = size(x)
    y = makeMatrix(y)
    x = hcat(x,y)
    DFull = size(x,2)
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

    if m.fitted
        verbosity >= STD && @warn "Continuing training of a pre-fitted model"
        gmmOut = gmm(x,K;initial_probmixtures=m.par.initial_probmixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy="given",maximum_iterations=maximum_iterations,rng = rng)
    else
        gmmOut = gmm(x,K;initial_probmixtures=initial_probmixtures,mixtures=mixtures,tol=tol,verbosity=verbosity,minimum_variance=minimum_variance,minimum_covariance=minimum_covariance,initialisation_strategy=initialisation_strategy,maximum_iterations=maximum_iterations,rng = rng)
    end
    probRecords = gmmOut.pₙₖ
    m.par  = GMMClusterLearnableParameters(mixtures = gmmOut.mixtures, initial_probmixtures=makeColVector(gmmOut.pₖ))
    m.cres = cache ?  probRecords  * [gmmOut.mixtures[k].μ[d] for k in 1:K, d in DX+1:DFull]  : nothing

    m.info[:error]          = gmmOut.ϵ
    m.info[:lL]             = gmmOut.lL
    m.info[:BIC]            = gmmOut.BIC
    m.info[:AIC]            = gmmOut.AIC
    m.info[:fitted_records] = get(m.info,:fitted_records,0) + size(x,1)
    m.info[:dimensions]     = size(x,2)
    m.fitted=true
    return cache ? m.cres : nothing
end    

"""
$(TYPEDSIGNATURES)

Predict the classes probabilities associated to new data assuming the mixtures computed fitting a [`GMMRegressor2`](@ref) model on a merged X and Y matrix
"""
function predict(m::GMMRegressor2,X)
    X    = makeMatrix(X)
    allowmissing!(X)
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

function show(io::IO, ::MIME"text/plain", m::GMMRegressor2)
    if m.fitted == false
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model (unfitted)")
    else
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model (fitted on $(m.info[:fitted_records]) records)")
    end
end

function show(io::IO, m::GMMRegressor2)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, unfitted)")
    else
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model ($(m.hpar.n_classes) classes, fitted on $(m.info[:fitted_records]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.initial_probmixtures)
    end
end