export GMMRegressor1, GMMRegressor2
import BetaML.Utils.allowmissing!

# ------------------------------------------------------------------------------
# GMMRegressor1 

Base.@kwdef mutable struct GMMRegressor1LearnableParameters <: BetaMLLearnableParametersSet
    mixtures::Vector{AbstractMixture}              = []
    probMixtures::Vector{Float64}                  = []
    probRecords::Union{Nothing,Matrix{Float64}}    = nothing
    meanYByMixture::Union{Nothing,Matrix{Float64}} = nothing
end

"""
    GMMRegressor1

A multi-dimensional, missing data friendly non-linear regressor based on Generative (Gaussian) Mixture Model (strategy "1").

The training data is used to fit a probabilistic model with latent mixtures (Gaussian distributions with different covariances are already implemented) and then predictions of new data is obtained by fitting the new data to the mixtures.

For hyperparameters see [`GMMClusterHyperParametersSet`](@ref) and [`GMMClusterOptionsSet`](@ref).

this strategy (GMMRegressor1) works by training the EM algorithm on the feature matrix X.
Once the data has been probabilistically assigned to the various classes, a mean value of Y is computed for each cluster (using the probabilities as weigths).
At predict time, the new data is first fitted to the learned mixtures using the e-step part of the EM algorithm to obtain the probabilistic assignment of each record to the various mixtures. Then these probabilities are multiplied to the mixture averages for the Y dimensions learned at training time to obtain the predicted value(s) for each record. 

"""
mutable struct GMMRegressor1 <: BetaMLUnsupervisedModel
    hpar::GMMClusterHyperParametersSet
    opt::GMMClusterOptionsSet
    par::Union{Nothing,GMMRegressor1LearnableParameters}
    trained::Bool
    info::Dict{Symbol,Any}
end

function GMMRegressor1(;kwargs...)
    # ugly manual case...
    if (:nClasses in keys(kwargs) && ! (:mixtures in keys(kwargs)))
        nClasses = kwargs[:nClasses]
        hps = GMMClusterHyperParametersSet(nClasses = nClasses, mixtures = [DiagonalGaussian() for i in 1:nClasses])
    else 
        hps = GMMClusterHyperParametersSet()
    end
    m = GMMRegressor1(hps,GMMClusterOptionsSet(),GMMRegressor1LearnableParameters(),false,Dict{Symbol,Any}())
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
    fit!(m::GMMRegressor1,x)

## Notes:
`fit!` caches as record probabilities only those of the last set of data used to train the model
"""
function fit!(m::GMMRegressor1,x,y)

    x = makeMatrix(x)
    y = makeMatrix(y)

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

    if m.trained
        verbosity >= STD && @warn "Continuing training of a pre-trained model"
        gmmOut = gmm(x,K;p₀=m.par.probMixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy="given",maxIter=maxIter,rng = rng)
    else
        gmmOut = gmm(x,K;p₀=p₀,mixtures=mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,maxIter=maxIter,rng = rng)
    end

    probRecords    = gmmOut.pₙₖ
    sumProbrecords = sum(probRecords,dims=1)
    ysum           = probRecords' * y
    ymean          = vcat(transpose([ysum[r,:] / sumProbrecords[1,r] for r in 1:size(ysum,1)])...)

    m.par  = GMMRegressor1LearnableParameters(mixtures = gmmOut.mixtures, probMixtures=makeColVector(gmmOut.pₖ), probRecords = gmmOut.pₙₖ,meanYByMixture = ymean)

    m.info[:error]          = gmmOut.ϵ
    m.info[:lL]             = gmmOut.lL
    m.info[:BIC]            = gmmOut.BIC
    m.info[:AIC]            = gmmOut.AIC
    m.info[:trainedRecords] = get(m.info,:trainedRecords,0) + size(x,1)
    m.info[:dimensions]     = size(x,2)
    m.trained=true
    return true
end    

function predict(m::GMMRegressor1,X)
    X    = makeMatrix(X)
    N,DX = size(X)
    mixtures = m.par.mixtures
    yByMixture = m.par.meanYByMixture
    probMixtures = m.par.probMixtures
    probRecords, lL = estep(X,probMixtures,mixtures)
    return probRecords * yByMixture
end

function show(io::IO, ::MIME"text/plain", m::GMMRegressor1)
    if m.trained == false
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model (untrained)")
    else
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model (trained on $(m.info[:trainedRecords]) records)")
    end
end

function show(io::IO, m::GMMRegressor1)
    if m.trained == false
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model ($(m.hpar.nClasses) classes, untrained)")
    else
        print(io,"GMMRegressor1 - A regressor based on Generative Mixture Model ($(m.hpar.nClasses) classes, trained on $(m.info[:trainedRecords]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.probMixtures)
    end
end

# ------------------------------------------------------------------------------
# GMMRegressor2
"""
    GMMRegressor2

A multi-dimensional, missing data friendly non-linear regressor based on Generative (Gaussian) Mixture Model.

The training data is used to fit a probabilistic model with latent mixtures (Gaussian distributions with different covariances are already implemented) and then predictions of new data is obtained by fitting the new data to the mixtures.

For hyperparameters see [`GMMClusterHyperParametersSet`](@ref) and [`GMMClusterOptionsSet`](@ref).

Thsi strategy (GMMRegressor2) works by training the EM algorithm on a combined (hcat) matrix of X and Y.
At predict time, the new data is first fitted to the learned mixtures using the e-step part of the EM algorithm (and using missing values for the dimensions belonging to Y) to obtain the probabilistic assignment of each record to the various mixtures. Thes these probabilities are multiplied to the mixture averages for the Y dimensions to obtain the predicted value(s) for each record. 

"""
mutable struct GMMRegressor2 <: BetaMLUnsupervisedModel
    hpar::GMMClusterHyperParametersSet
    opt::GMMClusterOptionsSet
    par::Union{Nothing,GMMClusterLearnableParameters}
    trained::Bool
    info::Dict{Symbol,Any}
end

function GMMRegressor2(;kwargs...)
    # ugly manual case...
    if (:nClasses in keys(kwargs) && ! (:mixtures in keys(kwargs)))
        nClasses = kwargs[:nClasses]
        hps = GMMClusterHyperParametersSet(nClasses = nClasses, mixtures = [DiagonalGaussian() for i in 1:nClasses])
    else 
        hps = GMMClusterHyperParametersSet()
    end
    m = GMMRegressor2(hps,GMMClusterOptionsSet(),GMMClusterLearnableParameters(),false,Dict{Symbol,Any}())
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
    fit!(m::GMMRegressor2,x)

## Notes:
`fit!` caches as record probabilities only those of the last set of data used to train the model
"""
function fit!(m::GMMRegressor2,x,y)

    x = makeMatrix(x)
    y = makeMatrix(y)
    x = hcat(x,y)

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

    if m.trained
        verbosity >= STD && @warn "Continuing training of a pre-trained model"
        gmmOut = gmm(x,K;p₀=m.par.probMixtures,mixtures=m.par.mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy="given",maxIter=maxIter,rng = rng)
    else
        gmmOut = gmm(x,K;p₀=p₀,mixtures=mixtures,tol=tol,verbosity=verbosity,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,maxIter=maxIter,rng = rng)
    end
    m.par  = GMMClusterLearnableParameters(mixtures = gmmOut.mixtures, probMixtures=makeColVector(gmmOut.pₖ), probRecords = gmmOut.pₙₖ)

    m.info[:error]          = gmmOut.ϵ
    m.info[:lL]             = gmmOut.lL
    m.info[:BIC]            = gmmOut.BIC
    m.info[:AIC]            = gmmOut.AIC
    m.info[:trainedRecords] = get(m.info,:trainedRecords,0) + size(x,1)
    m.info[:dimensions]     = size(x,2)
    m.trained=true
    return true
end    

function predict(m::GMMRegressor2,X)
    X    = makeMatrix(X)
    allowmissing!(X)
    N,DX = size(X)
    mixtures = m.par.mixtures
    DFull    = length(mixtures[1].μ)
    K        = length(mixtures)
    X        = hcat(X,fill(missing,N,DFull-DX))
    yByMixture = [mixtures[k].μ[d] for k in 1:K, d in DX+1:DFull]
    probMixtures = m.par.probMixtures
    probRecords, lL = estep(X,probMixtures,mixtures)
    return probRecords * yByMixture
end

function show(io::IO, ::MIME"text/plain", m::GMMRegressor2)
    if m.trained == false
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model (untrained)")
    else
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model (trained on $(m.info[:trainedRecords]) records)")
    end
end

function show(io::IO, m::GMMRegressor2)
    if m.trained == false
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model ($(m.hpar.nClasses) classes, untrained)")
    else
        print(io,"GMMRegressor2 - A regressor based on Generative Mixture Model ($(m.hpar.nClasses) classes, trained on $(m.info[:trainedRecords]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.probMixtures)
    end
end