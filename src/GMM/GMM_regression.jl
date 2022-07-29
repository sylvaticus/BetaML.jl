export GMMRegressor1, GMMRegressor2
import BetaML.Utils.allowmissing!

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
    fit!(m::GMMRegressor1,x)

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
        print(io,"GMMClusterModel - A Generative Mixture Model (untrained)")
    else
        print(io,"GMMClusterModel - A Generative Mixture Model (trained on $(m.info[:trainedRecords]) records)")
    end
end

function show(io::IO, m::GMMRegressor2)
    if m.trained == false
        print(io,"GMMClusterModel - A $(m.hpar.nClasses)-classes Generative Mixture Model (untrained)")
    else
        print(io,"GMMClusterModel - A $(m.hpar.nClasses)-classes Generative Mixture Model(trained on $(m.info[:trainedRecords]) records)")
        println(io,m.info)
        println(io,"Mixtures:")
        println(io,m.par.mixtures)
        println(io,"Probability of each mixture:")
        println(io,m.par.probMixtures)
    end
end