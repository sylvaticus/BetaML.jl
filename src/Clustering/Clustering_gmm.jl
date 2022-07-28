
"""
estep(X,pₖ,mixtures)

E-step: assign the posterior prob p(j|xi) and computing the log-Likelihood of the parameters given the set of data(this last one for informative purposes and terminating the algorithm only)

"""
function estep(X,pₖ,mixtures)
 (N,D)  = size(X)
 K      = length(mixtures)
 Xmask  = .! ismissing.(X)
 logpₙₖ = zeros(N,K)
 lL     = 0
 for n in 1:N
     if any(Xmask[n,:]) # if at least one true
         Xu    = X[n,Xmask[n,:]]
         logpx = lse([log(pₖ[k] + 1e-16) + lpdf(mixtures[k],Xu,Xmask[n,:]) for k in 1:K])
         lL += logpx
         for k in 1:K
             logpₙₖ[n,k] = log(pₖ[k] + 1e-16)+lpdf(mixtures[k],Xu,Xmask[n,:])-logpx
         end
     else
         logpₙₖ[n,:] = log.(pₖ)
     end
 end
 pₙₖ = exp.(logpₙₖ)
 return (pₙₖ,lL)
end



## The gmm algorithm (Lecture/segment 16.5 of https://www.edx.org/course/machine-learning-with-python-from-linear-models-to)

# no longer true with the numerical trick implemented
# - For mixtures with full covariance matrix (i.e. `FullGaussian(μ,σ²)`) the minCovariance should NOT be set equal to the minVariance, or if the covariance matrix goes too low, it will become singular and not invertible.
"""
gmm(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance,initStrategy)

Compute Expectation-Maximisation algorithm to identify K clusters of X data, i.e. employ a Generative Mixture Model as the underlying probabilistic model.

X can contain missing values in some or all of its dimensions. In such case the learning is done only with the available data.
Implemented in the log-domain for better numerical accuracy with many dimensions.

# Parameters:
* `X`  :           A (n x d) data to clusterise
* `K`  :           Number of cluster wanted
* `p₀` :           Initial probabilities of the categorical distribution (K x 1) [default: `[]`]
* `mixtures`:      An array (of length K) of the mixture to employ (see notes) [def: `[DiagonalGaussian() for i in 1:K]`]
* `tol`:           Tolerance to stop the algorithm [default: 10^(-6)]
* `verbosity`:     A verbosity parameter regulating the information messages frequency [def: `STD`]
* `minVariance`:   Minimum variance for the mixtures [default: 0.05]
* `minCovariance`: Minimum covariance for the mixtures with full covariance matrix [default: 0]. This should be set different than minVariance (see notes).
* `initStrategy`:  Mixture initialisation algorithm [def: `kmeans`]
* `maxIter`:       Maximum number of iterations [def: `typemax(Int64)`, i.e. ∞]
* `rng`:           Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Returns:
* A named touple of:
* `pₙₖ`:      Matrix of size (N x K) of the probabilities of each point i to belong to cluster j
* `pₖ`:       Probabilities of the categorical distribution (K x 1)
* `mixtures`: Vector (K x 1) of the estimated underlying distributions
* `ϵ`:        Vector of the discrepancy (matrix norm) between pⱼₓ and the lagged pⱼₓ at each iteration
* `lL`:       The log-likelihood (without considering the last mixture optimisation)
* `BIC`:      The Bayesian Information Criterion (lower is better)
* `AIC`:      The Akaike Information Criterion (lower is better)

# Notes:
- The mixtures currently implemented are `SphericalGaussian(μ,σ²)`,`DiagonalGaussian(μ,σ²)` and `FullGaussian(μ,σ²)`
- Reasonable choices for the minVariance/Covariance depends on the mixture. For example 0.25 seems a reasonable value for the SphericalGaussian, 0.05 seems better for the DiagonalGaussian, and FullGaussian seems to prefer either very low values of variance/covariance (e.g. `(0.05,0.05)` ) or very big but similar ones (e.g. `(100,100)` ).
- For `initStrategy`, look at the documentation of `initMixtures!` for the mixture you want. The provided gaussian mixtures support `grid`, `kmeans` or `given`. `grid` is faster (expecially if X contains missing values), but `kmeans` often provides better results.

# Resources:
- [Paper describing gmm with missing values](https://doi.org/10.1016/j.csda.2006.10.002)
- [Class notes from MITx 6.86x (Sec 15.9)](https://stackedit.io/viewer#!url=https://github.com/sylvaticus/MITx_6.86x/raw/master/Unit 04 - Unsupervised Learning/Unit 04 - Unsupervised Learning.md)
- [Limitations of gmm](https://www.r-craft.org/r-news/when-not-to-use-gaussian-mixture-model-gmm-clustering/)

# Example:
```julia
julia> clusters = gmm([1 10.5;1.5 0; 1.8 8; 1.7 15; 3.2 40; 0 0; 3.3 38; 0 -2.3; 5.2 -2.4],3,verbosity=HIGH)
```
"""
function gmm(X,K;p₀=Float64[],mixtures=[DiagonalGaussian() for i in 1:K],tol=10^(-6),verbosity=STD,minVariance=0.05,minCovariance=0.0,initStrategy="kmeans",maxIter=typemax(Int64),rng = Random.GLOBAL_RNG)
 if verbosity > STD
     @codeLocation
 end
 # debug:
 #X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
 #K = 3
 #p₀=nothing; tol=0.0001; msgStep=1; minVariance=0.25; initStrategy="grid"
 #mixtures = [SphericalGaussian() for i in 1:K]
 # ---------
 X     = makeMatrix(X)
 (N,D) = size(X)
 pₖ    = isempty(p₀) ? fill(1/K,K) : p₀

 # no longer true with the numerical trick implemented
 #if (minVariance == minCovariance)
 #    @warn("Setting the minVariance equal to the minCovariance may lead to singularity problems for mixtures with full covariance matrix.")
 #end

 msgStepMap = Dict(NONE => 0, LOW=>100, STD=>20, HIGH=>5, FULL=>1)
 msgStep    = msgStepMap[verbosity]


 # Initialisation of the parameters of the mixtures
 mixtures = identity.(mixtures) # to set the container to the minimum common denominator of element types
 initMixtures!(mixtures,X,minVariance=minVariance,minCovariance=minCovariance,initStrategy=initStrategy,rng=rng)

 pₙₖ = zeros(Float64,N,K) # The posteriors, i.e. the prob that item n belong to cluster k
 ϵ = Float64[]

 # Checking dimensions only once (but adding then inbounds doesn't change anything. Still good
 # to provide a nice informative message)
 if size(pₖ,1) != K || length(mixtures) != K
     error("Error in the dimensions of the inputs. Please check them.")
 end

 # finding empty/non_empty values
 #Xmask     =  .! ismissing.(X)


 lL = -Inf
 iter = 1
 while(true)
     oldlL = lL
     # E Step: assigning the posterior prob p(j|xi) and computing the log-Likelihood of the parameters given the set of data
     pₙₖlagged = copy(pₙₖ)
     pₙₖ, lL = estep(X,pₖ,mixtures) 
     push!(ϵ,norm(pₙₖlagged - pₙₖ))

     # M step: find parameters that maximise the likelihood
     # Updating the probabilities of the different mixtures
     nₖ = sum(pₙₖ,dims=1)'
     n  = sum(nₖ)
     pₖ = nₖ ./ n
     updateParameters!(mixtures, X, pₙₖ; minVariance=minVariance,minCovariance=minCovariance)

     # Information. Note the likelihood is whitout accounting for the new mu, sigma
     if msgStep != 0 && (length(ϵ) % msgStep == 0 || length(ϵ) == 1)
         println("Iter. $(length(ϵ)):\tVar. of the post  $(ϵ[end]) \t  Log-likelihood $(lL)")
     end

     # Closing conditions. Note that the logLikelihood is those without considering the new mu,sigma
     if ((lL - oldlL) <= (tol * abs(lL))) || (iter >= maxIter)
         npars = npar(mixtures) + (K-1)
         #BIC  = lL - (1/2) * npars * log(N)
         BICv = bic(lL,npars,N)
         AICv = aic(lL,npars)
     #if (ϵ[end] < tol)
        return (pₙₖ=pₙₖ,pₖ=pₖ,mixtures=mixtures,ϵ=ϵ,lL=lL,BIC=BICv,AIC=AICv)
    else
         iter += 1
    end
 end # end while loop
end # end function

#  - For mixtures with full covariance matrix (i.e. `FullGaussian(μ,σ²)`) the minCovariance should NOT be set equal to the minVariance, or if the covariance matrix goes too low, it will become singular and not invertible.
"""
predictMissing(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance)

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

# Avi v2..


Base.@kwdef mutable struct GMMClusterHyperParametersSet <: BetaMLHyperParametersSet
    nClasses::Int64                   = 3
    probMixtures::Vector{Float64}     = []
    mixtures::Vector{AbstractMixture} = [DiagonalGaussian() for i in 1:nClasses]
    tol::Float64                      = 10^(-6)
    minVariance::Float64              = 0.05
    minCovariance::Float64            = 0.0
    initStrategy::String              = "kmeans"
    maxIter::Int64                    = typemax(Int64)
end

Base.@kwdef mutable struct GMMClusterOptionsSet <: BetaMLOptionsSet
    verbosity::Verbosity = STD
    rng                  = Random.GLOBAL_RNG
end

Base.@kwdef mutable struct GMMClusterLearnableParameters <: BetaMLLearnableParametersSet
    mixtures::Vector{AbstractMixture}           = []
    probMixtures::Vector{Float64}               = []
    probRecords::Union{Nothing,Matrix{Float64}} = nothing
end



mutable struct GMMClusterModel <: BetaMLUnsupervisedModel
    hpar::GMMClusterHyperParametersSet
    opt::GMMClusterOptionsSet
    par::Union{Nothing,GMMClusterLearnableParameters}
    trained::Bool
    info::Dict{Symbol,Any}
end

function GMMClusterModel(;kwargs...)
    m = GMMClusterModel(GMMClusterHyperParametersSet(),GMMClusterOptionsSet(),GMMClusterLearnableParameters(),false,Dict{Symbol,Any}())
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
    train!(m::GMMClusterModel,x)

## Notes:
`train!` caches as record probabilities only those of the last set of data used to train the model
"""
function train!(m::GMMClusterModel,x)

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

function predict(m::GMMClusterModel)
    return m.par.probRecords
end

function predict(m::GMMClusterModel,X)
    X = makeMatrix(X)
    mixtures = m.par.mixtures
    probMixtures = m.par.probMixtures
    probRecords, lL = estep(X,probMixtures,mixtures)
    return probRecords
end

function show(io::IO, ::MIME"text/plain", m::GMMClusterModel)
    if m.trained == false
        print(io,"GMMClusterModel - A Generative Mixture Model (untrained)")
    else
        print(io,"GMMClusterModel - A Generative Mixture Model (trained on $(m.info[:trainedRecords]) records)")
    end
end

function show(io::IO, m::GMMClusterModel)
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