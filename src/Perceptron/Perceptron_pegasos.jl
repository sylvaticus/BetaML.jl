
"""
pegasos(x,y;θ,θ₀,λ,η,T,nMsgs,shuffle,forceOrigin,returnMeanHyperplane)

Train the multiclass classifier "pegasos" algorithm according to x (features) and y (labels)

Pegasos is a _linear_, gradient-based classifier. Multiclass is supported using a one-vs-all approach.

# Parameters:
* `x`:           Feature matrix of the training data (n × d)
* `y`:           Associated labels of the training data, can be in any format (string, integers..)
* `θ`:           Initial value of the weights (parameter) [def: `zeros(d)`]
* `θ₀`:          Initial value of the weight (parameter) associated to the constant term [def: `0`]
* `λ`:           Multiplicative term of the learning rate
* `η`:           Learning rate [def: (t -> 1/sqrt(t))]
* `T`:           Maximum number of iterations across the whole set (if the set is not fully classified earlier) [def: 1000]
* `nMsg`:        Maximum number of messages to show if all iterations are done
* `shuffle`:     Whether to randomly shuffle the data at each iteration [def: `false`]
* `forceOrigin`: Whehter to force `θ₀` to remain zero [def: `false`]
* `returnMeanHyperplane`: Whether to return the average hyperplane coefficients instead of the average ones  [def: `false`]
* `rng`:         Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Return a named tuple with:
* `θ`:          The weights of the classifier
* `θ₀`:         The weight of the classifier associated to the constant term
* `classes`:    The classes (unique values) of y

# Notes:
* The trained parameters can then be used to make predictions using the function `predict()`.
* This model is available in the MLJ framework as the `PegasosClassifier`

# Example:
```jldoctest
julia> model = pegasos([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
julia> ŷ     = predict([2.1 3.1; 7.3 5.2], model.θ, model.θ₀, model.classes)
```
"""
function pegasos(x, y; θ=nothing,θ₀=nothing, λ=0.5,η= (t -> 1/sqrt(t)), T=1000, nMsgs=0, shuffle=false, forceOrigin=false,returnMeanHyperplane=false, rng = Random.GLOBAL_RNG)
yclasses = unique(y)
nCl      = length(yclasses)
nD       = size(x,2)
#if nCl == 2
#    outθ        = Array{Vector{Float64},1}(undef,1)
#    outθ₀       = Array{Float64,1}(undef,1)
#else
    outθ        = Array{Vector{Float64},1}(undef,nCl)
    outθ₀       = Array{Float64,1}(undef,nCl)
#end

if θ₀ == nothing
    θ₀ = zeros(nCl)
end

if θ == nothing
    θ = [zeros(nD) for _ in 1:nCl]
end

for (i,c) in enumerate(yclasses)
    ybin = ((y .== c) .*2 .-1)  # conversion to -1/+1
    outBinary = pegasosBinary(x, ybin; θ=θ[i],θ₀=θ₀[i], λ=λ,η=η, T=T, nMsgs=nMsgs, shuffle=shuffle, forceOrigin=forceOrigin, rng=rng)
    if returnMeanHyperplane
        outθ[i]  = outBinary.avgθ
        outθ₀[i] = outBinary.avgθ₀
    else
        outθ[i]  = outBinary.θ
        outθ₀[i] = outBinary.θ₀
    end
    if nCl == 2
        outθ[2] = - outθ[1]
        outθ₀[2] = .- outθ₀[1]
        break    # if there are only two classes we do compute only one passage, as A vs B would be the same as B vs A
    end
end
return (θ=outθ,θ₀=outθ₀,classes=yclasses)
end



"""
pegasosBinary(x,y;θ,θ₀,λ,η,T,nMsgs,shuffle,forceOrigin)

Train the peagasos algorithm based on x and y (labels)

# Parameters:
* `x`:           Feature matrix of the training data (n × d)
* `y`:           Associated labels of the training data, in the format of ⨦ 1
* `θ`:           Initial value of the weights (parameter) [def: `zeros(d)`]
* `θ₀`:          Initial value of the weight (parameter) associated to the constant term [def: `0`]
* `λ`:           Multiplicative term of the learning rate
* `η`:           Learning rate [def: (t -> 1/sqrt(t))]
* `T`:           Maximum number of iterations across the whole set (if the set is not fully classified earlier) [def: 1000]
* `nMsg`:        Maximum number of messages to show if all iterations are done
* `shuffle`:    Whether to randomly shuffle the data at each iteration [def: `false`]
* `forceOrigin`: Whether to force `θ₀` to remain zero [def: `false`]

# Return a named tuple with:
* `θ`:          The final weights of the classifier
* `θ₀`:         The final weight of the classifier associated to the constant term
* `avgθ`:       The average weights of the classifier
* `avgθ₀`:      The average weight of the classifier associated to the constant term
* `errors`:     The number of errors in the last iteration
* `besterrors`: The minimum number of errors in classifying the data ever reached
* `iterations`: The actual number of iterations performed
* `separated`:  Weather the data has been successfully separated

# Notes:
* The trained parameters can then be used to make predictions using the function `predict()`.

# Example:
```jldoctest
julia> pegasos([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
```
"""
function pegasosBinary(x, y; θ=zeros(size(x,2)),θ₀=0.0, λ=0.5,η= (t -> 1/sqrt(t)), T=1000, nMsgs=10, shuffle=false, forceOrigin=false, rng = Random.GLOBAL_RNG)
if nMsgs != 0
  @codeLocation
  println("***\n*** Training pegasos for maximum $T iterations. Random shuffle: $shuffle")
end
x = makeMatrix(x)
(n,d) = size(x)
bestϵ = Inf
lastϵ = Inf
if forceOrigin θ₀ = 0.0; end
sumθ = θ; sumθ₀ = θ₀
@showprogress 1 "Training Pegasos..." for t in 1:T
  ϵ = 0
  ηₜ = η(t)
  if shuffle
     # random shuffle x and y
     ridx = Base.shuffle(rng, 1:size(x)[1])
     x = x[ridx, :]
     y = y[ridx]
  end
  for i in 1:n
      if y[i]*(θ' * x[i,:] + θ₀) <= eps()
          θ  = (1-ηₜ*λ) * θ + ηₜ * y[i] * x[i,:]
          θ₀ = forceOrigin ? 0.0 : θ₀ + ηₜ * y[i]
          sumθ += θ; sumθ₀ += θ₀
          ϵ += 1
      else
          θ  = (1-ηₜ*λ) * θ
      end
  end
  if (ϵ == 0)
      if nMsgs != 0
          println("*** Avg. error after epoch $t : $(ϵ/size(x)[1]) (all elements of the set has been correctly classified")
      end
      return (θ=θ,θ₀=θ₀,avgθ=sumθ/(n*T),avgθ₀=sumθ₀/(n*T),errors=0,besterrors=0,iterations=t,separated=true)
  elseif ϵ < bestϵ
      bestϵ = ϵ
  end
  lastϵ = ϵ
  if nMsgs != 0 && (t % ceil(T/nMsgs) == 0 || t == 1 || t == T)
    println("Avg. error after iteration $t : $(ϵ/size(x)[1])")
  end
end
return  (θ=θ,θ₀=θ₀,avgθ=sumθ/(n*T),avgθ₀=sumθ₀/(n*T),errors=lastϵ,besterrors=bestϵ,iterations=T,separated=false)
end

# ----------------------------------------------
# API V2...

"""
**`$(TYPEDEF)`**

Hyperparameters for the `Pegasos` model

## Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct PegasosHyperParametersSet <: BetaMLHyperParametersSet
    "Learning rate [def: (epoch -> 1/sqrt(epoch))]"
    learningRate::Function =  (epoch -> 1/sqrt(epoch)) 
    "Multiplicative term of the learning rate [def: `0.5`]"         
    learningRateMultiplicative::Float64 = 0.5           
    "Initial parameters. If given, should be a matrix of n-classes by feature dimension + 1 (to include the constant term as the first element) [def: `nothing`, i.e. zeros]"
    initPars::Union{Nothing,Matrix{Float64}} = nothing
    "Maximum number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 1000
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `false`]"
    shuffle::Bool = false  
    "Whether to force the parameter associated with the constant term to remain zero [def: `false`]"
    forceOrigin::Bool = false
    " Whether to return the average hyperplane coefficients instead of the final ones  [def: `false`]"
    returnMeanHyperplane::Bool=false
end

Base.@kwdef mutable struct PegasosLearnableParameters <: BetaMLLearnableParametersSet
    weigths::Union{Nothing,Matrix{Float64}} = nothing
    classes::Vector  = []
end

mutable struct Pegasos <: BetaMLSupervisedModel
    hpar::PegasosHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,PegasosLearnableParameters}
    cres::Union{Nothing,Vector}
    fitted::Bool
    info::Dict{Symbol,Any}
end

function Pegasos(;kwargs...)
    m              = Pegasos(PegasosHyperParametersSet(),BetaMLDefaultOptionsSet(),PegasosLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

function fit!(m::Pegasos,X,Y)
    

    # Parameter alias..
    learningRate               = m.hpar.learningRate
    learningRateMultiplicative = m.hpar.learningRateMultiplicative
    initPars                   = m.hpar.initPars
    epochs                     = m.hpar.epochs
    shuffle                    = m.hpar.shuffle
    forceOrigin                = m.hpar.forceOrigin
    returnMeanHyperplane       = m.hpar.returnMeanHyperplane
    cache                      = m.opt.cache
    verbosity                  = m.opt.verbosity
    rng                        = m.opt.rng

    nR,nD    = size(X)
    yclasses = unique(Y)
    nCl      = length(yclasses)
    initPars =  (initPars == nothing) ? zeros(nCl, nD+1) : initPars 
    
    if verbosity == NONE
        nMsgs = 0
    elseif verbosity <= LOW
        nMsgs = 5
    elseif verbosity <= STD
        nMsgs = 10
    elseif verbosity <= HIGH
        nMsgs = 100
    else
        nMsgs = 100000
    end

    out = pegasos(X,Y; θ₀=initPars[:,1], θ=[initPars[:,c] for c in 2:nD+1], λ=learningRateMultiplicative, η=learningRate, T=epochs, nMsgs=nMsgs, shuffle=shuffle, forceOrigin=forceOrigin, returnMeanHyperplane=returnMeanHyperplane, rng = rng)

    weights = hcat(out.θ₀,vcat(out.θ' ...))
    m.par = PegasosLearnableParameters(weights,out.classes)
    if cache
       out    = predict(X,out.θ,out.θ₀,out.classes)
       m.cres = cache ? out : nothing
    end

    m.info[:fittedRecords] = nR
    m.info[:dimensions]    = nD
    m.info[:nClasses]      = size(weights,1)

    m.fitted = true

    return true
end

function predict(m::Pegasos,X)
    θ₀ = [ i for i in m.par.weigths[:,1]]
    θ  = [r for r in eachrow(m.par.weigths[:,2:end])]
    return predict(X,θ,θ₀,m.par.classes)
end

function show(io::IO, ::MIME"text/plain", m::Pegasos)
    if m.fitted == false
        print(io,"Pegasos - a loss-based linear classifier without regularisation term (unfitted)")
    else
        print(io,"Pegasos - a loss-based linear classifier without regularisation term (fitted on $(m.info[:fittedRecords]) records)")
    end
end

function show(io::IO, m::Pegasos)
    if m.fitted == false
        println(io,"Pegasos - A $(m.info[:dimensions])-dimensions $(m.info[:nClasses])-classes a loss-based linear classifier without regularisation term (unfitted)")
    else
        println(io,"Pegasos - A $(m.info[:dimensions])-dimensions $(m.info[:nClasses])-classes a loss-based linear classifier without regularisation term (fitted on $(m.info[:fittedRecords]) records)")
        println(io,"Weights:")
        println(io,m.par.weights)
    end
end
