
"""
perceptron(x,y;θ,θ₀,T,nMsgs,shuffle,forceOrigin,returnMeanHyperplane)

Train the multiclass classifier "perceptron" algorithm  based on x and y (labels).

The perceptron is a _linear_ classifier. Multiclass is supported using a one-vs-all approach.

# Parameters:
* `x`:           Feature matrix of the training data (n × d)
* `y`:           Associated labels of the training data, can be in any format (string, integers..)
* `θ`:           Initial value of the weights (parameter) [def: `zeros(d)`]
* `θ₀`:          Initial value of the weight (parameter) associated to the constant
             term [def: `0`]
* `T`:           Maximum number of iterations across the whole set (if the set
             is not fully classified earlier) [def: 1000]
* `nMsg`:        Maximum number of messages to show if all iterations are done [def: `0`]
* `shuffle`:     Whether to randomly shuffle the data at each iteration [def: `false`]
* `forceOrigin`: Whether to force `θ₀` to remain zero [def: `false`]
* `returnMeanHyperplane`: Whether to return the average hyperplane coefficients instead of the final ones  [def: `false`]
* `rng`:         Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Return a named tuple with:
* `θ`:          The weights of the classifier
* `θ₀`:         The weight of the classifier associated to the constant term
* `classes`:    The classes (unique values) of y

# Notes:
* The trained parameters can then be used to make predictions using the function `predict()`.
* This model is available in the MLJ framework as the `PerceptronClassifier`

# Example:
```jldoctest
julia> model = perceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
julia> ŷ     = predict([2.1 3.1; 7.3 5.2], model.θ, model.θ₀, model.classes)
```
"""
function perceptron(x::AbstractMatrix, y::AbstractVector; θ=nothing,θ₀=nothing, T=1000, nMsgs=0, shuffle=false, forceOrigin=false, returnMeanHyperplane=false, rng = Random.GLOBAL_RNG)
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
    outBinary = perceptronBinary(x, ybin; θ=θ[i],θ₀=θ₀[i], T=T, nMsgs=nMsgs, shuffle=shuffle, forceOrigin=forceOrigin, rng=rng)
    if returnMeanHyperplane
        outθ[i]  = outBinary.avgθ
        outθ₀[i] = outBinary.avgθ₀
    else
        outθ[i]  = outBinary.θ
        outθ₀[i] = outBinary.θ₀
    end
    if i == 1 && nCl == 2
        outθ[2] = - outθ[1]
        outθ₀[2] = .- outθ₀[1]
        break    # if there are only two classes we do compute only one passage, as A vs B would be the same as B vs A
    end
end
return (θ=outθ,θ₀=outθ₀,classes=yclasses)
end

"""
perceptronBinary(x,y;θ,θ₀,T,nMsgs,shuffle,forceOrigin)

Train the binary classifier "perceptron" algorithm based on x and y (labels)

# Parameters:
* `x`:           Feature matrix of the training data (n × d)
* `y`:           Associated labels of the training data, in the format of ⨦ 1
* `θ`:           Initial value of the weights (parameter) [def: `zeros(d)`]
* `θ₀`:          Initial value of the weight (parameter) associated to the constant
             term [def: `0`]
* `T`:           Maximum number of iterations across the whole set (if the set
             is not fully classified earlier) [def: 1000]
* `nMsg`:        Maximum number of messages to show if all iterations are done
* `shuffle`:     Whether to randomly shuffle the data at each iteration [def: `false`]
* `forceOrigin`: Whether to force `θ₀` to remain zero [def: `false`]
* `rng`:         Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

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
julia> model = perceptronBinary([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
```
"""
function perceptronBinary(x, y; θ=zeros(size(x,2)),θ₀=0.0, T=1000, nMsgs=10, shuffle=false, forceOrigin=false, rng = Random.GLOBAL_RNG)
if nMsgs != 0
   @codeLocation
   println("***\n*** Training perceptron for maximum $T iterations. Random shuffle: $shuffle")
end
x = makeMatrix(x)
(n,d) = size(x)
bestϵ = Inf
lastϵ = Inf
if forceOrigin θ₀ = 0.0; end
sumθ = θ; sumθ₀ = θ₀
@showprogress 1 "Training Perceptron..." for t in 1:T
   ϵ = 0
   if shuffle
      # random shuffle x and y
      ridx = Random.shuffle(rng, 1:size(x)[1])
      x = x[ridx, :]
      y = y[ridx]
   end
   for i in 1:n
       if y[i]*(θ' * x[i,:] + θ₀) <= eps()
           θ  = θ + y[i] * x[i,:]
           θ₀ = forceOrigin ? 0.0 : θ₀ + y[i]
           sumθ += θ; sumθ₀ += θ₀
           ϵ += 1
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


"""
  predict(x,θ,θ₀)

Predict a binary label {-1,1} given the feature vector and the linear coefficients

# Parameters:
* `x`:        Feature matrix of the training data (n × d)
* `θ`:        The trained parameters
* `θ₀`:       The trained bias barameter [def: `0`]

# Return :
* `y`: Vector of the predicted labels

# Example:
```julia
julia> predict([1.1 2.1; 5.3 4.2; 1.8 1.7], [3.2,1.2])
```
"""
function predict(x,θ,θ₀=0.0)
    x = makeMatrix(x)
    θ = makeColVector(θ)
    (n,d) = size(x)
    d2 = length(θ)
    if (d2 != d) error("x and θ must have the same dimensions."); end
    y = zeros(Int64,n)
    for i in 1:n
        y[i] = (θ' * x[i,:] + θ₀) > eps() ? 1 : -1  # no need to divide by the norm to get the sign!
    end
    return y
end


"""
  predict(x,θ,θ₀,classes)

Predict a multiclass label given the feature vector, the linear coefficients and the classes vector

# Parameters:
* `x`:       Feature matrix of the training data (n × d)
* `θ`:       Vector of the trained parameters for each one-vs-all model (i.e. `model.θ`)
* `θ₀`:      Vector of the trained bias barameter for each one-vs-all model (i.e. `model.θ₀`)
* `classes`: The overal classes encountered in training (i.e. `model.classes`)

# Return :
* `ŷ`: Vector of dictionaries `label=>probability`

 # Notes:
 * Use `mode(ŷ)` if you want a single predicted label per record

# Example:
```julia
julia> model  = perceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
julia> ŷtrain = predict([10 10; 2.5 2.5],model.θ,model.θ₀, model.classes)
"""
function predict(x,θ::AbstractVector{T},θ₀::AbstractVector{Float64},classes::Vector{Tcl}) where {T<: AbstractVector{Float64},Tcl}
    (n,d) = size(x)
    nCl   = length(classes)
    y     = Array{Dict{Tcl,Float64},1}(undef,n)
    for i in 1:n
        probRaw = Array{Float64,1}(undef,nCl)
        for (c,cl) in enumerate(classes)
            if nCl == 2 && c ==2
                probRaw[2] = - probRaw[1]
            else
                probRaw[c] = (θ[c]' * x[i,:] + θ₀[c])
            end
        end
        prob = softmax(probRaw)
        y[i] = Dict(zip(classes,prob))
    end
    return y
end


# ----------------------------------------------
# API V2...

"""
**`$(TYPEDEF)`**

Hyperparameters for the `PerceptronClassic` model

## Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct PerceptronClassicHyperParametersSet <: BetaMLHyperParametersSet
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

Base.@kwdef mutable struct PerceptronClassicLearnableParameters <: BetaMLLearnableParametersSet
    weigths::Union{Nothing,Matrix{Float64}} = nothing
    classes::Vector  = []
end

"""

**`PerceptronClassic`**

The classical "perceptron" linear classifier (supervised).

For the parameters see  [`PerceptronClassicLearnableParameters`](@ref).

## Limitations:
- data must be numerical
- online training (retraining) not supported

"""
mutable struct PerceptronClassic <: BetaMLSupervisedModel
    hpar::PerceptronClassicHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,PerceptronClassicLearnableParameters}
    cres::Union{Nothing,Vector}
    fitted::Bool
    info::Dict{Symbol,Any}
end

function PerceptronClassic(;kwargs...)
    m              = PerceptronClassic(PerceptronClassicHyperParametersSet(),BetaMLDefaultOptionsSet(),PerceptronClassicLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

function fit!(m::PerceptronClassic,X,Y)
    

    # Parameter alias..
    initPars             = m.hpar.initPars
    epochs               = m.hpar.epochs
    shuffle              = m.hpar.shuffle
    forceOrigin          = m.hpar.forceOrigin
    returnMeanHyperplane = m.hpar.returnMeanHyperplane
    cache                = m.opt.cache
    verbosity            = m.opt.verbosity
    rng                  = m.opt.rng

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

    out = perceptron(X,Y; θ₀=initPars[:,1], θ=[initPars[:,c] for c in 2:nD+1], T=epochs, nMsgs=nMsgs, shuffle=shuffle, forceOrigin=forceOrigin, returnMeanHyperplane=returnMeanHyperplane, rng = rng)

    weights = hcat(out.θ₀,vcat(out.θ' ...))
    m.par = PerceptronClassicLearnableParameters(weights,out.classes)
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

function predict(m::PerceptronClassic,X)
    θ₀ = [ i for i in m.par.weigths[:,1]]
    θ  = [r for r in eachrow(m.par.weigths[:,2:end])]
    return predict(X,θ,θ₀,m.par.classes)
end

function show(io::IO, ::MIME"text/plain", m::PerceptronClassic)
    if m.fitted == false
        print(io,"PerceptronClassic - The classic linear perceptron classifier (unfitted)")
    else
        print(io,"PerceptronClassic - The classic linear perceptron classifier (fitted on $(m.info[:fittedRecords]) records)")
    end
end

function show(io::IO, m::PerceptronClassic)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        println(io,"PerceptronClassic - A $(m.info[:dimensions])-dimensions $(m.info[:nClasses])-classes linear perceptron classifier (unfitted)")
    else
        println(io,"PerceptronClassic - A $(m.info[:dimensions])-dimensions $(m.info[:nClasses])-classes linear perceptron classifier (fitted on $(m.info[:fittedRecords]) records)")
        println(io,"Weights:")
        println(io,m.par.weights)
    end
end
