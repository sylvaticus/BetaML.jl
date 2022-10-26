"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
    perceptron(x,y;θ,θ₀,T,nMsgs,shuffle,force_origin,return_mean_hyperplane)

Train the multiclass classifier "perceptron" algorithm  based on x and y (labels).

!!! warning
    Direct usage of this low-level function is deprecated. It has been unexported in BetaML 0.9.
    Use the model [`PerceptronClassifier`](@ref) instead. 

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
* `force_origin`: Whether to force `θ₀` to remain zero [def: `false`]
* `return_mean_hyperplane`: Whether to return the average hyperplane coefficients instead of the final ones  [def: `false`]
* `rng`:         Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Return a named tuple with:
* `θ`:          The weights of the classifier
* `θ₀`:         The weight of the classifier associated to the constant term
* `classes`:    The classes (unique values) of y

# Notes:
* The trained parameters can then be used to make predictions using the function `predict()`.
* This model is available in the MLJ framework as the `LinearPerceptron`

# Example:
```jldoctest
julia> model = perceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
julia> ŷ     = predict([2.1 3.1; 7.3 5.2], model.θ, model.θ₀, model.classes)
```
"""
function perceptron(x::AbstractMatrix, y::AbstractVector; θ=nothing,θ₀=nothing, T=1000, nMsgs=0, shuffle=false, force_origin=false, return_mean_hyperplane=false, rng = Random.GLOBAL_RNG)
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
    outBinary = perceptronBinary(x, ybin; θ=θ[i],θ₀=θ₀[i], T=T, nMsgs=nMsgs, shuffle=shuffle, force_origin=force_origin, rng=rng)
    if return_mean_hyperplane
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
    perceptronBinary(x,y;θ,θ₀,T,nMsgs,shuffle,force_origin)

!!! warning
    Direct usage of this low-level function is deprecated. It has been unexported in BetaML 0.9.
    Use the model PerceptronClassifier() instead. 

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
* `force_origin`: Whether to force `θ₀` to remain zero [def: `false`]
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
function perceptronBinary(x, y; θ=zeros(size(x,2)),θ₀=0.0, T=1000, nMsgs=10, shuffle=false, force_origin=false, rng = Random.GLOBAL_RNG)
if nMsgs != 0
   @codelocation
   println("***\n*** Training perceptron for maximum $T iterations. Random shuffle: $shuffle")
end
x = makematrix(x)
(n,d) = size(x)
ny = size(y,1)
ny == n || error("y has different number of records (rows) than x!")
bestϵ = Inf
lastϵ = Inf
if force_origin θ₀ = 0.0; end
sumθ = θ; sumθ₀ = θ₀
@showprogress 1 "Training Perceptron..." for t in 1:T
   ϵ = 0
   if shuffle
      # random shuffle x and y
      ridx = Random.shuffle(rng, 1:size(x)[1])
      x = x[ridx, :]
      y = y[ridx]
   end
   @inbounds for i in 1:n
       if y[i]*(θ' * x[i,:] + θ₀) <= eps()
           θ  = θ + y[i] * x[i,:]
           θ₀ = force_origin ? 0.0 : θ₀ + y[i]
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

!!! warning
    Direct usage of this low-level function is deprecated. It has been unexported in BetaML 0.9.
    Use the `predict` function with your desired model instead. 

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
    x = makematrix(x)
    θ = makecolvector(θ)
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

!!! warning
    Direct usage of this low-level function is deprecated. It has been unexported in BetaML 0.9.
    Use the `predict` function of your desired model instead. 

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
$(TYPEDEF)

Hyperparameters for the [`PerceptronClassifier`](@ref) model

# Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct PerceptronClassifierHyperParametersSet <: BetaMLHyperParametersSet
    "Initial parameters. If given, should be a matrix of n-classes by feature dimension + 1 (to include the constant term as the first element) [def: `nothing`, i.e. zeros]"
    initial_parameters::Union{Nothing,Matrix{Float64}} = nothing
    "Maximum number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 1000
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true 
    "Whether to force the parameter associated with the constant term to remain zero [def: `false`]"
    force_origin::Bool = false
    "Whether to return the average hyperplane coefficients instead of the final ones  [def: `false`]"
    return_mean_hyperplane::Bool=false
    """
    The method - and its parameters - to employ for hyperparameters autotuning.
    See [`SuccessiveHalvingSearch`](@ref) for the default method.
    To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and eventually change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
    """
    tunemethod::AutoTuneMethod                  = SuccessiveHalvingSearch(hpranges=Dict("epochs" =>[50,100,1000,10000], "shuffle"=>[true,false], "force_origin"=>[true,false],"return_mean_hyperplane"=>[true,false]),multithreads=true)
end

Base.@kwdef mutable struct PerceptronClassifierLearnableParameters <: BetaMLLearnableParametersSet
    weigths::Union{Nothing,Matrix{Float64}} = nothing
    classes::Vector  = []
end

"""
$(TYPEDEF)

The classical "perceptron" linear classifier (supervised).

For the parameters see [`?PerceptronClassifierHyperParametersSet`](@ref PerceptronClassifierHyperParametersSet) and [`?BetaMLDefaultOptionsSet`](@ref BetaMLDefaultOptionsSet).

# Notes:
- data must be numerical
- online fitting (re-fitting with new data) is not supported

"""
mutable struct PerceptronClassifier <: BetaMLSupervisedModel
    hpar::PerceptronClassifierHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,PerceptronClassifierLearnableParameters}
    cres::Union{Nothing,Vector}
    fitted::Bool
    info::Dict{String,Any}
end

function PerceptronClassifier(;kwargs...)
    m              = PerceptronClassifier(PerceptronClassifierHyperParametersSet(),BetaMLDefaultOptionsSet(),PerceptronClassifierLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit the [`PerceptronClassifier`](@ref) model to data

"""
function fit!(m::PerceptronClassifier,X,Y)
    
    m.fitted || autotune!(m,(X,Y))

    # Parameter alias..
    initial_parameters             = m.hpar.initial_parameters
    epochs               = m.hpar.epochs
    shuffle              = m.hpar.shuffle
    force_origin          = m.hpar.force_origin
    return_mean_hyperplane = m.hpar.return_mean_hyperplane
    cache                = m.opt.cache
    verbosity            = m.opt.verbosity
    rng                  = m.opt.rng

    nR,nD    = size(X)
    yclasses = unique(Y)
    nCl      = length(yclasses)
    initial_parameters =  (initial_parameters == nothing) ? zeros(nCl, nD+1) : initial_parameters 
    
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

    out = perceptron(X,Y; θ₀=initial_parameters[:,1], θ=[initial_parameters[c,2:end] for c in 1:nCl], T=epochs, nMsgs=nMsgs, shuffle=shuffle, force_origin=force_origin, return_mean_hyperplane=return_mean_hyperplane, rng = rng)

    weights = hcat(out.θ₀,vcat(out.θ' ...))
    m.par = PerceptronClassifierLearnableParameters(weights,out.classes)
    if cache
       out    = predict(X,out.θ,out.θ₀,out.classes)
       m.cres = cache ? out : nothing
    end

    m.info["fitted_records"] = nR
    m.info["xndims"]    = nD
    m.info["n_classes"]      = size(weights,1)

    m.fitted = true

    return cache ? m.cres : nothing
end

"""
$(TYPEDSIGNATURES)

Predict the labels associated to some feature data using the linear coefficients learned by fitting a [`PerceptronClassifier`](@ref) model

"""
function predict(m::PerceptronClassifier,X)
    θ₀ = [i for i in m.par.weigths[:,1]]
    θ  = [r for r in eachrow(m.par.weigths[:,2:end])]
    return predict(X,θ,θ₀,m.par.classes)
end

function show(io::IO, ::MIME"text/plain", m::PerceptronClassifier)
    if m.fitted == false
        print(io,"PerceptronClassifier - The classic linear perceptron classifier (unfitted)")
    else
        print(io,"PerceptronClassifier - The classic linear perceptron classifier (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, m::PerceptronClassifier)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        println(io,"PerceptronClassifier - A $(m.info["xndims"])-dimensions $(m.info["n_classes"])-classes linear perceptron classifier (unfitted)")
    else
        println(io,"PerceptronClassifier - A $(m.info["xndims"])-dimensions $(m.info["n_classes"])-classes linear perceptron classifier (fitted on $(m.info["fitted_records"]) records)")
        println(io,"Weights:")
        println(io,m.par.weigths)
    end
end
