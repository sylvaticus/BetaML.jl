"""
    Perceptron.jl file

Implement the BetaML.Perceptron module

`?BetaML.Perceptron` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Perceptron.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/lmlj.jl/blob/master/notebooks/Perceptron.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FPerceptron.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""

"""
    Perceptron module

Provide linear and kernel classifiers.

See a [runnable example on myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FPerceptron.ipynb)

- [`perceptron`](@ref): Train data using the classical perceptron
- [`kernelPerceptron`](@ref): Train data using the kernel perceptron
- [`pegasos`](@ref): Train data using the pegasos algorithm
- [`predict`](@ref): Predict data using parameters from one of the above algorithms

All algorithms are multiclass, with `perceptron` and `pegasos` employing a one-vs-all strategy, while `kernelPerceptron` employs a _one-vs-one_ approach, and return a "probability" for each class in term of a dictionary for each record. Use `mode(ŷ)` to return a single class prediction per record.

The binary equivalent algorithms, accepting only `{-1,+1}` labels, are available as `peceptronBinary`, `kernelPerceptronBinary` and `pegasosBinary`. They are slighly faster as they don't need to be wrapped in the multi-class equivalent and return a more informative output.

The multi-class versions are available in the MLJ framework as `PerceptronClassifier`,`KernelPerceptronClassifier` and `PegasosClassifier` respectivly.
"""
module Perceptron

using LinearAlgebra, Random, ProgressMeter, Reexport, CategoricalArrays

using ForceImport
@force using ..Api
@force using ..Utils


export perceptron, perceptronBinary, kernelPerceptron, kernelPerceptronBinary, pegasos, pegasosBinary, predict



# Todo (breaking): change the API so that a struct kernelPerceptronModel or linearModel is reported
# instead of a named tuple
# linearModel <: betaSupervisedModel <: betaModel

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
function perceptron(x::AbstractMatrix, y::AbstractVector; θ=zeros(size(x,2)),θ₀=0.0, T=1000, nMsgs=0, shuffle=false, forceOrigin=false, returnMeanHyperplane=false, rng = Random.GLOBAL_RNG)
    yclasses = unique(y)
    nCl      = length(yclasses)
    if nCl == 2
        outθ        = Array{Vector{Float64},1}(undef,1)
        outθ₀       = Array{Float64,1}(undef,1)
    else
        outθ        = Array{Vector{Float64},1}(undef,nCl)
        outθ₀       = Array{Float64,1}(undef,nCl)
    end
    for (i,c) in enumerate(yclasses)
        ybin = ((y .== c) .*2 .-1)  # conversion to -1/+1
        outBinary = perceptronBinary(x, ybin; θ=θ,θ₀=θ₀, T=T, nMsgs=nMsgs, shuffle=shuffle, forceOrigin=forceOrigin, rng=rng)
        if returnMeanHyperplane
            outθ[i]  = outBinary.avgθ
            outθ₀[i] = outBinary.avgθ₀
        else
            outθ[i]  = outBinary.θ
            outθ₀[i] = outBinary.θ₀
        end
        if nCl == 2
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
   kernelPerceptron(x,y;K,T,α,nMsgs,shuffle)

Train a multiclass kernel classifier "perceptron" algorithm based on x and y.

`kernelPerceptron` is a (potentially) non-linear perceptron-style classifier employing user-defined kernel funcions. Multiclass is supported using a one-vs-one approach.

# Parameters:
* `x`:        Feature matrix of the training data (n × d)
* `y`:        Associated labels of the training data, in the format of ⨦ 1
* `K`:        Kernel function to employ. See `?radialKernel` or `?polynomialKernel`for details or check `?BetaML.Utils` to verify if other kernels are defined (you can alsways define your own kernel) [def: [`radialKernel`](@ref)]
* `T`:        Maximum number of iterations (aka "epochs") across the whole set (if the set is not fully classified earlier) [def: 100]
* `α`:        Initial distribution of the errors [def: `zeros(length(y))`]
* `nMsg`:     Maximum number of messages to show if all iterations are done [def: `0`]
* `shuffle`:  Whether to randomly shuffle the data at each iteration [def: `false`]
* `rng`:      Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Return a named tuple with:
* `x`: The x data (eventually shuffled if `shuffle=true`)
* `y`: The label
* `α`: The errors associated to each record
* `classes`: The labels classes encountered in the training

# Notes:
* The trained model can then be used to make predictions using the function `predict()`.
* This model is available in the MLJ framework as the `KernelPerceptronClassifier`

# Example:
```jldoctest
julia> model  = kernelPerceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
julia> ŷtrain = Perceptron.predict(xtrain,model.x,model.y,model.α, model.classes,K=model.K)
julia> ϵtrain = error(ytrain, mode(ŷtrain))
```
"""
function kernelPerceptron(x, y; K=radialKernel, T=100, α=zeros(Int64,length(y)), nMsgs=0, shuffle=false, rng = Random.GLOBAL_RNG)
    x         = makeMatrix(x)
    yclasses  = unique(y)
    nCl       = length(yclasses)
    nModels   = Int((nCl  * (nCl - 1)) / 2)
    (n,d) = size(x)
    outX = Array{typeof(x),1}(undef,nModels)
    outY = Array{Array{Int64,1},1}(undef,nModels)
    outα = Array{Array{Int64,1},1}(undef,nModels)

    modelCounter = 1
    for (i,c) in enumerate(yclasses)
        for (i2,c2) in enumerate(yclasses)
            if i2 <= i continue end
            ids = ( (y .== c) .| (y .== c2) )
            thisx = x[ids,:]
            thisy = y[ids]
            thisα = α[ids]
            ybin = ((thisy .== c) .*2 .-1)  # conversion to +1 (if c) or -1 (if c2)
            outBinary = kernelPerceptronBinary(thisx, ybin; K=K, T=T, α=thisα, nMsgs=nMsgs, shuffle=shuffle, rng = rng)
            outX[modelCounter] = outBinary.x
            outY[modelCounter] = outBinary.y
            outα[modelCounter] = outBinary.α
            modelCounter += 1
        end
    end
    return (x=outX,y=outY,α=outα,classes=yclasses,K=K)
end

"""
   kernelPerceptronBinary(x,y;K,T,α,nMsgs,shuffle)

Train a multiclass kernel classifier "perceptron" algorithm based on x and y

# Parameters:
* `x`:        Feature matrix of the training data (n × d)
* `y`:        Associated labels of the training data, in the format of ⨦ 1
* `K`:        Kernel function to employ. See `?radialKernel` or `?polynomialKernel`for details or check `?BetaML.Utils` to verify if other kernels are defined (you can alsways define your own kernel) [def: [`radialKernel`](@ref)]
* `T`:        Maximum number of iterations across the whole set (if the set is not fully classified earlier) [def: 1000]
* `α`:        Initial distribution of the errors [def: `zeros(length(y))`]
* `nMsg`:     Maximum number of messages to show if all iterations are done
* `shuffle`:  Whether to randomly shuffle the data at each iteration [def: `false`]
* `rng`:      Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Return a named tuple with:
* `x`: the x data (eventually shuffled if `shuffle=true`)
* `y`: the label
* `α`: the errors associated to each record
* `errors`: the number of errors in the last iteration
* `besterrors`: the minimum number of errors in classifying the data ever reached
* `iterations`: the actual number of iterations performed
* `separated`: a flag if the data has been successfully separated

# Notes:
* The trained data can then be used to make predictions using the function `predict()`. **If the option `shuffle` has been used, it is important to use there the returned (x,y,α) as these would have been shuffle compared with the original (x,y)**.
* Please see @kernelPerceptron for a multi-class version
# Example:
```jldoctest
julia> model = kernelPerceptronBinary([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
```
"""
function kernelPerceptronBinary(x, y; K=radialKernel, T=1000, α=zeros(Int64,length(y)), nMsgs=10, shuffle=false, rng = Random.GLOBAL_RNG)
    if nMsgs != 0
        @codeLocation
        println("***\n*** Training kernel perceptron for maximum $T iterations. Random shuffle: $shuffle")
    end
    x = makeMatrix(x)
    (n,d) = size(x)
    bestϵ = Inf
    lastϵ = Inf
    @showprogress 1 "Training Kernel Perceptron..." for t in 1:T
        ϵ = 0
        if shuffle
           # random shuffle x, y and alpha
           ridx = Random.shuffle(rng, 1:size(x)[1])
           x = x[ridx, :]
           y = y[ridx]
           α = α[ridx]
        end
        for i in 1:n
            if y[i]*sum([α[j]*y[j]*K(x[j,:],x[i,:]) for j in 1:n]) <= 0 + eps()
                α[i] += 1
                ϵ += 1
            end
        end
        if (ϵ == 0)
            if nMsgs != 0
                println("*** Avg. error after epoch $t : $(ϵ/size(x)[1]) (all elements of the set has been correctly classified")
            end
            return (x=x,y=y,α=α,errors=0,besterrors=0,iterations=t,separated=true,K=K)
        elseif ϵ < bestϵ
            bestϵ = ϵ
        end
        lastϵ = ϵ
        if nMsgs != 0 && (t % ceil(T/nMsgs) == 0 || t == 1 || t == T)
          println("Avg. error after iteration $t : $(ϵ/size(x)[1])")
        end
    end
    return  (x=x,y=y,α=α,errors=lastϵ,besterrors=bestϵ,iterations=T,separated=false)
end


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
function pegasos(x, y; θ=zeros(size(x,2)),θ₀=0.0, λ=0.5,η= (t -> 1/sqrt(t)), T=1000, nMsgs=0, shuffle=false, forceOrigin=false,returnMeanHyperplane=false, rng = Random.GLOBAL_RNG)
    yclasses = unique(y)
    nCl      = length(yclasses)
    if nCl == 2
        outθ        = Array{Vector{Float64},1}(undef,1)
        outθ₀       = Array{Float64,1}(undef,1)
    else
        outθ        = Array{Vector{Float64},1}(undef,nCl)
        outθ₀       = Array{Float64,1}(undef,nCl)
    end
    for (i,c) in enumerate(yclasses)
        ybin = ((y .== c) .*2 .-1)  # conversion to -1/+1
        outBinary = pegasosBinary(x, ybin; θ=θ,θ₀=θ₀, λ=λ,η=η, T=T, nMsgs=nMsgs, shuffle=shuffle, forceOrigin=forceOrigin, rng=rng)
        if returnMeanHyperplane
            outθ[i]  = outBinary.avgθ
            outθ₀[i] = outBinary.avgθ₀
        else
            outθ[i]  = outBinary.θ
            outθ₀[i] = outBinary.θ₀
        end
        if nCl == 2
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



# ------------------------------------------------------------------------------
# Other functions

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


"""
  predict(x,xtrain,ytrain,α;K)

Predict a binary label {-1,1} given the feature vector and the training data together with their errors (as trained by a kernel perceptron algorithm)

# Parameters:
* `x`:      Feature matrix of the training data (n × d)
* `xtrain`: The feature vectors used for the training
* `ytrain`: The labels of the training set
* `α`:      The errors associated to each record
* `K`:      The kernel function used for the training and to be used for the prediction [def: [`radialKernel`](@ref)]

# Return :
* `y`: Vector of the predicted labels

# Example:
```julia
julia> predict([1.1 2.1; 5.3 4.2; 1.8 1.7], [3.2,1.2])
```
"""
function predict(x,xtrain,ytrain,α;K=radialKernel)
    x = makeMatrix(x)
    xtrain = makeMatrix(xtrain)
    (n,d) = size(x)
    (ntrain,d2) = size(xtrain)
    if (d2 != d) error("xtrain and x must have the same dimensions."); end
    if ( length(ytrain) != ntrain || length(α) != ntrain) error("xtrain, ytrain and α must all have the same length."); end
    y = zeros(Int64,n)
    for i in 1:n
        y[i] = sum([ α[j] * ytrain[j] * K(x[i,:],xtrain[j,:]) for j in 1:ntrain]) > eps() ? 1 : -1
    end
    return y
 end


 """
   predict(x,xtrain,ytrain,α,classes;K)

 Predict a multiclass label given the new feature vector and a trained kernel perceptron model.

 # Parameters:
 * `x`:      Feature matrix of the training data (n × d)
 * `xtrain`: A vector of the feature matrix used for training each of the one-vs-one class matches (i.e. `model.x`)
 * `ytrain`: A vector of the label vector used for training each of the one-vs-one class matches (i.e. `model.y`)
 * `α`:      A vector of the errors associated to each record (i.e. `model.α`)
 * `classes`: The overal classes encountered in training (i.e. `model.classes`)
 * `K`:      The kernel function used for the training and to be used for the prediction [def: [`radialKernel`](@ref)]

 # Return :
 * `ŷ`: Vector of dictionaries `label=>probability` (warning: it isn't really a probability, it is just the standardized number of matches "won" by this class compared with the other classes)

  # Notes:
  * Use `mode(ŷ)` if you want a single predicted label per record

 # Example:
 ```julia
 julia> model  = kernelPerceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
 julia> ŷtrain = Perceptron.predict([10 10; 2.2 2.5],model.x,model.y,model.α, model.classes,K=model.K)
 ```
 """
 function predict(x,xtrain,ytrain,α,classes::AbstractVector{Tcl};K=radialKernel) where {Tcl}
     (n,d)   = size(x)
     nCl     = length(classes)
     y       = Array{Dict{Tcl,Float64},1}(undef,n)
     nModels = Int((nCl  * (nCl - 1)) / 2)
     if !(nModels == length(xtrain) == length(ytrain) == length(α)) error("xtrain, ytrain or α have a length not compatible with the number of classes in this model."); end
     x = makeMatrix(x)
     d2 = size(xtrain[1],2)
     if (d2 != d) error("xtrain and x must have the same dimensions."); end
     for i in 1:n
         #countByClass = zeros(Float64,nCl)
         countByClass = zeros(Int64,nCl)
         mCounter = 1
          for (ic,c) in enumerate(classes)
             for (ic2,c2) in enumerate(classes)
                 if ic2 <= ic
                     continue
                 end
                 nThisModel = size(xtrain[mCounter],1)
                 if ( length(ytrain[mCounter]) != nThisModel || length(α[mCounter]) != nThisModel) error("xtrain, ytrain and α must all have the same length."); end
                 # note that we assign "winning" scores between pair of classes matches only based on who win, not by how much he did
                 # todo check
                 #if sum([ α[mCounter][j] * ((ytrain[mCounter][j] .== c) .*2 .-1) * K(x[i,:],xtrain[mCounter][j,:]) for j in 1:nThisModel]) > eps()
                 score = sum([ α[mCounter][j] * (ytrain[mCounter][j]) * K(x[i,:],xtrain[mCounter][j,:]) for j in 1:nThisModel])
                 #println(score)
                 if score > eps()
                     countByClass[ic] += 1
                     #countByClass[ic] += score
                 else
                     countByClass[ic2] += 1
                     #countByClass[ic2] += -score
                 end
                 mCounter += 1
             end
         end
         #println(countByClass)
         prob = softmax(countByClass)
         y[i] = Dict(zip(classes,prob))
    end
    return y
end

# MLJ interface
include("Perceptron_MLJ.jl")

end
