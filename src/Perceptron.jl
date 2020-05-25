

module Perceptron

using LinearAlgebra, Random, ProgressMeter
import ..Utils: radialKernel, polynomialKernel, makeMatrix, makeColVector, error, accuracy

export perceptron, kernelPerceptron, pegasus, predict, radialKernel, polynomialKernel,
       makeMatrix, error, accuracy



"""
  perceptron(x,y;θ,θ₀,T,nMsgs,rShuffle,forceOrigin)

Train a perceptron algorithm based on x and y (labels)

# Parameters:
* `x`:           Feature matrix of the training data (n × d)
* `y`:           Associated labels of the training data, in the format of ⨦ 1
* `θ`:           Initial value of the weights (parameter) [def: `zeros(d)`]
* `θ₀`:          Initial value of the weight (parameter) associated to the constant
                 term [def: `0`]
* `T`:           Maximum number of iterations across the whole set (if the set
                 is not fully classified earlier) [def: 1000]
* `nMsg`:        Maximum number of messages to show if all iterations are done
* `rShuffle`:    Wheter to randomly shuffle the data at each iteration [def: `false`]
* `forceOrigin`: Wheter to force `θ₀` to remain zero [def: `false`]

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
* The trained data can then be used to make predictions using the function
 `predict()`. **If the option `randomShuffle` has been used, it is important to
  use there the returned (x,y,α) as these would have been shuffle compared with
  the original (x,y)**.

# Example:
julia> perceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
"""
function perceptron(x, y; θ=zeros(size(x,2)),θ₀=0.0, T=1000, nMsgs=10, rShuffle=false, forceOrigin=false)
   if nMsgs != 0
       println("***\n*** Training perceptron for maximum $T iterations. Random shuffle: $rShuffle")
   end
   x = makeMatrix(x)
   (n,d) = size(x)
   bestϵ = Inf
   lastϵ = Inf
   if forceOrigin θ₀ = 0.0; end
   sumθ = θ; sumθ₀ = θ₀
   @showprogress 1 "Training Perceptron..." for t in 1:T
       ϵ = 0
       if rShuffle
          # random shuffle x and y
          ridx = shuffle(1:size(x)[1])
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
   kernelPerceptron(x,y;K,T,α,nMsgs,rShuffle)

Train a Kernel Perceptron algorithm based on x and y

# Parameters:
* `x`:        Feature matrix of the training data (n × d)
* `y`:        Associated labels of the training data, in the format of ⨦ 1
* `K`:        Kernel function to emplpy. See `?radialKernel` or `?polynomialKernel`
              for details or check `?Bmlt.Utils` to verify if other kernels are
              defined (you  can alsways define your own kernel) [def: `radialKernel`]
* `T`:        Maximum number of iterations across the whole set (if the set is
              not fully classified earlier) [def: 1000]
* `α`:        Initial distribution of the errors [def: `zeros(length(y))`]
* `nMsg`:     Maximum number of messages to show if all iterations are done
* `rShuffle`: Wheter to randomly shuffle the data at each iteration [def: `false`]

# Return a named tuple with:
* `x`: the x data (eventually shuffled if `rShuffle=true`)
* `y`: the label
* `α`: the errors associated to each record
* `errors`: the number of errors in the last iteration
* `besterrors`: the minimum number of errors in classifying the data ever reached
* `iterations`: the actual number of iterations performed
* `separated`: a flag if the data has been successfully separated

# Notes:
* The trained data can then be used to make predictions using the function
  `predict()`. **If the option `randomShuffle` has been used, it is important to
   use there the returned (x,y,α) as these would have been shuffle compared with
   the original (x,y)**.

# Example:
julia> kernelPerceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
"""
function kernelPerceptron(x, y; K=radialKernel, T=1000, α=zeros(length(y)), nMsgs=10, rShuffle=false)
    if nMsgs != 0
        println("***\n*** Training kernel perceptron for maximum $T iterations. Random shuffle: $rShuffle")
    end
    x = makeMatrix(x)
    (n,d) = size(x)
    bestϵ = Inf
    lastϵ = Inf
    @showprogress 1 "Training Kernel Perceptron..." for t in 1:T
        ϵ = 0
        if rShuffle
           # random shuffle x, y and alpha
           ridx = shuffle(1:size(x)[1])
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
            return (x=x,y=y,α=α,errors=0,besterrors=0,iterations=t,separated=true)
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
 pegasus(x,y;θ,θ₀,λ,η,T,nMsgs,rShuffle,forceOrigin)

Train the peagasus algorithm based on x and y (labels)

# Parameters:
* `x`:           Feature matrix of the training data (n × d)
* `y`:           Associated labels of the training data, in the format of ⨦ 1
* `θ`:           Initial value of the weights (parameter) [def: `zeros(d)`]
* `θ₀`:          Initial value of the weight (parameter) associated to the constant
                term [def: `0`]
* `λ`:           Multiplicative term of the learning rate
* `η`:           Learning rate [def: (t -> 1/sqrt(t))]
* `T`:           Maximum number of iterations across the whole set (if the set
                is not fully classified earlier) [def: 1000]
* `nMsg`:        Maximum number of messages to show if all iterations are done
* `rShuffle`:    Wheter to randomly shuffle the data at each iteration [def: `false`]
* `forceOrigin`: Wheter to force `θ₀` to remain zero [def: `false`]

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
* The trained data can then be used to make predictions using the function
`predict()`. **If the option `randomShuffle` has been used, it is important to
 use there the returned (x,y,α) a
 s these would have been shuffle compared with
 the original (x,y)**.

# Example:
julia> pegasus([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
"""
function pegasus(x, y; θ=zeros(size(x,2)),θ₀=0.0, λ=0.5,η= (t -> 1/sqrt(t)), T=1000, nMsgs=10, rShuffle=false, forceOrigin=false)
  if nMsgs != 0
      println("***\n*** Training pegasus for maximum $T iterations. Random shuffle: $rShuffle")
  end
  x = makeMatrix(x)
  (n,d) = size(x)
  bestϵ = Inf
  lastϵ = Inf
  if forceOrigin θ₀ = 0.0; end
  sumθ = θ; sumθ₀ = θ₀
  @showprogress 1 "Training Perceptron..." for t in 1:T
      ϵ = 0
      ηₜ = η(t)
      if rShuffle
         # random shuffle x and y
         ridx = shuffle(1:size(x)[1])
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
  predict(x,θ,θ₀=0.0)

Predict a binary label {-1,1} given the feature vector and the linear coefficients

# Parameters:
* `x`:        Feature matrix of the training data (n × d)
* `y`:        Associated labels of the training data, in the format of ⨦ 1
* `K`:        Kernel function to emplo]y. See `?radialKernel` or `?polynomialKernel`
              for details or check `?Bmlt.Utils` to verify if other kernels are
              defined (you  can alsways define your own kernel) [def: `radialKernel`]
* `T`:        Maximum number of iterations across the whole set (if the set is
              not fully classified earlier) [def: 1000]
* `α`:        Initial distribution of the errors [def: `zeros(length(y))`]
* `nMsg`:     Maximum number of messages to show if all iterations are done
* `rShuffle`: Wheter to randomly shuffle the data at each iteration [def: `false`]

# Return a named tuple with:
* `x`: the x data (eventually shuffled if `rShuffle=true`)
* `y`: the label
* `α`: the errors associated to each record
* `errors`: the number of errors in the last iteration
* `besterrors`: the minimum number of errors in classifying the data ever reached
* `iterations`: the actual number of iterations performed
* `separated`: a flag if the data has been successfully separated

# Notes:
* The trained data can then be used to make predictions using the function
  `predict()`. **If the option `randomShuffle` has been used, it is important to
   use there the returned (x,y,α) as these would have been shuffle compared with
   the original (x,y)**.

# Example:
julia> kernelPerceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
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

function predict(x,xtrain,ytrain,α;K=radialKernel)
    x = makeMatrix(x)
    xtrain = makeMatrix(xtrain)
    (n,d) = size(x)
    (ntrain,d2) = size(xtrain)
    if (d2 != d) error("xtrain and x must have the same dimensions."); end
    if ( length(ytrain) != ntrain || length(α) != ntrain) error("xtrain, ytrain and α must al lhave the same length."); end
    y = zeros(Int64,n)
    for i in 1:n
        y[i] = sum([ α[j] * ytrain[j] * K(x[i,:],xtrain[j,:]) for j in 1:ntrain]) > eps() ? 1 : -1
    end
    return y
 end




end
