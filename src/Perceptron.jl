

module Perceptron

using LinearAlgebra, Random, ProgressMeter
import ..Utils: radialKernel, polynomialKernel, makeMatrix, error

export kernelPerceptron, predict, radialKernel, polynomialKernel,
       makeMatrix, error

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



function predict(xtest,xtrain,ytrain,α;K=radialKernel)
    xtest = makeMatrix(xtest)
    xtrain = makeMatrix(xtrain)
    (ntest,d) = size(xtest)
    (ntrain,d2) = size(xtrain)
    if (d2 != d) error("xtrain and xtest must have the same dimensions."); end
    if ( length(ytrain) != ntrain || length(α) != ntrain) error("xtrain, ytrain and α must al lhave the same length."); end
    ytest = zeros(Int64,ntest)
    for i in 1:ntest
        ytest[i] = sum([ α[j] * ytrain[j] * K(xtest[i,:],xtrain[j,:]) for j in 1:ntrain]) > 0 ? 1 : -1
    end
    return ytest
 end


end
