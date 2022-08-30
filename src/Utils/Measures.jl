"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Part of submodule Utils of BetaML _ the Beta Machine Learning Toolkit
# Various measures of pairs (x,y) (including vectors or matrix pairs)

# ------------------------------------------------------------------------------
# Some common distance measures

"""L1 norm distance (aka _Manhattan Distance_)"""
l1_distance(x,y)     = sum(abs.(x-y))
"""Euclidean (L2) distance"""
l2_distance(x,y)     = norm(x-y)
"""Squared Euclidean (L2) distance"""
l2²_distance(x,y)    = norm(x-y)^2
"""Cosine distance"""
cosine_distance(x,y) = dot(x,y)/(norm(x)*norm(y))


################################################################################
### VARIOUS ERROR / LOSS / ACCURACY MEASURES
################################################################################

# ------------------------------------------------------------------------------
# Classification tasks...

# Used as neural network loss function
"""
   crossEntropy(ŷ, y; weight)

Compute the (weighted) cross-entropy between the predicted and the sampled probability distributions.

To be used in classification problems.

"""
crossEntropy(ŷ, y; weight = ones(eltype(y),length(y)))  = -sum(y .* log.(ŷ .+ 1e-15) .* weight)
dCrossEntropy(ŷ, y; weight = ones(eltype(y),length(y))) = - y .* weight ./ (ŷ .+ 1e-15)


""" accuracy(ŷ,y;ignoreLabels=false) - Categorical accuracy between two vectors (T vs T). """
function accuracy(ŷ::AbstractArray{T,1},y::AbstractArray{T,1}; ignoreLabels=false)  where {T}
    # See here for better performances: https://discourse.julialang.org/t/permutations-of-a-vector-that-retain-the-vector-structure/56790/7
    if(!ignoreLabels)
        return sum(ŷ .== y)/length(ŷ)
    else
        classes  = unique(y)
        nCl      = length(classes)
        N        = size(y,1)
        pSet     =  collect(permutations(1:nCl))
        bestAcc  = -Inf
        yOrigIdx = [findfirst(x -> x == y[i] , classes) for i in 1:N]
        ŷOrigIdx = [findfirst(x -> x == ŷ[i] , classes) for i in 1:N]
        for perm in pSet
            py = perm[yOrigIdx] # permuted specific version
            acc = sum(ŷOrigIdx .== py)/N
            if acc > bestAcc
                bestAcc = acc
            end
        end
        return bestAcc
    end
end

""" error(ŷ,y;ignoreLabels=false) - Categorical error (T vs T)"""
error(ŷ::AbstractArray{T,1},y::AbstractArray{T,1}; ignoreLabels=false) where {T} = (1 - accuracy(ŷ,y;ignoreLabels=ignoreLabels) )


"""
    accuracy(ŷ,y;tol)
Categorical accuracy with probabilistic prediction of a single datapoint (PMF vs Int).

Use the parameter tol [def: `1`] to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.
"""
function accuracy(ŷ::Array{T,1},y_pos::Int64;tol=1,rng=Random.GLOBAL_RNG) where {T <: Number}
    #if  length(Set(ŷ) == 1                         # all classes the same prob
    #    return rand(rng) < (1 / length(y)) ? 1 : 0 # If all values have the same prob, it returns 1 with prob 1/n_classes
    #end
    tol > 1 || return mode(ŷ;rng=rng) == y_pos ? 1 : 0 # if tol is one we delegate the choice of a single prediction to mode, that handles multimodal pmfs
    sIdx = sortperm(ŷ)[end:-1:1]
    if ŷ[y_pos] in ŷ[sIdx[1:min(tol,length(sIdx))]]
        return 1
    else
        return 0
    end
end

"""
    accuracy(ŷ,y;tol)

Categorical accuracy with probabilistic prediction of a single datapoint given in terms of a dictionary of probabilities (Dict{T,Float64} vs T).

# Parameters:
- `ŷ`: The returned probability mass function in terms of a Dictionary(Item1 => Prob1, Item2 => Prob2, ...)
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].
"""
function accuracy(ŷ::Dict{T,Float64},y::T;tol=1,rng=Random.GLOBAL_RNG) where {T}
    if !(y in keys(ŷ)) return 0 end
    tol > 1 || return (mode(ŷ;rng=rng) == y) ? 1 : 0 # if tol is one we delegate the choice of a single prediction to mode, that handles multimodal pmfs
    sIdx  = sortperm(collect(values(ŷ)))[end:-1:1]            # sort by decreasing values of the dictionary values
    sKeys = collect(keys(ŷ))[sIdx][1:min(tol,length(sIdx))]  # retrieve the corresponding keys
    return (y in sKeys) ? 1 : 0
end

@doc raw"""
   accuracy(ŷ,y;tol,ignoreLabels)

Categorical accuracy with probabilistic predictions of a dataset (PMF vs Int).

# Parameters:
- `ŷ`: An (N,K) matrix of probabilities that each ``\hat y_n`` record with ``n \in 1,....,N``  being of category ``k`` with $k \in 1,...,K$.
- `y`: The N array with the correct category for each point $n$.
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].
- `ignoreLabels`: Whether to ignore the specific label order in y. Useful for unsupervised learning algorithms where the specific label order don't make sense [def: false]

"""
function accuracy(ŷ::Array{T,2},y::Array{Int64,1};tol=1,ignoreLabels=false,rng=Random.GLOBAL_RNG) where {T <: Number}
    (N,D) = size(ŷ)
    pSet = ignoreLabels ? collect(permutations(1:D)) : [collect(1:D)]
    bestAcc = -Inf
    for perm in pSet
        pŷ = hcat([ŷ[:,c] for c in perm]...)
        acc = sum([accuracy(pŷ[i,:],y[i];tol=tol,rng=rng) for i in 1:N])/N
        if acc > bestAcc
            bestAcc = acc
        end
    end
    return bestAcc
end

@doc raw"""
   accuracy(ŷ,y;tol)

Categorical accuracy with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (Dict{T,Float64} vs T).

# Parameters:
- `ŷ`: An array where each item is the estimated probability mass function in terms of a Dictionary(Item1 => Prob1, Item2 => Prob2, ...)
- `y`: The N array with the correct category for each point $n$.
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].

"""
function accuracy(ŷ::Array{Dict{T,Float64},1},y::Array{T,1};tol=1,rng=Random.GLOBAL_RNG) where {T}
    N = size(ŷ,1)
    acc = sum([accuracy(ŷ[i],y[i];tol=tol,rng=rng) for i in 1:N])/N
    return acc
end


""" error(ŷ,y) - Categorical error with probabilistic prediction of a single datapoint (PMF vs Int). """
error(ŷ::Array{T,1},y::Int64;tol=1) where {T <: Number} = 1 - accuracy(ŷ,y;tol=tol)
""" error(ŷ,y) - Categorical error with probabilistic predictions of a dataset (PMF vs Int). """
error(ŷ::Array{T,2},y::Array{Int64,1};tol=1) where {T <: Number} = 1 - accuracy(ŷ,y;tol=tol)
""" error(ŷ,y) - Categorical error with with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (Dict{T,Float64} vs T). """
error(ŷ::Array{Dict{T,Float64},1},y::Array{T,1};tol=1) where {T} = 1 - accuracy(ŷ,y;tol=tol)

"""
    ConfusionMatrix


Scores and measures resulting from a comparation between true and predicted categorical variables

Use the function `ConfusionMatrix(ŷ,y;classes,labels,rng)` to build it and `report(cm::ConfusionMatrix;what)` to visualise it, or use the individual parts of interest, e.g. `display(cm.scores)`.

# Fields:
- `labels`: Array of categorical labels
- `accuracy`: Overall accuracy rate
- `misclassification`: Overall misclassification rate
- `actualCount`: Array of counts per lebel in the actual data
- `predictedCount`: Array of counts per label in the predicted data
- `scores`: Matrix actual (rows) vs predicted (columns)
- `normalisedScores`: Normalised scores
- `tp`: True positive (by class)
- `tn`: True negative (by class)
- `fp`: False positive (by class), aka "type I error" or "false allarm"
- `fn`: False negative (by class), aka "type II error" or "miss"
- `precision`: True class i over predicted class i (by class)
- `recall`: Predicted class i over true class i (by class), aka "True Positive Rate (TPR)", "Sensitivity" or "Probability of detection"
- `specificity`: Predicted not class i over true not class i (by class), aka "True Negative Rate (TNR)"
- `f1Score`: Harmonic mean of precision and recall
- `meanPrecision`: Mean by class, respectively unweighted and weighted by actualCount
- `meanRecall`: Mean by class, respectively unweighted and weighted by actualCount
- `meanSpecificity`: Mean by class, respectively unweighted and weighted by actualCount
- `meanF1Score`: Mean by class, respectively unweighted and weighted by actualCount


"""
struct ConfusionMatrix{T}
    classes::Vector{T}                      # Array of categorical labels
    labels::Vector{String}                  # String representation of the categories
    accuracy::Float64                       # Overall accuracy rate
    misclassification::Float64              # Overall misclassification rate
    actualCount::Vector{Int64}              # Array of counts per lebel in the actual data
    predictedCount::Vector{Int64}           # Array of counts per label in the predicted data
    scores::Array{Int64,2}                  # Matrix actual (rows) vs predicted (columns)
    normalisedScores::Array{Float64,2}      # Normalised scores
    tp::Vector{Int64}                       # True positive (by class)
    tn::Vector{Int64}                       # True negative (by class)
    fp::Vector{Int64}                       # False positive (by class)
    fn::Vector{Int64}                       # False negative (by class)
    precision::Vector{Float64}              # True class i over predicted class i (by class)
    recall::Vector{Float64}                 # Predicted class i over true class i (by class)
    specificity::Vector{Float64}            # Predicted not class i over true not class i (by class)
    f1Score::Vector{Float64}                # Harmonic mean of precision and recall
    meanPrecision::Tuple{Float64,Float64}   # Mean by class, respectively unweighted and weighted by actualCount
    meanRecall::Tuple{Float64,Float64}      # Mean by class, respectively unweighted and weighted by actualCount
    meanSpecificity::Tuple{Float64,Float64} # Mean by class, respectively unweighted and weighted by actualCount
    meanF1Score::Tuple{Float64,Float64}     # Mean by class, respectively unweighted and weighted by actualCount
end

# Resources concerning Confusion Matrices:
# https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
# https://en.wikipedia.org/wiki/Confusion_matrix
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

"""
    ConfusionMatrix(ŷ,y;classes,labels,rng)

Build a "confusion matrix" between predicted (columns) vs actual (rows) categorical values

# Parameters:
- `ŷ`: Vector of predicted categorical data
- `y`: Vector of actual categorical data
- `classes`: The full set of possible classes (useful to give a specicif order or if not al lclasses are represented in `y`) [def: `unique(y)` ]
- `labels`: String representation of the classes [def: `string.(classes)`]
- `rng`: Random number generator. Used only if `ŷ` is given in terms of a PMF and there are multi-modal values, as these are assigned randomply [def: `Random.GLOBAL_RNG`]

# Return:
- a `ConfusionMatrix` object
"""
function ConfusionMatrix(ŷ,y::AbstractArray{T};classes=unique(y),labels=string.(classes),rng=Random.GLOBAL_RNG) where {T}
    nCl              = length(labels)
    ŷ                = typeof(ŷ) <: AbstractVector{T} ? ŷ : mode(ŷ,rng=rng) # get the mode if needed
    N                = length(y)
    length(ŷ) == N || @error "ŷ and y must have the same length in ConfusionMatrix"
    actualCount      = [get(classCountsWithLabels(y),i,0) for i in classes]   # TODO just use classCount
    predictedCount   = [get( classCountsWithLabels(ŷ),i,0) for i in classes]  # TODO just use classCount
    scores           = zeros(Int64,(nCl,nCl))
    normalisedScores = zeros(Float64,(nCl,nCl))
    [scores[findfirst(x -> x == y[i],classes),findfirst(x -> x == ŷ[i],classes)] += 1 for i in 1:N]
    [normalisedScores[r,:] = scores[r,:] ./ actualCount[r] for r in 1:nCl]
    tp = [scores[i,i] for i in 1:nCl]
    tn = [sum(scores[r,c] for r in 1:nCl, c in 1:nCl if r != i && c != i)  for i in 1:nCl]
    fp = [sum(scores[r,c] for r in 1:nCl, c in 1:nCl if r != i && c == i)  for i in 1:nCl]
    fn = [sum(scores[r,c] for r in 1:nCl, c in 1:nCl if r == i && c != i)  for i in 1:nCl]
    precision         = tp ./ (tp .+ fp)
    recall            = tp ./ (tp .+ fn)
    specificity       = tn ./ (tn .+ fp)
    #f1Score           = 2 .* (precision .* recall) ./ (precision .+ recall)
    f1Score           = (2 .* tp) ./ (2 .* tp  .+ fp .+ fn )
    meanPrecision     = (mean(precision), sum(precision .* actualCount) / sum(actualCount) )
    meanRecall        = (mean(recall), sum(recall .* actualCount) / sum(actualCount) )
    meanSpecificity   = (mean(specificity), sum(specificity .* actualCount) / sum(actualCount) )
    meanF1Score       = (mean(f1Score), sum(f1Score .* actualCount) / sum(actualCount) )
    accuracy          = sum(tp)/N
    misclassification = 1-accuracy
    return  ConfusionMatrix(classes,labels,accuracy,misclassification,actualCount,predictedCount,scores,normalisedScores,tp,tn,fp,fn,precision,recall,specificity,f1Score,meanPrecision,meanRecall,meanSpecificity,meanF1Score)
end


import Base.print, Base.println
"""
    print(cm,what)

Print a `ConfusionMatrix` object

The `what` parameter is a string vector that can include "all", "scores", "normalisedScores" or "report" [def: `["all"]`]
"""
function print(io::IO,cm::ConfusionMatrix{T},what="all") where T
   if what == "all" || what == ["all"]
       what = ["scores", "normalisedScores", "report" ]
   end
   nCl = length(cm.labels)

   println("\n-----------------------------------------------------------------\n")
   if( "scores" in what || "normalisedScores" in what)
     println("*** CONFUSION MATRIX ***")
   end
   if "scores" in what
       println("")
       println("Scores actual (rows) vs predicted (columns):\n")
       displayScores = vcat(permutedims(cm.labels),cm.scores)
       displayScores = hcat(vcat("Labels",cm.labels),displayScores)
       show(stdout, "text/plain", displayScores)
   end
   if "normalisedScores" in what
       println(io,"")
       println(io,"Normalised scores actual (rows) vs predicted (columns):\n")
       displayScores = vcat(permutedims(cm.labels),cm.normalisedScores)
       displayScores = hcat(vcat("Labels",cm.labels),displayScores)
       show(stdout, "text/plain", displayScores)
   end
   if "report" in what
     println("\n *** CONFUSION REPORT ***\n")
     labelWidth =  max(8,   maximum(length.(string.(cm.labels)))+1  )
     println("- Accuracy:               $(cm.accuracy)")
     println("- Misclassification rate: $(cm.misclassification)")
     println("- Number of classes:      $(nCl)")
     println("")
     println("  N ",rpad("Class",labelWidth),"precision   recall  specificity  f1Score  actualCount  predictedCount")
     println("    ",rpad(" ",labelWidth), "              TPR       TNR                 support                  ")
     println("")
     # https://discourse.julialang.org/t/printf-with-variable-format-string/3805/4
     print_formatted(fmt, args...) = @eval @printf($fmt, $(args...))
     for i in 1:nCl
        print_formatted("%3d %-$(labelWidth)s %8.3f %8.3f %12.3f %8.3f %12i %15i\n", i, string(cm.labels[i]),  cm.precision[i], cm.recall[i], cm.specificity[i], cm.f1Score[i], cm.actualCount[i], cm.predictedCount[i])
     end
     println("")
     print_formatted("- %-$(labelWidth+2)s %8.3f %8.3f %12.3f %8.3f\n", "Simple   avg.",  cm.meanPrecision[1], cm.meanRecall[1], cm.meanSpecificity[1], cm.meanF1Score[1])
     print_formatted("- %-$(labelWidth+2)s %8.3f %8.3f %12.3f %8.3f\n", "Weigthed avg.",  cm.meanPrecision[2], cm.meanRecall[2], cm.meanSpecificity[2], cm.meanF1Score[2])
   end
   println("\n-----------------------------------------------------------------")
   return nothing
end
println(io::IO, cm::ConfusionMatrix{T}, what="all") where T = begin  print(cm,what);print("\n"); return nothing end

# ------------------------------------------------------------------------------
# Regression tasks...

# Used as neural network loss function
"""
   squaredCost(ŷ,y)

Compute the squared costs between a vector of prediction and one of observations as (1/2)*norm(y - ŷ)^2.

Aside the 1/2 term, it correspond to the squared l-2 norm distance and when it is averaged on multiple datapoints corresponds to the Mean Squared Error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error)).
It is mostly used for regression problems.
"""
squaredCost(ŷ,y)   = (1/2)*norm(y - ŷ)^2
dSquaredCost(ŷ,y)  = ( ŷ - y)
"""
    mse(ŷ,y)

Compute the mean squared error (MSE) (aka mean squared deviation - MSD) between two vectors ŷ and y.
Note that while the deviation is averaged by the length of `y` is is not scaled to give it a relative meaning.
"""
mse(ŷ,y) = (sum((y-ŷ).^(2))/length(y))

"""
  meanRelError(ŷ,y;normDim=true,normRec=true,p=1)

Compute the mean relative error (l-1 based by default) between ŷ and y.

There are many ways to compute a mean relative error. In particular, if normRec (normDim) is set to true, the records (dimensions) are normalised, in the sense that it doesn't matter if a record (dimension) is bigger or smaller than the others, the relative error is first computed for each record (dimension) and then it is averaged.
With both `normDim` and `normRec` set to `false` the function returns the relative mean error; with both set to `true` (default) it returns the mean relative error (i.e. with p=1 the "[mean absolute percentage error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)")
The parameter `p` [def: `1`] controls the p-norm used to define the error.

The _mean relative error_ enfatises the relativeness of the error, i.e. all observations and dimensions weigth the same, wether large or small. Conversly, in the _relative mean error_ the same relative error on larger observations (or dimensions) weights more.

For example, given `y = [1,44,3]` and `ŷ = [2,45,2]`, the _mean relative error_ `meanRelError(ŷ,y)` is `0.452`, while the _relative mean error_ `meanRelError(ŷ,y, normRec=false)` is "only" `0.0625`.

"""
function meanRelError(ŷ,y;normDim=true,normRec=true,p=1)
    ŷ = makeMatrix(ŷ)
    y = makeMatrix(y)
    (n,d) = size(y)
    #ϵ = abs.(ŷ-y) .^ p
    if (!normDim && !normRec) # relative mean error
        avgϵRel = (sum(abs.(ŷ-y).^p)^(1/p) / (n*d)) / (sum( abs.(y) .^p)^(1/p) / (n*d)) # (avg error) / (avg y)
        # avgϵRel = (norm((ŷ-y),p)/(n*d)) / (norm(y,p) / (n*d))
    elseif (!normDim && normRec) # normalised by record (i.e. all records play the same weigth)
        avgϵRel_byRec = (sum(abs.(ŷ-y) .^ (1/p),dims=2).^(1/p) ./ d) ./   (sum(abs.(y) .^ (1/p) ,dims=2) ./d)
        avgϵRel = mean(avgϵRel_byRec)
    elseif (normDim && !normRec) # normalised by dimensions (i.e.  all dimensions play the same weigth)
        avgϵRel_byDim = (sum(abs.(ŷ-y) .^ (1/p),dims=1).^(1/p) ./ n) ./   (sum(abs.(y) .^ (1/p) ,dims=1) ./n)
        avgϵRel = mean(avgϵRel_byDim)
    else # mean relative error
        avgϵRel = sum(abs.((ŷ-y)./ y).^p)^(1/p)/(n*d) # avg(error/y)
        # avgϵRel = (norm((ŷ-y)./ y,p)/(n*d))
    end
    return avgϵRel
end
