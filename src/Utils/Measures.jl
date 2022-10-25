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
l2squared_distance(x,y)    = norm(x-y)^2
"""Cosine distance"""
cosine_distance(x,y) = dot(x,y)/(norm(x)*norm(y))


################################################################################
### VARIOUS ERROR / LOSS / ACCURACY MEASURES
################################################################################

# ------------------------------------------------------------------------------
# Classification tasks...

# Used as neural network loss function
"""
   crossentropy(y,ŷ; weight)

Compute the (weighted) cross-entropy between the predicted and the sampled probability distributions.

To be used in classification problems.

"""
crossentropy(y,ŷ ; weight = ones(eltype(y),length(y)))  = -sum(y .* log.(ŷ .+ 1e-15) .* weight)
dcrossentropy(y,ŷ; weight = ones(eltype(y),length(y))) = - y .* weight ./ (ŷ .+ 1e-15)


""" accuracy(ŷ,y;ignorelabels=false) - Categorical accuracy between two vectors (T vs T). """
function accuracy(y::AbstractArray{T,1},ŷ::AbstractArray{T,1}; ignorelabels=false)  where {T}
    # See here for better performances: https://discourse.julialang.org/t/permutations-of-a-vector-that-retain-the-vector-structure/56790/7
    if(!ignorelabels)
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

""" error(y,ŷ;ignorelabels=false) - Categorical error (T vs T)"""
error(y::AbstractArray{T,1},ŷ::AbstractArray{T,1}; ignorelabels=false) where {T} = (1 - accuracy(y,ŷ;ignorelabels=ignorelabels) )


"""
    accuracy(y,ŷ;tol)
Categorical accuracy with probabilistic prediction of a single datapoint (PMF vs Int).

Use the parameter tol [def: `1`] to determine the tollerance of the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values.
"""
function accuracy(y_pos::Int64,ŷ::Array{T,1};tol=1,rng=Random.GLOBAL_RNG) where {T <: Number}
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
    accuracy(y,ŷ;tol)

Categorical accuracy with probabilistic prediction of a single datapoint given in terms of a dictionary of probabilities (Dict{T,Float64} vs T).

# Parameters:
- `ŷ`: The returned probability mass function in terms of a Dictionary(Item1 => Prob1, Item2 => Prob2, ...)
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].
"""
function accuracy(y::T,ŷ::Dict{T,Float64};tol=1,rng=Random.GLOBAL_RNG) where {T}
    if !(y in keys(ŷ)) return 0 end
    tol > 1 || return (mode(ŷ;rng=rng) == y) ? 1 : 0 # if tol is one we delegate the choice of a single prediction to mode, that handles multimodal pmfs
    sIdx  = sortperm(collect(values(ŷ)))[end:-1:1]            # sort by decreasing values of the dictionary values
    sKeys = collect(keys(ŷ))[sIdx][1:min(tol,length(sIdx))]  # retrieve the corresponding keys
    return (y in sKeys) ? 1 : 0
end

@doc raw"""
   accuracy(y,ŷ;tol,ignorelabels)

Categorical accuracy with probabilistic predictions of a dataset (PMF vs Int).

# Parameters:
- `y`: The N array with the correct category for each point $n$.
- `ŷ`: An (N,K) matrix of probabilities that each ``\hat y_n`` record with ``n \in 1,....,N``  being of category ``k`` with $k \in 1,...,K$.
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].
- `ignorelabels`: Whether to ignore the specific label order in y. Useful for unsupervised learning algorithms where the specific label order don't make sense [def: false]

"""
function accuracy(y::Array{Int64,1},ŷ::Array{T,2};tol=1,ignorelabels=false,rng=Random.GLOBAL_RNG) where {T <: Number}
    (N,D) = size(ŷ)
    pSet = ignorelabels ? collect(permutations(1:D)) : [collect(1:D)]
    bestAcc = -Inf
    for perm in pSet
        pŷ = hcat([ŷ[:,c] for c in perm]...)
        acc = sum([accuracy(y[i],pŷ[i,:];tol=tol,rng=rng) for i in 1:N])/N
        if acc > bestAcc
            bestAcc = acc
        end
    end
    return bestAcc
end

@doc raw"""
   accuracy(y,ŷ;tol)

Categorical accuracy with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (Dict{T,Float64} vs T).

# Parameters:
- `ŷ`: An array where each item is the estimated probability mass function in terms of a Dictionary(Item1 => Prob1, Item2 => Prob2, ...)
- `y`: The N array with the correct category for each point $n$.
- `tol`: The tollerance to the prediction, i.e. if considering "correct" only a prediction where the value with highest probability is the true value (`tol` = 1), or consider instead the set of `tol` maximum values [def: `1`].

"""
function accuracy(y::Array{T,1},ŷ::Array{Dict{T,Float64},1};tol=1,rng=Random.GLOBAL_RNG) where {T}
    N = size(ŷ,1)
    acc = sum([accuracy(y[i],ŷ[i];tol=tol,rng=rng) for i in 1:N])/N
    return acc
end

"""
$(TYPEDEF)

Compute the loss of a given model over a given (x,y) dataset running cross-validation
"""
function l2loss_by_cv(m,data;nsplits=5,rng=Random.GLOBAL_RNG)
    x,y = data[1],data[2]
    sampler = KFold(nsplits=nsplits,rng=rng)
    if (ndims(y) == 1)
        ohm = OneHotEncoder(handle_unknown="infrequent",cache=false)
        fit!(ohm,y)
    end
    (μ,σ) = cross_validation([x,y],sampler) do trainData,valData,rng
        (xtrain,ytrain) = trainData; (xval,yval) = valData
        fit!(m,xtrain,ytrain)
        ŷval     = predict(m,xval)
        if (eltype(ŷval) <: Dict)
            yval = predict(ohm,yval)
            ŷval = predict(ohm,ŷval)
        end
        ϵ               = norm(yval-ŷval)/size(yval,1) 
        reset!(m)
        return ismissing(ϵ) ? Inf : ϵ 
      end
    return μ
end 

""" error(y,ŷ) - Categorical error with probabilistic prediction of a single datapoint (Int vs PMF). """
error(y::Int64,ŷ::Array{T,1};tol=1) where {T <: Number} = 1 - accuracy(y,ŷ;tol=tol)
""" error(y,ŷ) - Categorical error with probabilistic predictions of a dataset (Int vs PMF). """
error(y::Array{Int64,1},ŷ::Array{T,2};tol=1) where {T <: Number} = 1 - accuracy(y,ŷ;tol=tol)
""" error(y,ŷ) - Categorical error with with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (T vs Dict{T,Float64}). """
error(y::Array{T,1},ŷ::Array{Dict{T,Float64},1};tol=1) where {T} = 1 - accuracy(y,ŷ;tol=tol)

"""
$(TYPEDEF)

Hyperparameters for [`ConfusionMatrix`](@ref)

# Parameters:
$(FIELDS)

"""
Base.@kwdef mutable struct ConfusionMatrixHyperParametersSet <: BetaMLHyperParametersSet
  "The categories (aka \"levels\") to represent. [def: `nothing`, i.e. unique ground true values]."  
  categories::Union{Vector,Nothing} = nothing
  "How to handle categories not seen in the ground true values or not present in the provided `categories` array? \"error\" (default) rises an error, \"infrequent\" adds a specific category for these values."
  handle_unknown::String = "error"
  "How to handle missing values in either ground true or predicted values ? \"error\" [default] will rise an error, \"drop\" will drop the record"
  handle_missing::String = "error"
  "Which value to assign to the \"other\" category (i.e. categories not seen in the gound truth or not present in the provided `categories` array? [def: ` nothing`, i.e. typemax(Int64) for integer vectors and \"other\" for other types]. This setting is active only if `handle_unknown=\"infrequent\"` and in that case it MUST be specified if the vector to one-hot encode is neither integer or strings"
  other_categories_name = nothing
  "A dictionary to map categories to some custom names. Useful for example if categories are integers, or you want to use shorter names [def: `Dict()`, i.e. not used]. This option isn't currently compatible with missing values or when some record has a value not in this provided dictionary."
  categories_names = Dict()
  "Wheter `predict` should return the normalised scores. Note that both unnormalised and normalised scores remain available using `info`. [def: `true`]"
  normalise_scores = true
end

Base.@kwdef mutable struct ConfusionMatrixLearnableParameters <: BetaMLLearnableParametersSet
    categories_applied::Vector = []
    original_vector_eltype::Union{Type,Nothing} = nothing 
    scores::Union{Nothing,Matrix{Int64}} = nothing 
  end

"""
$(TYPEDEF)

Compute a confusion matrix detailing the mismatch between observations and predictions of a categorical variable

For the parameters see [`ConfusionMatrixHyperParametersSet`](@ref) and [`BetaMLDefaultOptionsSet`](@ref).

The "predicted" values are either the scores or the normalised scores (depending on the parameter `normalise_scores` [def: `true`]).

# Notes: 
- The Confusion matrix report can be printed (i.e. `print(cm_model)`. If you plan to print the Confusion Matrix report, be sure that the type of the data in `y` and `ŷ` can be converted to `String`.

- Information in a structured way is available trought the `info(cm)` function that returns the following dictionary:
  - `accuracy`:           Oveall accuracy rate
  - `misclassification`:  Overall misclassification rate
  - `actual_count`:       Array of counts per lebel in the actual data
  - `predicted_count`:    Array of counts per label in the predicted data
  - `scores`:             Matrix actual (rows) vs predicted (columns)
  - `normalised_scores`:  Normalised scores
  - `tp`:                 True positive (by class)
  - `tn`:                 True negative (by class)
  - `fp`:                 False positive (by class)
  - `fn`:                 False negative (by class)
  - `precision`:          True class i over predicted class i (by class)
  - `recall`:             Predicted class i over true class i (by class)
  - `specificity`:        Predicted not class i over true not class i (by class)
  - `f1score`:            Harmonic mean of precision and recall
  - `mean_precision`:     Mean by class, respectively unweighted and weighted by actual_count
  - `mean_recall`:        Mean by class, respectively unweighted and weighted by actual_count
  - `mean_specificity`:   Mean by class, respectively unweighted and weighted by actual_count
  - `mean_f1score`:       Mean by class, respectively unweighted and weighted by actual_count
  - `categories`:         The categories considered
  - `fitted_records`:     Number of records considered
  - `n_categories`:       Number of categories considered

- The confusion matrix can also be plotted, e.g.:
```
using Plots, BetaML 
y  = [1,2,2,1,3,2,3]
ŷ  = [1,3,2,2,3,3,3]
cm = ConfusionMatrix()
fit!(cm,y,ŷ)
res = info(cm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")
```

"""
mutable struct ConfusionMatrix <: BetaMLUnsupervisedModel
    hpar::ConfusionMatrixHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,ConfusionMatrixLearnableParameters}
    cres::Union{Nothing,Matrix{Int64},Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}
end

function ConfusionMatrix(;kwargs...)
    m = ConfusionMatrix(ConfusionMatrixHyperParametersSet(),BetaMLDefaultOptionsSet(),ConfusionMatrixLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

Fit a [`ConfusionMatrix`](@ref) model to data.

!!! warning
    Data is expected in the order "ground truth, predictions" (i.e. `fit!(cm_model,y,ŷ)`)

This model supports multiple training (but the categories, if not provided, are extracteed from the first training y only), while prediction with new data (i.e. `predict(cm_model,ŷnew)`) is not supported.

"""
function fit!(m::ConfusionMatrix,Y,Ŷ)
    nR = size(Y,1)
    size(Ŷ,1) == nR || error("Y and Ŷ have different number of elements!")

    rng                    = m.opt.rng
    if eltype(Ŷ) <: Dict || ndims(Ŷ) > 1# allow probabilistic outputs
        Ŷ = mode(Ŷ,rng=rng)
    end

    vtype = eltype(Y) 

    # Parameter aliases
    categories             = m.hpar.categories
    handle_unknown         = m.hpar.handle_unknown
    handle_missing         = m.hpar.handle_missing
    other_categories_name  = m.hpar.other_categories_name
    categories_names       = m.hpar.categories_names
    if isnothing(other_categories_name)
        if nonmissingtype(vtype) <: Integer
            other_categories_name = typemax(Int64)
        else
            other_categories_name = "other"
        end
    end
    normalise_scores       = m.hpar.normalise_scores
    cache                  = m.opt.cache
    verbosity              = m.opt.verbosity

    fitted                 = m.fitted


    if categories_names != Dict()
        Y = map(x->categories_names[x], Y) 
        Ŷ = map(x->categories_names[x], Ŷ)
    end


    if fitted
        categories_applied = m.par.categories_applied
        nCl = length(categories_applied)
        scores = m.par.scores
    else
        categories_applied = isnothing(categories) ? collect(skipmissing(unique(Y))) : deepcopy(categories)
        handle_unknown == "infrequent" && push!(categories_applied,other_categories_name)
        nCl = length(categories_applied)
        scores = zeros(Int64,nCl,nCl)
    end

    for n in 1:nR
        if (ismissing(Y[n]) || ismissing(Ŷ[n]))
            if handle_missing == "error"
                error("Found a `missing` value in the data. To automatically drop missing data use the option `handle_missing=\"drop\"` in the `ConfusionMatrix` constructor.")
            else
                continue
            end
        end
        r = findfirst(x -> isequal(x,Y[n]),categories_applied)
        c = findfirst(x -> isequal(x,Ŷ[n]),categories_applied)
        if isnothing(r)
            if handle_unknown == "error"
                error("Found a category ($(Y[n])) not present in `categories` and the `handle_unknown` is set to `error`. Perhaps you want to swith it to `infrequent`.")
            elseif handle_unknown == "infrequent"
                r = length(categories_applied)
            else
                error("I don't know how to process `handle_unknown == $(handle_unknown)`")
            end
        end
        if isnothing(c)
            if handle_unknown == "error"
                error("Found a predicted category ($(Y[n])) not present in `categories` or in the true categories and the `handle_unknown` is set to `error`. Perhaps you want to swith it to `infrequent`.")
            elseif handle_unknown == "infrequent"
                c = length(categories_applied)
            else
                error("I don't know how to process `handle_unknown == $(handle_unknown)`")
            end
        end
        scores[r,c] += 1
    end

    predicted_count = dropdims(sum(scores,dims=1)',dims=2)
    actual_count    = dropdims(sum(scores,dims=2),dims=2)

    normalised_scores = zeros(nCl, nCl)
    [normalised_scores[r,:] = scores[r,:] ./ actual_count[r] for r in 1:nCl]
    tp = [scores[i,i] for i in 1:nCl]
    tn = [sum(scores[r,c] for r in 1:nCl, c in 1:nCl if r != i && c != i)  for i in 1:nCl]
    fp = [sum(scores[r,c] for r in 1:nCl, c in 1:nCl if r != i && c == i)  for i in 1:nCl]
    fn = [sum(scores[r,c] for r in 1:nCl, c in 1:nCl if r == i && c != i)  for i in 1:nCl]
    precision         = tp ./ (tp .+ fp)
    recall            = tp ./ (tp .+ fn)
    specificity       = tn ./ (tn .+ fp)
    f1score           = (2 .* tp) ./ (2 .* tp  .+ fp .+ fn )
    mean_precision     = (mean(precision), sum(precision .* actual_count) / sum(actual_count) )
    mean_recall        = (mean(recall), sum(recall .* actual_count) / sum(actual_count) )
    mean_specificity   = (mean(specificity), sum(specificity .* actual_count) / sum(actual_count) )
    mean_f1score       = (mean(f1score), sum(f1score .* actual_count) / sum(actual_count) )
    accuracy           = sum(tp)/sum(scores)
    misclassification  = 1-accuracy

    cache && (m.cres = normalise_scores ? normalised_scores : scores)

    m.par = ConfusionMatrixLearnableParameters(categories_applied,vtype,scores)

    m.info["accuracy"]          = accuracy           # Overall accuracy rate
    m.info["misclassification"] = misclassification  # Overall misclassification rate
    m.info["actual_count"]      = actual_count       # Array of counts per lebel in the actual data
    m.info["predicted_count"]   = predicted_count    # Array of counts per label in the predicted data
    m.info["scores"]            = scores             # Matrix actual (rows) vs predicted (columns)
    m.info["normalised_scores"] = normalised_scores  # Normalised scores
    m.info["tp"]                = tp                 # True positive (by class)
    m.info["tn"]                = tn                 # True negative (by class)
    m.info["fp"]                = fp                 # False positive (by class)
    m.info["fn"]                = fn                 # False negative (by class)
    m.info["precision"]         = precision          # True class i over predicted class i (by class)
    m.info["recall"]            = recall             # Predicted class i over true class i (by class)
    m.info["specificity"]       = specificity        # Predicted not class i over true not class i (by class)
    m.info["f1score"]           = f1score            # Harmonic mean of precision and recall
    m.info["mean_precision"]    = mean_precision     # Mean by class, respectively unweighted and weighted by actual_count
    m.info["mean_recall"]       = mean_recall        # Mean by class, respectively unweighted and weighted by actual_count
    m.info["mean_specificity"]  = mean_specificity   # Mean by class, respectively unweighted and weighted by actual_count
    m.info["mean_f1score"]      = mean_f1score       # Mean by class, respectively unweighted and weighted by actual_count

    m.info["categories"]        = categories_applied
    m.info["fitted_records"]    = sum(scores)
    m.info["n_categories"]      = nCl
    m.fitted = true
    return cache ? m.cres : nothing

end

function show(io::IO, m::ConfusionMatrix)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"A $(typeof(m)) BetaMLModel (unfitted)")
    else
        println(io,"A $(typeof(m)) BetaMLModel (fitted)")
        res    = info(m)
        labels = string.(res["categories"])
        nCl    = length(labels)

        println(io,"\n-----------------------------------------------------------------\n")
        println(io, "*** CONFUSION MATRIX ***")
        println(io,"")
        println(io,"Scores actual (rows) vs predicted (columns):\n")
        displayScores = vcat(permutedims(labels),res["scores"])
        displayScores = hcat(vcat("Labels",labels),displayScores)
        show(io, "text/plain", displayScores)
        println(io,"")
        println(io,"Normalised scores actual (rows) vs predicted (columns):\n")
        displayScores = vcat(permutedims(labels),res["normalised_scores"])
        displayScores = hcat(vcat("Labels",labels),displayScores)
        show(io, "text/plain", displayScores)

        println(io,"\n\n *** CONFUSION REPORT ***\n")
        labelWidth =  max(8,   maximum(length.(labels))+1  )
        println(io,"- Accuracy:               $(res["accuracy"])")
        println(io,"- Misclassification rate: $(res["misclassification"])")
        println(io,"- Number of classes:      $(nCl)")
        println(io,"")
        println(io,"  N ",rpad("Class",labelWidth),"precision   recall  specificity  f1score  actual_count  predicted_count")
        println(io,"    ",rpad(" ",labelWidth), "              TPR       TNR                 support                  ")
        println(io,"")
        # https://discourse.julialang.org/t/printf-with-variable-format-string/3805/4
        print_formatted(io, fmt, args...) = @eval @printf($io, $fmt, $(args...))
        for i in 1:nCl
            print_formatted(io, "%3d %-$(labelWidth)s %8.3f %8.3f %12.3f %8.3f %12i %15i\n", i, labels[i],  res["precision"][i], res["recall"][i], res["specificity"][i], res["f1score"][i], res["actual_count"][i], res["predicted_count"][i])
        end
        println(io,"")
        print_formatted(io, "- %-$(labelWidth+2)s %8.3f %8.3f %12.3f %8.3f\n", "Simple   avg.",  res["mean_precision"][1], res["mean_recall"][1], res["mean_specificity"][1], res["mean_f1score"][1])
        print_formatted(io, "- %-$(labelWidth+2)s %8.3f %8.3f %12.3f %8.3f\n", "Weigthed avg.",  res["mean_precision"][2], res["mean_recall"][2], res["mean_specificity"][2], res["mean_f1score"][2])
        println("\n-----------------------------------------------------------------")
        println("Output of `info(cm)`:")
        for (k,v) in info(m)
            print(io,"- ")
            print(io,k)
            print(io,":\t")
            println(io,v)
        end
    end
end




# OLD START HERE ---------------------------------------------------------------

# Resources concerning Confusion Matrices:
# https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
# https://en.wikipedia.org/wiki/Confusion_matrix
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html


# ------------------------------------------------------------------------------
# Regression tasks...

# Used as neural network loss function
"""
   squared_cost(y,ŷ)

Compute the squared costs between a vector of observations and one of prediction as (1/2)*norm(y - ŷ)^2.

Aside the 1/2 term, it correspond to the squared l-2 norm distance and when it is averaged on multiple datapoints corresponds to the Mean Squared Error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error)).
It is mostly used for regression problems.
"""
squared_cost(y,ŷ)   = (1/2)*norm(y - ŷ)^2
dsquared_cost(y,ŷ)  = ( ŷ - y)
"""
    mse(y,ŷ)

Compute the mean squared error (MSE) (aka mean squared deviation - MSD) between two vectors y and ŷ.
Note that while the deviation is averaged by the length of `y` is is not scaled to give it a relative meaning.
"""
mse(y,ŷ) = (sum((y-ŷ).^(2))/length(y))

"""
  relative_mean_error(y, ŷ;normdim=false,normrec=false,p=1)

Compute the relative mean error (l-1 based by default) between y and ŷ.

There are many ways to compute a relative mean error. In particular, if normrec (normdim) is set to true, the records (dimensions) are normalised, in the sense that it doesn't matter if a record (dimension) is bigger or smaller than the others, the relative error is first computed for each record (dimension) and then it is averaged.
With both `normdim` and `normrec` set to `false` (default) the function returns the relative mean error; with both set to `true` it returns the mean relative error (i.e. with p=1 the "[mean absolute percentage error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)")
The parameter `p` [def: `1`] controls the p-norm used to define the error.

The _mean relative error_ enfatises the relativeness of the error, i.e. all observations and dimensions weigth the same, wether large or small. Conversly, in the _relative mean error_ the same relative error on larger observations (or dimensions) weights more.

For example, given `y = [1,44,3]` and `ŷ = [2,45,2]`, the _mean relative error_ `mean_relative_error(y,ŷ,normrec=true)` is `0.452`, while the _relative mean error_ `relative_mean_error(y,ŷ, normrec=false)` is "only" `0.0625`.

"""
function relative_mean_error(y,ŷ;normdim=false,normrec=false,p=1)
    ŷ = makematrix(ŷ)
    y = makematrix(y)
    (n,d) = size(y)
    #ϵ = abs.(ŷ-y) .^ p
    if (!normdim && !normrec) # relative mean error
        avgϵRel = (sum(abs.(ŷ-y).^p)^(1/p) / (n*d)) / (sum( abs.(y) .^p)^(1/p) / (n*d)) # (avg error) / (avg y)
        # avgϵRel = (norm((ŷ-y),p)/(n*d)) / (norm(y,p) / (n*d))
    elseif (!normdim && normrec) # normalised by record (i.e. all records play the same weigth)
        avgϵRel_byRec = (sum(abs.(ŷ-y) .^ (1/p),dims=2).^(1/p) ./ d) ./   (sum(abs.(y) .^ (1/p) ,dims=2) ./d)
        avgϵRel = mean(avgϵRel_byRec)
    elseif (normdim && !normrec) # normalised by dimensions (i.e.  all dimensions play the same weigth)
        avgϵRel_byDim = (sum(abs.(ŷ-y) .^ (1/p),dims=1).^(1/p) ./ n) ./   (sum(abs.(y) .^ (1/p) ,dims=1) ./n)
        avgϵRel = mean(avgϵRel_byDim)
    else # mean relative error
        avgϵRel = sum(abs.((ŷ-y)./ y).^p)^(1/p)/(n*d) # avg(error/y)
        # avgϵRel = (norm((ŷ-y)./ y,p)/(n*d))
    end
    return avgϵRel
end
