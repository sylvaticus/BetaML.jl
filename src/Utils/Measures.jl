"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Part of submodule Utils of BetaML _ the Beta Machine Learning Toolkit
# Various measures of pairs (x,y) (including vectors or matrix pairs)

# ------------------------------------------------------------------------------
# Some common distance measures

# https://weaviate.io/blog/distance-metrics-in-vector-search
"""L1 norm distance (aka _Manhattan Distance_)"""
l1_distance(x,y)     = sum(abs.(x-y))
"""Euclidean (L2) distance"""
l2_distance(x,y)     = norm(x-y)
"""Squared Euclidean (L2) distance"""
l2squared_distance(x,y)    = norm(x-y)^2
"""Cosine distance"""
cosine_distance(x,y) = 1-dot(x,y)/(norm(x)*norm(y))
"""
$(TYPEDSIGNATURES)

Compute pairwise distance matrix between elements of an array identified across dimension `dims`.

# Parameters: 
- `x`: the data array 
- distance: a distance measure [def: `l2_distance`]
- dims: the dimension of the observations [def: `1`, i.e. records on rows]

# Returns:
- a n_records by n_records simmetric matrix of the pairwise distances

# Notes:
- if performances matters, you can use something like `Distances.pairwise(Distances.euclidean,x,dims=1)` from the [`Distances`](https://github.com/JuliaStats/Distances.jl) package.
"""
function pairwise(x::AbstractArray;distance=l2_distance,dims=1)
    N   = size(x,dims)
    out = zeros(N,N)
    
    for r in 1:N
        for c in 1:r
            out[r,c] =  distance(selectdim(x,dims,r),selectdim(x,dims,c))
        end
    end
    for r in 1:N
        for c in r+1:N
            out[r,c] = out[c,r]
        end
    end
    
    return out
end    


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
function accuracy(y_pos::Int64,ŷ::AbstractArray{T,1};tol=1,rng=Random.GLOBAL_RNG) where {T <: Number}
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
function accuracy(y::T,ŷ::AbstractDict{T,Float64};tol=1,rng=Random.GLOBAL_RNG) where {T}
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
function accuracy(y::AbstractArray{Int64,1},ŷ::AbstractArray{T,2};tol=1,ignorelabels=false,rng=Random.GLOBAL_RNG) where {T <: Number}
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
function accuracy(y::AbstractArray{T,1},ŷ::AbstractArray{Dict{T,Float64},1};tol=1,rng=Random.GLOBAL_RNG) where {T}
    N = size(ŷ,1)
    acc = sum([accuracy(y[i],ŷ[i];tol=tol,rng=rng) for i in 1:N])/N
    return acc
end

"""
$(TYPEDEF)

Compute the loss of a given model over a given (x,y) dataset running cross-validation
"""
function l2loss_by_cv(m,data;nsplits=5,rng=Random.GLOBAL_RNG)
    if length(data) == 2 # supervised model
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
    elseif length(data) == 1 # unsupervised model with inverse_predict
        x= data[1]
        sampler = KFold(nsplits=nsplits,rng=rng)
        (μ,σ) = cross_validation([x],sampler) do trainData,valData,rng
            (xtrain,) = trainData; (xval,) = valData
            fit!(m,xtrain)
            x̂val     = inverse_predict(m,xval)
            ϵ        = norm(xval .- x̂val)/size(xval,1) 
            reset!(m)
            return ismissing(ϵ) ? Inf : ϵ 
        end
        return μ
    else
        @error "Function `l2loss_by_cv` accepts only 1-lenght or 2-length data for respectivelly unsupervised and supervised models"
    end
end 

""" error(y,ŷ) - Categorical error with probabilistic prediction of a single datapoint (Int vs PMF). """
error(y::Int64,ŷ::Array{T,1};tol=1) where {T <: Number} = 1 - accuracy(y,ŷ;tol=tol)
""" error(y,ŷ) - Categorical error with probabilistic predictions of a dataset (Int vs PMF). """
error(y::Array{Int64,1},ŷ::Array{T,2};tol=1) where {T <: Number} = 1 - accuracy(y,ŷ;tol=tol)
""" error(y,ŷ) - Categorical error with with probabilistic predictions of a dataset given in terms of a dictionary of probabilities (T vs Dict{T,Float64}). """
error(y::Array{T,1},ŷ::Array{Dict{T,Float64},1};tol=1) where {T} = 1 - accuracy(y,ŷ;tol=tol)

"""
$(TYPEDSIGNATURES)

Provide Silhouette scoring for cluster outputs

# Parameters:
- `distances`: the nrecords by nrecords pairwise distance matrix
- `classes`: the vector of assigned classes to each record

# Notes:
- the matrix of pairwise distances can be obtained with the function [`pairwise`](@ref)
- this function doesn't sample. Eventually sample before
- to get the score for the cluster simply compute the `mean`
- see also the [Wikipedia article](https://en.wikipedia.org/wiki/Silhouette_(clustering))

# Example:
```julia
julia> x  = [1 2 3 3; 1.2 3 3.1 3.2; 2 4 6 6.2; 2.1 3.5 5.9 6.3];

julia> s_scores = silhouette(pairwise(x),[1,2,2,2])
4-element Vector{Float64}:
  0.0
 -0.7590778795827623
  0.5030093571833065
  0.4936350560759424
```
"""
function silhouette(distances,classes)
    uclasses = unique(classes)
    K        = length(uclasses)
    N        = size(distances,1)
    out      = Array{Float64,1}(undef,N)
    positions = hcat([classes .== cl for cl in uclasses]...) # N by K
    nByClass  = reshape(sum(positions,dims=1),K)
    #println(nByClass)
    for n in 1:N
       cl = classes[n]
       a = 0.0
       b = Inf
       #println("---------")
       #println("n: $n")
       for clidx in 1:K
            #print("- cl $clidx")
            cldists = distances[n,positions[:,clidx]]
            if cl == uclasses[clidx] # own cluster
                a  = sum(cldists)/ (nByClass[clidx]-1)
                #println(" a: $a")
            else
                btemp = sum(cldists) / nByClass[clidx]
                #println(" b: $btemp")
                if btemp < b
                    b = btemp
                end
            end
        end
        if isnan(a)
            out[n] = 0.0
        elseif a < b
            out[n] = 1-(a/b)
        else
            out[n] = (b/a) -1
        end 
        #println("- s: $(out[n])")
    end
    return out
end



"""
$(TYPEDEF)

Hyperparameters for [`ConfusionMatrix`](@ref)

# Parameters:
$(FIELDS)

"""
Base.@kwdef mutable struct ConfusionMatrix_hp <: BetaMLHyperParametersSet
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
  "Wether `predict` should return the normalised scores. Note that both unnormalised and normalised scores remain available using `info`. [def: `true`]"
  normalise_scores = true
end

Base.@kwdef mutable struct ConfusionMatrix_lp <: BetaMLLearnableParametersSet
    categories_applied::Vector = []
    original_vector_eltype::Union{Type,Nothing} = nothing 
    scores::Union{Nothing,Matrix{Int64}} = nothing 
  end

"""
$(TYPEDEF)

Compute a confusion matrix detailing the mismatch between observations and predictions of a categorical variable

For the parameters see [`ConfusionMatrix_hp`](@ref) and [`BML_options`](@ref).

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

# Example:

The confusion matrix can also be plotted, e.g.:

```julia
julia> using Plots, BetaML

julia> y  = ["apple","mandarin","clementine","clementine","mandarin","apple","clementine","clementine","apple","mandarin","clementine"];

julia> ŷ  = ["apple","mandarin","clementine","mandarin","mandarin","apple","clementine","clementine",missing,"clementine","clementine"];

julia> cm = ConfusionMatrix(handle_missing="drop")
A ConfusionMatrix BetaMLModel (unfitted)

julia> normalised_scores = fit!(cm,y,ŷ)
3×3 Matrix{Float64}:
 1.0  0.0       0.0
 0.0  0.666667  0.333333
 0.0  0.2       0.8

julia> println(cm)
A ConfusionMatrix BetaMLModel (fitted)

-----------------------------------------------------------------

*** CONFUSION MATRIX ***

Scores actual (rows) vs predicted (columns):

4×4 Matrix{Any}:
 "Labels"       "apple"   "mandarin"   "clementine"
 "apple"       2         0            0
 "mandarin"    0         2            1
 "clementine"  0         1            4
Normalised scores actual (rows) vs predicted (columns):

4×4 Matrix{Any}:
 "Labels"       "apple"   "mandarin"   "clementine"
 "apple"       1.0       0.0          0.0
 "mandarin"    0.0       0.666667     0.333333
 "clementine"  0.0       0.2          0.8

 *** CONFUSION REPORT ***

- Accuracy:               0.8
- Misclassification rate: 0.19999999999999996
- Number of classes:      3

  N Class      precision   recall  specificity  f1score  actual_count  predicted_count
                             TPR       TNR                 support                  

  1 apple          1.000    1.000        1.000    1.000            2               2
  2 mandarin       0.667    0.667        0.857    0.667            3               3
  3 clementine     0.800    0.800        0.800    0.800            5               5

- Simple   avg.    0.822    0.822        0.886    0.822
- Weigthed avg.    0.800    0.800        0.857    0.800

-----------------------------------------------------------------
Output of `info(cm)`:
- mean_precision:       (0.8222222222222223, 0.8)
- fitted_records:       10
- specificity:  [1.0, 0.8571428571428571, 0.8]
- precision:    [1.0, 0.6666666666666666, 0.8]
- misclassification:    0.19999999999999996
- mean_recall:  (0.8222222222222223, 0.8)
- n_categories: 3
- normalised_scores:    [1.0 0.0 0.0; 0.0 0.6666666666666666 0.3333333333333333; 0.0 0.2 0.8]
- tn:   [8, 6, 4]
- mean_f1score: (0.8222222222222223, 0.8)
- actual_count: [2, 3, 5]
- accuracy:     0.8
- recall:       [1.0, 0.6666666666666666, 0.8]
- f1score:      [1.0, 0.6666666666666666, 0.8]
- mean_specificity:     (0.8857142857142858, 0.8571428571428571)
- predicted_count:      [2, 3, 5]
- scores:       [2 0 0; 0 2 1; 0 1 4]
- tp:   [2, 2, 4]
- fn:   [0, 1, 1]
- categories:   ["apple", "mandarin", "clementine"]
- fp:   [0, 1, 1]

julia> res = info(cm);

julia> heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")
```
![CM plot](assets/cmClementines.png) 

"""
mutable struct ConfusionMatrix <: BetaMLUnsupervisedModel
    hpar::ConfusionMatrix_hp
    opt::BML_options
    par::Union{Nothing,ConfusionMatrix_lp}
    cres::Union{Nothing,Matrix{Int64},Matrix{Float64}}
    fitted::Bool
    info::Dict{String,Any}
end

function ConfusionMatrix(;kwargs...)
    m = ConfusionMatrix(ConfusionMatrix_hp(),BML_options(),ConfusionMatrix_lp(),nothing,false,Dict{Symbol,Any}())
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

    m.par = ConfusionMatrix_lp(categories_applied,vtype,scores)

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


# ------------------------------------------------------------------------------
# VariableImportance

"""
$(TYPEDEF)

Hyperparameters for [`VariableImportance`](@ref)

# Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct VariableImportance_hp <: BetaMLHyperParametersSet
    "Estimator model"
    model
    "Method to employ. Currently two methods are provided, \"sobol\" uses the variance-analysis based Sobol index, \"mda\" uses the mean decrease in accuracy"
    method::String = "sobol"
    "Function to apply the sobol index between ŷ and ŷ⁽⁻ʲ⁾, i.e. the reduction in output explained variance if the jth output variable is removed. Emploied only if `method` is \"sobol\". Note that the default function works only for regression tasks."
    sobol_loss::Function = sobol_index
    "Function to apply the loss between y and ŷ,. If `nothing` a loss function is applied that adapt to both regression and classification problems. Emploied only if `method` is \"mda\""
    mda_loss::Union{Function,Nothing} = nothing
    "The default `sobol_loss` and `mda_loss` functions treat integer Y as regression. Use `force_classification = true` to treat that integers as classes. Note that this has no effect on model training, where it has to be determined eventually in the model hyperparameters"
    force_classification = false
    "If `false` the variance importance is computed in a single loop over all the variables, otherwise the less important variable is removed (and provided) and then the algorithm is run again with the remaining variabels, recursively"
    recursive::Bool = false
    "Number of splits in the cross-validation function used to judge the importance of each dimension"
    nsplits::Int64 = 5
    """Minimum number of records (or share of it, if a float) to consider in the first loop used to retrieve the less important variable. The sample is then linearly increased up to `sample_max` to retrieve the most important variable.
    This parameter is ignored if `recursive=false`.
    Note that there is a fixed limit of `nplits*5` that prevails if lower."""
    sample_min::Union{Float64,Int64} = 30
    """Maximum number of records (or share of it, if a float) to consider in the last loop used to retrieve the most important variable, or if `recursive=false`."""
    sample_max::Union{Float64,Int64} = 1.0
    "The function used by the estimator(s) to fit the model. It should take as fist argument the model itself, as second argument a matrix representing the features, and as third argument a vector representing the labels. This parameter is mandatory for non-BetaML estimators and can be a single value or a vector (one per estimator) in case of different estimator packages used. [default: `BetaML.fit!`]"
    fit_function::Function     = fit!
    "The function used by the estimator(s) to predict the labels. It should take as fist argument the model itself and as second argument a matrix representing the features. This parameter is mandatory for non-BetaML estimators and can be a single value or a vector (one per estimator) in case of different estimator packages used. [default: `BetaML.predict`]"
    predict_function::Function = predict
    "The keyword to ignore specific dimensions in prediction. If the model supports this keyword in the prediciton function, when we loop over the various dimensions we use only prediction with this keyword instead of retraining."
    # See https://towardsdatascience.com/variable-importance-in-random-forests-20c6690e44e0
    ignore_dims_keyword::String = "ignore_dims"
end

Base.@kwdef struct VariableImportance_lp <: BetaMLLearnableParametersSet
    fitted_models::Dict = Dict() # Dictionary of array of of column indices and fitted models. For example [1,2,4] => a_fitted_RF_model store the fidded model for cols 1, 2 and 4
    variable_scores::Vector{Int64} = Int64[]
end

"""
$(TYPEDEF)

Compute Variable Importance

Key principles:
- can choose recursive or not
- can increase samples to determine most important variables
- can use Sobol index or mean average decreased accuracy
- can exploit models that can predict ignoring columns without retraining

See [`VariableImportance_hp`](@ref) for all the hyper-parameters.

# Examples:

"""
mutable struct VariableImportance <: BetaMLModel
    hpar::VariableImportance_hp
    opt::BML_options
    par::Union{VariableImportance_lp,Nothing}
    cres
    fitted::Bool
    info::Dict{String,Any}    
end

function VariableImportance(;kwargs...)
    
    hps = VariableImportance_hp()
    m   = VariableImportance(hps,BML_options(),VariableImportance_lp(),nothing,false,Dict{Symbol,Any}())
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

Fit a Variable Importance  model using [`VariableImportance`](@ref)
"""
function fit!(m::VariableImportance,X,y)
    nR,nC   = size(X)
    multiple_imputations  = m.hpar.multiple_imputations
    recursive_passages    = m.hpar.recursive_passages
    cache                = m.opt.cache
    verbosity            = m.opt.verbosity 
    rng                  = m.opt.rng

    # determining cols_to_impute...
    if m.hpar.cols_to_impute == "auto"
        cols2imp = findall(i -> i==true, [any(ismissing.(c)) for c in eachcol(X)]) #  ismissing.(sum.(eachcol(X))))
    elseif m.hpar.cols_to_impute == "all"
        cols2imp = collect(1:size(X,2))
    else
        cols2imp = m.hpar.cols_to_impute
    end

    nD2Imp = length(cols2imp)

    # Setting `estimators`, a matrix of multiple_imputations x nD2Imp individual models...
    if ! m.fitted
        if m.hpar.estimator == nothing
            estimators = [RandomForestEstimator(rng = m.opt.rng, verbosity=verbosity) for i in 1:multiple_imputations, d in 1:nD2Imp]
        elseif typeof(m.hpar.estimator) <: AbstractVector
            length(m.hpar.estimator) == nD2Imp || error("I can't use $(length(m.hpar.estimator)) estimators to impute $(nD2Imp) columns.")
            estimators = vcat([permutedims(deepcopy(m.hpar.estimator)) for i in 1:multiple_imputations]...)
        else # single estimator
            estimators = [deepcopy(m.hpar.estimator) for i in 1:multiple_imputations, j in 1:nD2Imp]
        end
    else
        m.opt.verbosity >= STD && @warn "This imputer has already been fitted. Not all learners support multiple training."
        estimators = m.par.fittedModels
    end

    missing_supported = typeof(m.hpar.missing_supported) <: AbstractArray ? m.hpar.missing_supported : fill(m.hpar.missing_supported,nD2Imp) 
    fit_functions = typeof(m.hpar.fit_function) <: AbstractArray ? m.hpar.fit_function : fill(m.hpar.fit_function,nD2Imp) 
    predict_functions = typeof(m.hpar.predict_function) <: AbstractArray ? m.hpar.predict_function : fill(m.hpar.predict_function,nD2Imp) 


    imputed = fill(similar(X),multiple_imputations)

    missingMask    = ismissing.(X)
    nonMissingMask = .! missingMask 
    n_imputed_values = sum(missingMask)
    x_used_cols = [Int64[] for d in 1:size(X,2)]

    for imputation in 1:multiple_imputations
        verbosity >= STD && println("** Processing imputation $imputation")
        Xout           = copy(X)
        sortedDims     = reverse(sortperm(makecolvector(sum(missingMask,dims=1)))) # sorted from the dim with more missing values
        for pass in 1:recursive_passages
            Xout_passage = copy(Xout)
            m.opt.verbosity >= HIGH && println("- processing passage $pass")
            if pass > 1
                shuffle!(rng, sortedDims) # randomise the order we go trough the various dimensions at this passage
            end 
            for d in sortedDims
                !(d in cols2imp) && continue
                dIdx = findfirst(x -> x == d, cols2imp)
                verbosity >= FULL && println("  - processing dimension $d")
                msup = missing_supported[dIdx]
                if msup # missing is support, I consider all non-missing y rows and all dimensions..
                    nmy  = nonMissingMask[:,d]
                    y    = identity.(X[nmy,d]) # otherwise for some models it remains a classification
                    ty   = nonmissingtype(eltype(y))
                    y    = convert(Vector{ty},y)
                    Xd   = Matrix(Xout[nmy,[1:(d-1);(d+1):end]])
                    x_used_cols[d] = setdiff(collect(1:nC),d)
                else # missing is NOT supported, I consider only cols with nonmissing data in rows to impute and full rows in the remaining cols
                    nmy  = nonMissingMask[:,d]
                    # Step 1 removing cols with missing values in the rows that we will need to impute (i.e. that are also missing in the the y col)..
                    # I need to remove col and not row, as I need to impute this value, I can't just skip the row
                    candidates_d = setdiff(collect(1:nC),d)
                    for (ri,r) in enumerate(eachrow(Xout))
                        !nmy[ri] || continue # we want to look only where y is missing to remove cols 
                        for dc in candidates_d
                            if ismissing(r[dc])
                                candidates_d = setdiff(candidates_d,dc)
                            end
                        end
                    end
                    x_used_cols[d] = candidates_d
                    Xd = Xout[:,candidates_d]
                    # Step 2: for training, consider only the rows where not-dropped cols values are all nonmissing
                    nmxrows = [all(.! ismissing.(r)) for r in eachrow(Xd)]
                    nmrows = nmxrows .& nmy # non missing both in Y and remained X rows

                    y    = identity.(X[nmrows,d]) # otherwise for some models it remains a classification
                    ty   = nonmissingtype(eltype(y))
                    y    = convert(Vector{ty},y)
                    tX   = nonmissingtype(eltype(Xd))
                    Xd   = convert(Matrix{tX},Matrix(Xd[nmrows,:]))

                end
                dmodel = deepcopy(estimators[imputation,dIdx])
                fit_functions[dIdx](dmodel,Xd,y)

                # imputing missing values in d...
                for i in 1:nR
                    if ! missingMask[i,d]
                        continue
                    end
                    xrow = Vector(Xout[i,x_used_cols[d]])
                    if !msup # no missing supported, the row shoudn't contain missing values
                        xrow = Utils.disallowmissing(xrow)
                    end
                    yest = predict_functions[dIdx](dmodel,xrow)
                    # handling some particualr cases... 
                    if typeof(yest) <: AbstractMatrix
                        yest = yest[1,1]
                    elseif typeof(yest) <: AbstractVector
                        yest = yest[1]
                    end
                    if typeof(yest) <: AbstractVector{<:AbstractDict}
                        yest = mode(yest[1],rng=rng)
                    elseif typeof(yest) <: AbstractDict
                        yest = mode(yest,rng=rng)
                    end
                    
                    if ty <: Int 
                        if typeof(yest) <: AbstractString
                            yest = parse(ty,yest)
                        elseif typeof(yest) <: Number
                            yest = Int(round(yest))
                        else
                            error("I don't know how to convert this type $(typeof(yest)) to an integer!")
                        end
                    end

                    Xout_passage[i,d] = yest
                    #return Xout
                end
                # This is last passage: save the model
                if pass == recursive_passages 
                    estimators[imputation,dIdx] = dmodel 
                end
            end # end dimension
            Xout = copy(Xout_passage)
        end # end recursive passage pass
        imputed[imputation]   = Xout
    end # end individual imputation
    m.par = GeneralImputer_lp(estimators,cols2imp,x_used_cols)
    if cache
        if multiple_imputations == 1
            m.cres = Utils.disallowmissing(imputed[1])
        else
            m.cres = Utils.disallowmissing.(imputed)
        end
    end 
    m.info["n_imputed_values"] = n_imputed_values
    m.fitted = true
    return cache ? m.cres : nothing
end


"""
$(TYPEDSIGNATURES)


"""
function predict(m::VariableImportance,X,y)

end

function show(io::IO, ::MIME"text/plain", m::VariableImportance)
    if m.fitted == false
        print(io,"VariableImportance - A meta-model to extract variable importance of an arbitrary regressor/classifier (unfitted)")
    else
        print(io,"VariableImportance - A meta-model to extract variable importance of an arbitrary regressor/classifier (fitted)")
    end
end

function show(io::IO, m::VariableImportance)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"VariableImportance - A meta-model to extract variable importance of an arbitrary regressor/classifier (unfitted)")
    else
        print(io,"VariableImportance - A meta-model to extract variable importance of an arbitrary regressor/classifier (fitted)")
        println(io,m.info)
    end
end

# ------------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Compute the variance-analysis based Sobol index.

Provided the first input is the model output with all the variables (dimensions) considered and the second input it the model output with the variable _j_ removed, the Sobol index returns the reduction in output explained variance if the jth output variable is removed, i.e. higher values highliths a more important variable

# Example
```julia
julia> ŷ   = [1.0, 2.4, 1.5, 1.8];
julia> ŷ₋₁ = [0.8, 2.5, 1.5, 1.7];
julia> ŷ₋₂ = [0.7,2.6,1.4,1.6];
julia> sobol_index(ŷ,ŷ₋₁)
0.03892944038929439
julia> sobol_index(ŷ,ŷ₋₂)
0.15571776155717756
```
"""
# https://towardsdatascience.com/variable-importance-in-random-forests-20c6690e44e0
# https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis
function sobol_index(ŷ,ŷ₋ⱼ)
    mean_prediction = mean(ŷ)
    return sum(ŷ .- ŷ₋ⱼ)^2/sum((ŷ .- mean_prediction).^2)
end

