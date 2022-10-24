"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."
      
# ------------------------------------------------------------------------------
# TYPE HIERARCHY AND DEFINITIONS

"""
    Forest{Ty}

TLow level type representing a Random Forest.

Individual trees are stored in the array `trees`. The "type" of the forest is given by the type of the labels on which it has been trained.

# Struct members:
- `trees`:        The individual Decision Trees
- `is_regression`: Whether the forest is to be used for regression jobs or classification
- `oobData`:      For each tree, the rows number if the data that have _not_ being used to train the specific tree
- `ooberror`:     The out of bag error (if it has been computed)
- `weights`:      A weight for each tree depending on the tree's score on the oobData (see [`buildForest`](@ref))
"""
mutable struct Forest{Ty} <: BetaMLLearnableParametersSet
    trees::Array{Union{AbstractDecisionNode,Leaf{Ty}},1}
    is_regression::Bool
    oobData::Array{Array{Int64,1},1}
    ooberror::Float64
    weights::Array{Float64,1}
end

# Api V2..

"""

$(TYPEDEF)

Hyperparameters for [`RandomForestEstimator`](@ref) (Random Forest).

## Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct RFHyperParametersSet <: BetaMLHyperParametersSet
    "Number of (decision) trees in the forest [def: `30`]"
    n_trees::Int64                               = 30
    "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `nothing`, i.e. no limits]"
    max_depth::Union{Nothing,Int64}              = nothing
    "The minimum information gain to allow for a node's partition [def: `0`]"
    min_gain::Float64                            = 0.0
    "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
    min_records::Int64                           = 2
    "The maximum number of (random) features to consider when choosing the optimal partition of the dataset [def: `nothing`, i.e. square root of the dimensions of the training data`]"
    max_features::Union{Nothing,Int64}           = nothing
    "Whether to force a classification task even if the labels are numerical (typically when labels are integers encoding some feature rather than representing a real cardinal measure) [def: `false`]"
    force_classification::Bool                   = false
    "Either `gini`, `entropy` or `variance`. This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `nothing`, i.e. `gini` for categorical labels (classification task) and `variance` for numerical labels(regression task)]. It can be an anonymous function."
    splitting_criterion::Union{Nothing,Function} = nothing
    "Use an experimental faster algoritm for looking up the best split in ordered fields (colums). Currently it brings down the fitting time of an order of magnitude, but predictions are sensibly affected. If used, control the meaning of integer fields with `integer_encoded_cols`."
    fast_algorithm::Bool                         = false
    "A vector of columns positions to specify which integer columns should be treated as encoding of categorical variables insteads of ordered classes/values. [def: `nothing`, integer columns with less than 20 unique values are considered categorical]. Useful in conjunction with `fast_algorithm`, little difference otherwise."
    integer_encoded_cols::Union{Nothing,Array{Int64,1}} =nothing
    "Parameter that regulate the weights of the scoring of each tree, to be (optionally) used in prediction based on the error of the individual trees computed on the records on which trees have not been trained. Higher values favour \"better\" trees, but too high values will cause overfitting [def: `0`, i.e. uniform weigths]"
    beta::Float64                               = 0.0
    "Wheter to compute the _Out-Of-Bag_ error, an estimation of the validation error (the mismatching error for classification and the relative mean error for regression jobs)."
    oob::Bool                                   = false
    """
    The method - and its parameters - to employ for hyperparameters autotuning.
    See [`SuccessiveHalvingSearch`](@ref) for the default method.
    To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and eventually change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
    """
    tunemethod::AutoTuneMethod                  = SuccessiveHalvingSearch(hpranges=Dict("n_trees" => [10, 20, 30, 40], "max_depth" =>[5,10,nothing], "min_gain"=>[0.0, 0.1, 0.5], "min_records"=>[2,3,5],"max_features"=>[nothing,5,10,30],"beta"=>[0,0.01,0.1]),multithreads=false) # RF are already MT
end

Base.@kwdef mutable struct RFLearnableParameters <: BetaMLLearnableParametersSet
    forest::Union{Nothing,Forest} = nothing #TODO: Forest contain info that is actualy in report. Currently we duplicate, we should just remove them from par by making a dedicated struct instead of Forest
    Ty::DataType = Any
end


"""
$(TYPEDEF)

A Random Forest classifier and regressor (supervised).

Random forests are _ensemble_ of Decision Trees models (see [`?DecisionTreeEstimator`](@ref DecisionTreeEstimator)).

For the parameters see [`?RFHyperParametersSet`](@ref RFHyperParametersSet) and [`?BetaMLDefaultOptionsSet`](@ref BetaMLDefaultOptionsSet).

# Notes :
- Each individual decision tree is built using bootstrap over the data, i.e. "sampling N records with replacement" (hence, some records appear multiple times and some records do not appear in the specific tree training). The `maxx_feature` injects further variability and reduces the correlation between the forest trees.
- The predictions of the "forest" (using the function `predict()`) are then the aggregated predictions of the individual trees (from which the name "bagging": **b**oostrap **agg**regat**ing**).
- The performances of each individual trees,  as measured using the records they have not being trained with, can then be (optionally) used as weights in the `predict` function. The parameter `beta ≥ 0` regulate the distribution of these weights: larger is `β`, the greater the importance (hence the weights) attached to the best-performing trees compared to the low-performing ones. Using these weights can significantly improve the forest performances (especially using small forests), however the correct value of `beta` depends on the problem under exam (and the chosen caratteristics of the random forest estimator) and should be cross-validated to avoid over-fitting.
- Note that training `RandomForestEstimator` uses multiple threads if these are available. You can check the number of threads available with `Threads.nthreads()`. To set the number of threads in Julia either set the environmental variable `JULIA_NUM_THREADS` (before starting Julia) or start Julia with the command line option `--threads` (most integrated development editors for Julia already set the number of threads to 4).
- Online fitting (re-fitting with new data) is not supported
- Missing data (in the feature dataset) is supported.

"""
mutable struct RandomForestEstimator <: BetaMLSupervisedModel
    hpar::RFHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,RFLearnableParameters} 
    cres
    fitted::Bool
    info::Dict{String,Any}
end

function RandomForestEstimator(;kwargs...)
m              = RandomForestEstimator(RFHyperParametersSet(),BetaMLDefaultOptionsSet(),RFLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

# ------------------------------------------------------------------------------
# MODEL ALGORITHMS AND TRAINING

"""
   buildForest(x, y, n_trees; max_depth, min_gain, min_records, max_features, splitting_criterion, force_classification)

Builds (define and train) a "forest" of Decision Trees.

!!! warning
    Direct usage of this low-level function is deprecated and it has been unexported in BetaML 0.9.
    Use [`RandomForestEstimator`](@ref) instead. 

# Parameters:
See [`buildTree`](@ref). The function has all the parameters of `bildTree` (with the `max_features` defaulting to `√D` instead of `D`) plus the following parameters:
- `n_trees`: Number of trees in the forest [def: `30`]
- `β`: Parameter that regulate the weights of the scoring of each tree, to be (optionally) used in prediction (see later) [def: `0`, i.e. uniform weigths]
- `oob`: Whether to coompute the out-of-bag error, an estimation of the generalization accuracy [def: `false`]
- `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Output:
- The function returns a Forest object.
- The forest weights default to array of ones if `β ≤ 0` and the oob error to `+Inf` if `oob` == `false`.

# Notes :
- Each individual decision tree is built using bootstrap over the data, i.e. "sampling N records with replacement" (hence, some records appear multiple times and some records do not appear in the specific tree training). The `maxFeature` injects further variability and reduces the correlation between the forest trees.
- The predictions of the "forest" (using the function `predict()`) are then the aggregated predictions of the individual trees (from which the name "bagging": **b**oostrap **agg**regat**ing**).
- This function optionally reports a weight distribution of the performances of eanch individual trees, as measured using the records he has not being trained with. These weights can then be (optionally) used in the `predict` function. The parameter `β ≥ 0` regulate the distribution of these weights: larger is `β`, the greater the importance (hence the weights) attached to the best-performing trees compared to the low-performing ones. Using these weights can significantly improve the forest performances (especially using small forests), however the correct value of β depends on the problem under exam (and the chosen caratteristics of the random forest estimator) and should be cross-validated to avoid over-fitting.
- Note that this function uses multiple threads if these are available. You can check the number of threads available with `Threads.nthreads()`. To set the number of threads in Julia either set the environmental variable `JULIA_NUM_THREADS` (before starting Julia) or start Julia with the command line option `--threads` (most integrated development editors for Julia already set the number of threads to 4).
"""
function buildForest(x, y::AbstractArray{Ty,1}, n_trees=30; max_depth = size(x,1), min_gain=0.0, min_records=2, max_features=Int(round(sqrt(size(x,2)))), force_classification=false, splitting_criterion = (Ty <: Number && !force_classification) ? variance : gini, integer_encoded_cols=nothing, fast_algorithm=false, β=0, oob=false,rng = Random.GLOBAL_RNG) where {Ty}
    # Force what would be a regression task into a classification task
    if force_classification && Ty <: Number
        y = string.(y)
    end
    trees            = Array{Union{AbstractDecisionNode,Leaf{Ty}},1}(undef,n_trees)
    notSampledByTree = Array{Array{Int64,1},1}(undef,n_trees) # to later compute the Out of Bag Error

    errors = Float64[]

    jobIsRegression = (force_classification || !(eltype(y) <: Number )) ? false : true # we don't need the tertiary operator here, but it is more clear with it...
    (N,D) = size(x)

    if isnothing(integer_encoded_cols)
        integer_encoded_cols = Int64[]
        for (d,c) in enumerate(eachcol(x))
            if(all(isinteger_bml.(skipmissing(c)))) && length(unique(skipmissing(c))) < 20 # hardcoded: when using automatic identifier of integer encoded cols, if more than XX values, we consider that is not a categorical variable 
              push!(integer_encoded_cols,d)
            end
        end
    end

    masterSeed = rand(rng,100:9999999999999) ## Some RNG have problems with very small seed. Also, the master seed has to be computed _before_ generate_parallel_rngs
    rngs = generate_parallel_rngs(rng,Threads.nthreads())

    #for i in 1:n_trees # for easier debugging/profiling...
    Threads.@threads for i in 1:n_trees
        tsrng = rngs[Threads.threadid()] # Thread safe random number generator
        Random.seed!(tsrng,masterSeed+i*10)
        toSample = rand(tsrng, 1:N,N)
        notToSample = setdiff(1:N,toSample)
        bootstrappedx = x[toSample,:] # "boosted is different than "bootstrapped": https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f
        bootstrappedy = y[toSample]
        #controlx = x[notToSample,:]
        #controly = y[notToSample]
        tree = buildTree(bootstrappedx, bootstrappedy; max_depth = max_depth, min_gain=min_gain, min_records=min_records, max_features=max_features, splitting_criterion = splitting_criterion, force_classification=force_classification, integer_encoded_cols=integer_encoded_cols, fast_algorithm=fast_algorithm, rng = tsrng)
        #ŷ = predict(tree,controlx)
        trees[i] = tree
        notSampledByTree[i] = notToSample
    end

    weights = ones(Float64,n_trees)
    if β > 0
        weights = updateTreesWeights!(Forest{Ty}(trees,jobIsRegression,notSampledByTree,0.0,weights), x, y, β=β, rng=rng)
    end
    oobe = +Inf
    if oob
        oobe = ooberror(Forest{Ty}(trees,jobIsRegression,notSampledByTree,0.0,weights),x,y,rng=rng)
    end
    return Forest{Ty}(trees,jobIsRegression,notSampledByTree,oobe,weights)
end

# API V2

"""
$(TYPEDSIGNATURES)

Fit a [`RandomForestEstimator`](@ref) to the data

"""
function fit!(m::RandomForestEstimator,x,y::AbstractArray{Ty,1}) where {Ty}

    if m.fitted
        @warn "This model has already been fitted and it doesn't support multiple training. This training will override the previous one(s)"
    else
        autotune!(m,(x,y))
    end

    Tynm = nonmissingtype(Ty)
    # Setting default parameters that depends from the data...
    max_depth    = m.hpar.max_depth    == nothing ?  size(x,1) : m.hpar.max_depth
    max_features = m.hpar.max_features == nothing ?  Int(round(sqrt(size(x,2)))) : m.hpar.max_features
    splitting_criterion = m.hpar.splitting_criterion == nothing ? ( (Tynm <: Number && !m.hpar.force_classification) ? variance : gini) : m.hpar.splitting_criterion

    if (Tynm <: Integer && m.hpar.force_classification)
        y = convert.(BetaMLClass,y)
    end

    # Setting schortcuts to other hyperparameters/options....
    min_gain             = m.hpar.min_gain
    min_records          = m.hpar.min_records
    force_classification = m.hpar.force_classification
    n_trees              = m.hpar.n_trees
    fast_algorithm       = m.hpar.fast_algorithm
    integer_encoded_cols = m.hpar.integer_encoded_cols
    β                   = m.hpar.beta
    oob                 = m.hpar.oob
    cache               = m.opt.cache
    rng                 = m.opt.rng
    verbosity           = m.opt.verbosity
    
    forest = buildForest(x, y, n_trees; max_depth = max_depth, min_gain=min_gain, min_records=min_records, max_features=max_features, force_classification=force_classification, splitting_criterion = splitting_criterion, fast_algorithm=fast_algorithm, integer_encoded_cols=integer_encoded_cols, β=β, oob=false,  rng = rng)

    m.par = RFLearnableParameters(forest,Tynm)

    if cache
        rawout = predictSingle.(Ref(forest),eachrow(x),rng=rng)
        if (Tynm <: Integer && m.hpar.force_classification)
           out = [ Dict([convert(Tynm,k) => v for (k,v) in e]) for e in rawout]
        else
           out = rawout
        end
        m.cres = out
     else
        m.cres = nothing
     end
    
    if oob
        m.par.forest.ooberror = ooberror(m.par.forest,x,y;rng = rng) 
    end

    m.fitted = true
    
    m.info["fitted_records"]   = size(x,1)
    m.info["xndims"]           = max_features
    m.info["jobIsRegression"]  = m.par.forest.is_regression ? 1 : 0
    m.info["oob_errors"]       = m.par.forest.ooberror
    depths = vcat([transpose([computeDepths(tree)[1],computeDepths(tree)[2]]) for tree in m.par.forest.trees]...)
    (m.info["avgAvgDepth"],m.info["avgMmax_depth"]) = mean(depths,dims=1)[1], mean(depths,dims=1)[2]
    return cache ? m.cres : nothing
end

# ------------------------------------------------------------------------------
# MODEL PREDICTIONS 

# Optionally a weighted mean of tree's prediction is used if the parameter `weights` is given.
"""
    predictSingle(forest,x)

Predict the label of a single feature record. See [`predict`](@ref).
"""
function predictSingle(forest::Forest{Ty}, x; rng = Random.GLOBAL_RNG) where {Ty}
    trees   = forest.trees
    weights = forest.weights
    predictions  = predictSingle.(trees,Ref(x),rng=rng)
    if eltype(predictions) <: AbstractDict   # categorical
        #weights = 1 .- treesErrors # back to the accuracy
        return mean_dicts(predictions,weights=weights)
    else
        #weights = exp.( - treesErrors)
        return dot(predictions,weights)/sum(weights)
    end
end


"""
   [predict(forest,x)](@id forest_prediction)

Predict the labels of a feature dataset.

!!! warning
    Direct usage of this low-level function is deprecated and it has been unexported in BetaML 0.9.
    Use [`RandomForestEstimator`](@ref) and the associated `predict(m::Model,x)` function instead.

For each record of the dataset and each tree of the "forest", recursivelly traverse the tree to find the prediction most opportune for the given record.
If the labels the tree has been trained with are numeric, the prediction is also numeric (the mean of the different trees predictions, in turn the mean of the labels of the training records ended in that leaf node).
If the labels were categorical, the prediction is a dictionary with the probabilities of each item and in such case the probabilities of the different trees are averaged to compose the forest predictions. This is a bit different than most other implementations where the mode instead is reported.

In the first case (numerical predictions) use `relative_mean_error(ŷ,y)` to assess the mean relative error, in the second case you can use `accuracy(ŷ,y)`.
"""
function predict(forest::Forest{Ty}, x;rng = Random.GLOBAL_RNG) where {Ty}
    predictions = predictSingle.(Ref(forest),eachrow(x),rng=rng)
    return predictions
end

# API V2...
"""
$(TYPEDSIGNATURES)

Predict the labels associated to some feature data using a trained [`RandomForestEstimator`](@ref)

"""
function predict(m::RandomForestEstimator,x)
    #TODO: get Tynm here! and OrdinalEncoder!
    #Ty = get_parametric_types(m.par)[1] |> nonmissingtype
    Ty = m.par.Ty # this should already be the nonmissing type
    rawout = predictSingle.(Ref(m.par.forest),eachrow(x),rng=m.opt.rng)
    if (Ty <: Integer && m.hpar.force_classification)
        return [ Dict([convert(Ty,k) => v for (k,v) in e]) for e in rawout]
    else
        return rawout
    end
end

# ------------------------------------------------------------------------------
# OTHER (MODEL OPTIONAL PARTS, INFO, VISUALISATION,...)

"""
   updateTreesWeights!(forest,x,y;β)

Update the weights of each tree (to use in the prediction of the forest) based on the error of the individual tree computed on the records on which it has not been trained.

As training a forest is expensive, this function can be used to "just" upgrade the trees weights using different betas, without retraining the model.
"""
function updateTreesWeights!(forest::Forest{Ty},x,y;β=50,rng = Random.GLOBAL_RNG) where {Ty}
    trees            = forest.trees
    notSampledByTree = forest.oobData
    jobIsRegression  = forest.is_regression
    weights          = Float64[]
    for (i,tree) in enumerate(trees)
        yoob = y[notSampledByTree[i]]
        if length(yoob) > 0
            ŷ = predict(tree,x[notSampledByTree[i],:],rng=rng)
            if jobIsRegression
                push!(weights,exp(- β*relative_mean_error(yoob,ŷ)))
            else
                push!(weights,accuracy(yoob,ŷ)*β)
            end
        else  # there has been no data that has not being used for this tree, because by a (rare!) chance all the sampled data for this tree was on a different row
            push!(weights,forest.weights[i])
        end
    end
    forest.weights = weights
    return weights
end

"""
   ooberror(forest,x,y;rng)

Compute the Out-Of-Bag error, an estimation of the validation error.

This function is called at time of train the forest if the parameter `oob` is `true`, or can be used later to get the oob error on an already trained forest.
The oob error reported is the mismatching error for classification and the relative mean error for regression. 
"""
function ooberror(forest::Forest{Ty},x,y;rng = Random.GLOBAL_RNG) where {Ty}
    trees            = forest.trees
    jobIsRegression  = forest.is_regression
    notSampledByTree = forest.oobData
    weights          = forest.weights
    B                = length(trees)
    N                = size(x,1)

    if jobIsRegression
        ŷ = Array{Float64,1}(undef,N)
    else
        ŷ = Array{Dict{Ty,Float64},1}(undef,N)
    end
    # Rarelly a given n has been visited by al lthe trees of the forest, so there is no trees available to compute the oob error
    # This serves as a mask to remove this n from the computation of the oob error
    nMask = fill(true,N)
    for (n,x) in enumerate(eachrow(x))
        unseenTreesBools  = in.(n,notSampledByTree)
        if sum(unseenTreesBools) == 0 # this particular record has been visited by all trees of the forest
            nMask[n] = false
            continue
        end
        unseenTrees = trees[(1:B)[unseenTreesBools]]
        unseenTreesWeights = weights[(1:B)[unseenTreesBools]]
        ŷi   = predictSingle(Forest{Ty}(unseenTrees,jobIsRegression,forest.oobData,0.0,unseenTreesWeights),x,rng=rng)
        if !jobIsRegression && Ty <: Number # we are in the ugly case where we want integers but we have dict of Strings, need to convert
            ŷi   = Dict(map((k,v) -> parse(Int,k)=>v, keys(ŷi), values(ŷi)))
        end
        ŷ[n] = ŷi
    end
    if jobIsRegression
        return relative_mean_error(y[nMask],ŷ[nMask],normdim=false,normrec=false)
    else
        return error(y[nMask],ŷ[nMask])
    end
end

function show(io::IO, ::MIME"text/plain", m::RandomForestEstimator)
    if m.fitted == false
        print(io,"RandomForestEstimator - A $(m.hpar.n_trees) trees Random Forest model (unfitted)")
    else
        job = m.info["jobIsRegression"] == 1 ? "regressor" : "classifier"
        print(io,"RandomForestEstimator - A $(m.hpar.n_trees) trees Random Forest $job (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, m::RandomForestEstimator)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"RandomForestEstimator - A $(m.hpar.n_trees) trees Random Forest model (unfitted)")
    else
        job = m.info["jobIsRegression"] == 1 ? "regressor" : "classifier"
        println(io,"RandomForestEstimator - A $(m.hpar.n_trees) trees Random Forest $job (fitted on $(m.info["fitted_records"]) records)")
        println(io,m.info)
    end
end