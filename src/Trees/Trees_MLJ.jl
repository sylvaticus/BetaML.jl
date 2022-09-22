"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Decision Trees/Random Forests models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export DecisionTreeRegressor, RandomForestRegressor, DecisionTreeClassifier, RandomForestClassifier


# ------------------------------------------------------------------------------
# Model Structure declarations..

"""
$(TYPEDEF)

A simple Decision Tree for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

"""
mutable struct DecisionTreeRegressor <: MMI.Deterministic
    "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `0`, i.e. no limits]"
    max_depth::Int64
    "The minimum information gain to allow for a node's partition [def: `0`]"
    min_gain::Float64
    "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
    min_records::Int64
    "The maximum number of (random) features to consider at each partitioning [def: `0`, i.e. look at all features]"
    max_features::Int64
    "This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `variance`]. Either `variance` or a custom function. It can also be an anonymous function."
    splitting_criterion::Function
    "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG
end
DecisionTreeRegressor(;
   max_depth=0, #typemax(Int)
   min_gain=0.0,
   min_records=2,
   max_features=0,
   splitting_criterion=variance,
   rng = Random.GLOBAL_RNG,
   ) = DecisionTreeRegressor(max_depth,min_gain,min_records,max_features,splitting_criterion,rng)

"""
$(TYPEDEF)

A simple Decision Tree for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

"""
mutable struct DecisionTreeClassifier <: MMI.Probabilistic
   "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `0`, i.e. no limits]"
   max_depth::Int64
   "The minimum information gain to allow for a node's partition [def: `0`]"
   min_gain::Float64
   "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
   min_records::Int64
   "The maximum number of (random) features to consider at each partitioning [def: `0`, i.e. look at all features]"
   max_features::Int64
   "This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `gini`]. Either `gini`, `entropy` or a custom function. It can also be an anonymous function."
   splitting_criterion::Function
   "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG
end
DecisionTreeClassifier(;
  max_depth=0,
  min_gain=0.0,
  min_records=2,
  max_features=0,
  splitting_criterion=gini,
  rng = Random.GLOBAL_RNG,
  ) = DecisionTreeClassifier(max_depth,min_gain,min_records,max_features,splitting_criterion,rng)

"""
$(TYPEDEF)

A simple Random Forest for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

"""
mutable struct RandomForestRegressor <: MMI.Deterministic
   n_trees::Int64
   "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `0`, i.e. no limits]"
   max_depth::Int64
   "The minimum information gain to allow for a node's partition [def: `0`]"
   min_gain::Float64
   "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
   min_records::Int64
   "The maximum number of (random) features to consider at each partitioning [def: `0`, i.e. square root of the data dimension]"
   max_features::Int64
   "This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `variance`]. Either `variance` or a custom function. It can also be an anonymous function."
   splitting_criterion::Function
   "Parameter that regulate the weights of the scoring of each tree, to be (optionally) used in prediction based on the error of the individual trees computed on the records on which trees have not been trained. Higher values favour \"better\" trees, but too high values will cause overfitting [def: `0`, i.e. uniform weigths]"
   β::Float64
   "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG
end
RandomForestRegressor(;
  n_trees=30,
  max_depth=0,
  min_gain=0.0,
  min_records=2,
  max_features=0,
  splitting_criterion=variance,
  β=0.0,
  rng = Random.GLOBAL_RNG,
  ) = RandomForestRegressor(n_trees,max_depth,min_gain,min_records,max_features,splitting_criterion,β,rng)

"""
$(TYPEDEF)

A simple Random Forest for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

"""
mutable struct RandomForestClassifier <: MMI.Probabilistic
    n_trees::Int64
    "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `0`, i.e. no limits]"
    max_depth::Int64
    "The minimum information gain to allow for a node's partition [def: `0`]"
    min_gain::Float64
    "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
    min_records::Int64
    "The maximum number of (random) features to consider at each partitioning [def: `0`, i.e. square root of the data dimensions]"
    max_features::Int64
    "This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `gini`]. Either `gini`, `entropy` or a custom function. It can also be an anonymous function."
    splitting_criterion::Function
    "Parameter that regulate the weights of the scoring of each tree, to be (optionally) used in prediction based on the error of the individual trees computed on the records on which trees have not been trained. Higher values favour \"better\" trees, but too high values will cause overfitting [def: `0`, i.e. uniform weigths]"
    β::Float64
    "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG
end
RandomForestClassifier(;
    n_trees=30,
    max_depth=0,
    min_gain=0.0,
    min_records=2,
    max_features=0,
    splitting_criterion=gini,
    β=0.0,
    rng = Random.GLOBAL_RNG,
) = RandomForestClassifier(n_trees,max_depth,min_gain,min_records,max_features,splitting_criterion,β,rng)

#=
# skipped for now..
# ------------------------------------------------------------------------------
# Hyperparameters ranges definition (for automatic tuning)

MMI.hyperparameter_ranges(::Type{<:DecisionTreeRegressor}) = (
#    (range(Float64, :alpha, lower=0, upper=1, scale=:log),
#     range(Int, :beta, lower=1, upper=Inf, origin=100, unit=50, scale=:log),
#         nothing)
    range(Int64,:max_depth,lower=0,upper=Inf,scale=:log),
    range(Float64,:min_gain,lower=0,upper=Inf,scale=:log),
    range(Int64,:min_records,lower=0,upper=Inf,scale=:log),
    range(Int64,:max_features,lower=0,upper=Inf,scale=:log),
    nothing
)
=#

# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(model::Union{DecisionTreeRegressor,RandomForestRegressor}, verbosity, X, y)
   x = MMI.matrix(X)                     # convert table to matrix
   max_depth         = model.max_depth == 0 ? size(x,1) : model.max_depth
   # Using low level API here. We could switch to APIV2...
   if (typeof(model) == DecisionTreeRegressor)
       max_features = model.max_features == 0 ? size(x,2) : model.max_features
       fitresult   = buildTree(x, y, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion,rng=model.rng)
   else
       max_features = model.max_features == 0 ? Int(round(sqrt(size(x,2)))) : model.max_features
       fitresult   = buildForest(x, y, model.n_trees, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion, β=model.β,rng=model.rng)
   end
   cache=nothing
   report=nothing
   return fitresult, cache, report
end

function MMI.fit(model::Union{DecisionTreeClassifier,RandomForestClassifier}, verbosity, X, y)
   x                = MMI.matrix(X)                        # convert table to matrix
   a_target_element = y[1]                                 # a CategoricalValue or CategoricalString
   #y_plain          = MMI.int(y) .- 1                     # integer relabeling should start at 0
   yarray           = convert(Vector{eltype(levels(y))},y) # convert to a simple Array{T}
   max_depth         = model.max_depth == 0 ? size(x,1) : model.max_depth
   # Using low level API here. We could switch to APIV2...
   if (typeof(model) == DecisionTreeClassifier)
       max_features   = model.max_features == 0 ? size(x,2) : model.max_features
       fittedmodel   = buildTree(x, yarray, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion, force_classification=true,rng=model.rng)
   else
       max_features   = model.max_features == 0 ? Int(round(sqrt(size(x,2)))) : model.max_features
       fittedmodel   = buildForest(x, yarray, model.n_trees, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion, force_classification=true, β=model.β,rng=model.rng)
   end
   cache            = nothing
   report           = nothing
   fitresult        = (fittedmodel,a_target_element)
   return (fitresult, cache, report)
end


# ------------------------------------------------------------------------------
# Predict functions....

MMI.predict(model::Union{DecisionTreeRegressor,RandomForestRegressor}, fitresult, Xnew) = Trees.predict(fitresult, MMI.matrix(Xnew))

function MMI.predict(model::Union{DecisionTreeClassifier,RandomForestClassifier}, fitresult, Xnew)
    fittedModel      = fitresult[1]
    a_target_element = fitresult[2]
    decode           = MMI.decoder(a_target_element)
    classes          = MMI.classes(a_target_element)
    nLevels          = length(classes)
    nRecords         = MMI.nrows(Xnew)
    treePredictions  = Trees.predict(fittedModel, MMI.matrix(Xnew),rng=model.rng)
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(classes)
            predMatrix[n,c] = get(treePredictions[n],string(cl),0.0)
        end
    end
    predictions = MMI.UnivariateFinite(classes, predMatrix)
    return predictions
end

# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(DecisionTreeRegressor,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    target_scitype   = AbstractVector{<: MMI.Continuous},           # for a supervised model, what target?
    supports_weights = false,                                       # does the model support sample weights?
    descr            = "A simple Decision Tree for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.DecisionTreeRegressor"
    )
MMI.metadata_model(RandomForestRegressor,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    target_scitype   = AbstractVector{<: MMI.Continuous},
    supports_weights = false,
    descr            = "A simple Random Forest ensemble for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.RandomForestRegressor"
    )
MMI.metadata_model(DecisionTreeClassifier,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    target_scitype   = AbstractVector{<: Union{MMI.Missing,MMI.Finite}},
    supports_weights = false,
    descr            = "A simple Decision Tree for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.DecisionTreeClassifier"
    )
MMI.metadata_model(RandomForestClassifier,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    target_scitype   = AbstractVector{<: Union{MMI.Missing,MMI.Finite}},
    supports_weights = false,
    descr            = "A simple Random Forest ensemble for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.RandomForestClassifier"
    )
