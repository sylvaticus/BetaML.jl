"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Decision Trees/Random Forests models

export DecisionTreeRegressor, RandomForestRegressor, DecisionTreeClassifier, RandomForestClassifier


# ------------------------------------------------------------------------------
# Model Structure declarations..

"""
$(TYPEDEF)

A simple Decision Tree model for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:

```julia
julia> using MLJ

julia> X, y        = @load_boston;

julia> modelType   = @load DecisionTreeRegressor pkg = "BetaML" verbosity=0
BetaML.Trees.DecisionTreeRegressor

julia> model       = modelType()
DecisionTreeRegressor(
  max_depth = 0, 
  min_gain = 0.0, 
  min_records = 2, 
  max_features = 0, 
  splitting_criterion = BetaML.Utils.variance, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);
[ Info: Training machine(DecisionTreeRegressor(max_depth = 0, …), …).

julia> ŷ           = predict(mach, X);

julia> hcat(y,ŷ)
506×2 Matrix{Float64}:
 24.0  26.35
 21.6  21.6
 34.7  34.8
  ⋮    
 23.9  23.75
 22.0  22.2
 11.9  13.2
```

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
   splitting_criterion=BetaML.Utils.variance,
   rng = Random.GLOBAL_RNG,
   ) = DecisionTreeRegressor(max_depth,min_gain,min_records,max_features,splitting_criterion,rng)

"""
$(TYPEDEF)

A simple Decision Tree model for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:
```julia
julia> using MLJ

julia> X, y        = @load_iris;

julia> modelType   = @load DecisionTreeClassifier pkg = "BetaML" verbosity=0
BetaML.Trees.DecisionTreeClassifier

julia> model       = modelType()
DecisionTreeClassifier(
  max_depth = 0, 
  min_gain = 0.0, 
  min_records = 2, 
  max_features = 0, 
  splitting_criterion = BetaML.Utils.gini, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);
[ Info: Training machine(DecisionTreeClassifier(max_depth = 0, …), …).

julia> cat_est    = predict(mach, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>1.0, versicolor=>0.0, virginica=>0.0)
 UnivariateFinite{Multiclass{3}}(setosa=>1.0, versicolor=>0.0, virginica=>0.0)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>0.0, versicolor=>0.0, virginica=>1.0)
 UnivariateFinite{Multiclass{3}}(setosa=>0.0, versicolor=>0.0, virginica=>1.0)
 UnivariateFinite{Multiclass{3}}(setosa=>0.0, versicolor=>0.0, virginica=>1.0)
```

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
  splitting_criterion=BetaML.Utils.gini,
  rng = Random.GLOBAL_RNG,
  ) = DecisionTreeClassifier(max_depth,min_gain,min_records,max_features,splitting_criterion,rng)

"""
$(TYPEDEF)

A simple Random Forest model for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example: 
```julia
julia> using MLJ

julia> X, y        = @load_boston;

julia> modelType   = @load RandomForestRegressor pkg = "BetaML" verbosity=0
BetaML.Trees.RandomForestRegressor

julia> model       = modelType()
RandomForestRegressor(
  n_trees = 30, 
  max_depth = 0, 
  min_gain = 0.0, 
  min_records = 2, 
  max_features = 0, 
  splitting_criterion = BetaML.Utils.variance, 
  β = 0.0, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);
[ Info: Training machine(RandomForestRegressor(n_trees = 30, …), …).

julia> ŷ           = predict(mach, X);

julia> hcat(y,ŷ)
506×2 Matrix{Float64}:
 24.0  25.8433
 21.6  22.4317
 34.7  35.5742
 33.4  33.9233
  ⋮    
 23.9  24.42
 22.0  22.4433
 11.9  15.5833
```
"""
mutable struct RandomForestRegressor <: MMI.Deterministic
   "Number of (decision) trees in the forest [def: `30`]"
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
  splitting_criterion=BetaML.Utils.variance,
  β=0.0,
  rng = Random.GLOBAL_RNG,
  ) = RandomForestRegressor(n_trees,max_depth,min_gain,min_records,max_features,splitting_criterion,β,rng)

"""
$(TYPEDEF)

A simple Random Forest model for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example :
```julia
julia> using MLJ

julia> X, y        = @load_iris;

julia> modelType   = @load RandomForestClassifier pkg = "BetaML" verbosity=0
BetaML.Trees.RandomForestClassifier

julia> model       = modelType()
RandomForestClassifier(
  n_trees = 30, 
  max_depth = 0, 
  min_gain = 0.0, 
  min_records = 2, 
  max_features = 0, 
  splitting_criterion = BetaML.Utils.gini, 
  β = 0.0, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);
[ Info: Training machine(RandomForestClassifier(n_trees = 30, …), …).

julia> cat_est    = predict(mach, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt32, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>1.0, versicolor=>0.0, virginica=>0.0)
 UnivariateFinite{Multiclass{3}}(setosa=>1.0, versicolor=>0.0, virginica=>0.0)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>0.0, versicolor=>0.0, virginica=>1.0)
 UnivariateFinite{Multiclass{3}}(setosa=>0.0, versicolor=>0.0667, virginica=>0.933)
```
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
    splitting_criterion=BetaML.Utils.gini,
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
   typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
   verbosity = mljverbosity_to_betaml_verbosity(verbosity)
   max_depth         = model.max_depth == 0 ? size(x,1) : model.max_depth
   # Using low level API here. We could switch to APIV2...
   if (typeof(model) == DecisionTreeRegressor)
       max_features = model.max_features == 0 ? size(x,2) : model.max_features
       fitresult   = BetaML.Trees.buildTree(x, y, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion,rng=model.rng, verbosity=verbosity)
   else
       max_features = model.max_features == 0 ? Int(round(sqrt(size(x,2)))) : model.max_features
       fitresult   = BetaML.Trees.buildForest(x, y, model.n_trees, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion, β=model.β,rng=model.rng,verbosity=verbosity)
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
   typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
   verbosity = mljverbosity_to_betaml_verbosity(verbosity)
   max_depth         = model.max_depth == 0 ? size(x,1) : model.max_depth
   # Using low level API here. We could switch to APIV2...
   if (typeof(model) == DecisionTreeClassifier)
       max_features   = model.max_features == 0 ? size(x,2) : model.max_features
       fittedmodel   = BetaML.Trees.buildTree(x, yarray, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion, force_classification=true,rng=model.rng, verbosity=verbosity)
   else
       max_features   = model.max_features == 0 ? Int(round(sqrt(size(x,2)))) : model.max_features
       fittedmodel   = BetaML.Trees.buildForest(x, yarray, model.n_trees, max_depth=max_depth, min_gain=model.min_gain, min_records=model.min_records, max_features=max_features, splitting_criterion=model.splitting_criterion, force_classification=true, β=model.β,rng=model.rng, verbosity=verbosity)
   end
   cache            = nothing
   report           = nothing
   fitresult        = (fittedmodel,a_target_element)
   return (fitresult, cache, report)
end


# ------------------------------------------------------------------------------
# Predict functions....

MMI.predict(model::Union{DecisionTreeRegressor,RandomForestRegressor}, fitresult, Xnew) = BetaML.Trees.predict(fitresult, MMI.matrix(Xnew))

function MMI.predict(model::Union{DecisionTreeClassifier,RandomForestClassifier}, fitresult, Xnew)
    fittedModel      = fitresult[1]
    a_target_element = fitresult[2]
    decode           = MMI.decoder(a_target_element)
    classes          = MMI.classes(a_target_element)
    nLevels          = length(classes)
    nRecords         = MMI.nrows(Xnew)
    treePredictions  = BetaML.Trees.predict(fittedModel, MMI.matrix(Xnew),rng=model.rng)
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
    input_scitype = Union{
        MMI.Table(Union{MMI.Known,MMI.Missing}),
        AbstractMatrix{<:Union{MMI.Known,MMI.Missing}},
    },
    target_scitype   = AbstractVector{<: MMI.Continuous},           # for a supervised model, what target?
    supports_weights = false,                                       # does the model support sample weights?
	load_path        = "BetaML.Bmlj.DecisionTreeRegressor"
    )
MMI.metadata_model(RandomForestRegressor,
    input_scitype = Union{
        MMI.Table(Union{MMI.Known,MMI.Missing}),
        AbstractMatrix{<:Union{MMI.Known,MMI.Missing}},
    },
    target_scitype   = AbstractVector{<: MMI.Continuous},
    supports_weights = false,
	load_path        = "BetaML.Bmlj.RandomForestRegressor"
    )
MMI.metadata_model(DecisionTreeClassifier,
    input_scitype = Union{
        MMI.Table(Union{MMI.Known,MMI.Missing}),
        AbstractMatrix{<:Union{MMI.Known,MMI.Missing}},
    },
    target_scitype   = AbstractVector{<: Union{MMI.Missing,MMI.Finite}},
    supports_weights = false,
	load_path        = "BetaML.Bmlj.DecisionTreeClassifier"
    )
MMI.metadata_model(RandomForestClassifier,
    input_scitype = Union{
        MMI.Table(Union{MMI.Known,MMI.Missing}),
        AbstractMatrix{<:Union{MMI.Known,MMI.Missing}},
    },
    target_scitype   = AbstractVector{<: Union{MMI.Missing,MMI.Finite}},
    supports_weights = false,
	load_path        = "BetaML.Bmlj.RandomForestClassifier"
    )
