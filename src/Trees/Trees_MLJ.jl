# MLJ interface for Decision Trees/Random Forests models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export DecisionTreeRegressor, RandomForestRegressor, DecisionTreeClassifier, RandomForestClassifier


# ------------------------------------------------------------------------------
# Model Structure declarations..

mutable struct DecisionTreeRegressor <: MMI.Deterministic
    maxDepth::Int64
    minGain::Float64
    minRecords::Int64
    maxFeatures::Int64
    splittingCriterion::Function
    rng::AbstractRNG
end
DecisionTreeRegressor(;
   maxDepth=0, #typemax(Int)
   minGain=0.0,
   minRecords=2,
   maxFeatures=0,
   splittingCriterion=variance,
   rng = Random.GLOBAL_RNG,
   ) = DecisionTreeRegressor(maxDepth,minGain,minRecords,maxFeatures,splittingCriterion,rng)

mutable struct DecisionTreeClassifier <: MMI.Probabilistic
   maxDepth::Int64
   minGain::Float64
   minRecords::Int64
   maxFeatures::Int64
   splittingCriterion::Function
   rng::AbstractRNG
end
DecisionTreeClassifier(;
  maxDepth=0,
  minGain=0.0,
  minRecords=2,
  maxFeatures=0,
  splittingCriterion=gini,
  rng = Random.GLOBAL_RNG,
  ) = DecisionTreeClassifier(maxDepth,minGain,minRecords,maxFeatures,splittingCriterion,rng)

mutable struct RandomForestRegressor <: MMI.Deterministic
   nTrees::Int64
   maxDepth::Int64
   minGain::Float64
   minRecords::Int64
   maxFeatures::Int64
   splittingCriterion::Function
   β::Float64
   rng::AbstractRNG
end
RandomForestRegressor(;
  nTrees=30,
  maxDepth=0,
  minGain=0.0,
  minRecords=2,
  maxFeatures=0,
  splittingCriterion=variance,
  β=0.0,
  rng = Random.GLOBAL_RNG,
  ) = RandomForestRegressor(nTrees,maxDepth,minGain,minRecords,maxFeatures,splittingCriterion,β,rng)

mutable struct RandomForestClassifier <: MMI.Probabilistic
    nTrees::Int64
    maxDepth::Int64
    minGain::Float64
    minRecords::Int64
    maxFeatures::Int64
    splittingCriterion::Function
    β::Float64
    rng::AbstractRNG
end
RandomForestClassifier(;
    nTrees=30,
    maxDepth=0,
    minGain=0.0,
    minRecords=2,
    maxFeatures=0,
    splittingCriterion=gini,
    β=0.0,
    rng = Random.GLOBAL_RNG,
) = RandomForestClassifier(nTrees,maxDepth,minGain,minRecords,maxFeatures,splittingCriterion,β,rng)

#=
# skipped for now..
# ------------------------------------------------------------------------------
# Hyperparameters ranges definition (for automatic tuning)

MMI.hyperparameter_ranges(::Type{<:DecisionTreeRegressor}) = (
#    (range(Float64, :alpha, lower=0, upper=1, scale=:log),
#     range(Int, :beta, lower=1, upper=Inf, origin=100, unit=50, scale=:log),
#         nothing)
    range(Int64,:maxDepth,lower=0,upper=Inf,scale=:log),
    range(Float64,:minGain,lower=0,upper=Inf,scale=:log),
    range(Int64,:minRecords,lower=0,upper=Inf,scale=:log),
    range(Int64,:maxFeatures,lower=0,upper=Inf,scale=:log),
    nothing
)
=#

# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(model::Union{DecisionTreeRegressor,RandomForestRegressor}, verbosity, X, y)
   x = MMI.matrix(X)                     # convert table to matrix
   maxDepth         = model.maxDepth == 0 ? size(x,1) : model.maxDepth
   # Using low level API here. We could switch to APIV2...
   if (typeof(model) == DecisionTreeRegressor)
       maxFeatures = model.maxFeatures == 0 ? size(x,2) : model.maxFeatures
       fitresult   = buildTree(x, y, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion,rng=model.rng)
   else
       maxFeatures = model.maxFeatures == 0 ? Int(round(sqrt(size(x,2)))) : model.maxFeatures
       fitresult   = buildForest(x, y, model.nTrees, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion, β=model.β,rng=model.rng)
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
   maxDepth         = model.maxDepth == 0 ? size(x,1) : model.maxDepth
   # Using low level API here. We could switch to APIV2...
   if (typeof(model) == DecisionTreeClassifier)
       maxFeatures   = model.maxFeatures == 0 ? size(x,2) : model.maxFeatures
       fittedmodel   = buildTree(x, yarray, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion, forceClassification=true,rng=model.rng)
   else
       maxFeatures   = model.maxFeatures == 0 ? Int(round(sqrt(size(x,2)))) : model.maxFeatures
       fittedmodel   = buildForest(x, yarray, model.nTrees, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion, forceClassification=true, β=model.β,rng=model.rng)
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
    #println(typeof(classes))
    nLevels          = length(classes)
    nRecords         = MMI.nrows(Xnew)
    treePredictions  = Trees.predict(fittedModel, MMI.matrix(Xnew),rng=model.rng)
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(classes)
            predMatrix[n,c] = get(treePredictions[n],cl,0.0)
        end
    end
    #predictions = [MMI.UnivariateFinite(classes, predMatrix[i,:])
    #               for i in 1:nRecords]
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
