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
end
DecisionTreeRegressor(;
   maxDepth=0, #typemax(Int)
   minGain=0.0,
   minRecords=2,
   maxFeatures=0,
   splittingCriterion=variance
   ) = DecisionTreeRegressor(maxDepth,minGain,minRecords,maxFeatures,splittingCriterion)

mutable struct DecisionTreeClassifier <: MMI.Probabilistic
   maxDepth::Int64
   minGain::Float64
   minRecords::Int64
   maxFeatures::Int64
   splittingCriterion::Function
end
DecisionTreeClassifier(;
  maxDepth=0,
  minGain=0.0,
  minRecords=2,
  maxFeatures=0,
  splittingCriterion=gini
  ) = DecisionTreeClassifier(maxDepth,minGain,minRecords,maxFeatures,splittingCriterion)

mutable struct RandomForestRegressor <: MMI.Deterministic
   nTrees::Int64
   maxDepth::Int64
   minGain::Float64
   minRecords::Int64
   maxFeatures::Int64
   splittingCriterion::Function
   β::Float64
end
RandomForestRegressor(;
  nTrees=30,
  maxDepth=0,
  minGain=0.0,
  minRecords=2,
  maxFeatures=0,
  splittingCriterion=variance,
  β=0.0
  ) = RandomForestRegressor(nTrees,maxDepth,minGain,minRecords,maxFeatures,splittingCriterion,β)

mutable struct RandomForestClassifier <: MMI.Probabilistic
    nTrees::Int64
    maxDepth::Int64
    minGain::Float64
    minRecords::Int64
    maxFeatures::Int64
    splittingCriterion::Function
    β::Float64
end
RandomForestClassifier(;
    nTrees=30,
    maxDepth=0,
    minGain=0.0,
    minRecords=2,
    maxFeatures=0,
    splittingCriterion=gini,
    β=0.0
) = RandomForestClassifier(nTrees,maxDepth,minGain,minRecords,maxFeatures,splittingCriterion,β)

# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(model::Union{DecisionTreeRegressor,RandomForestRegressor}, verbosity, X, y)
   x = MMI.matrix(X)                     # convert table to matrix
   maxDepth         = model.maxDepth == 0 ? size(x,1) : model.maxDepth
   if (typeof(model) == DecisionTreeRegressor)
       maxFeatures = model.maxFeatures == 0 ? size(x,2) : model.maxFeatures
       fitresult   = buildTree(x, y, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion)
   else
       maxFeatures = model.maxFeatures == 0 ? Int(round(sqrt(size(x,2)))) : model.maxFeatures
       fitresult   = buildForest(x, y, model.nTrees, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion, β=model.β)
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
   if (typeof(model) == DecisionTreeClassifier)
       maxFeatures   = model.maxFeatures == 0 ? size(x,2) : model.maxFeatures
       fittedmodel   = buildTree(x, yarray, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion, forceClassification=true)
   else
       maxFeatures   = model.maxFeatures == 0 ? Int(round(sqrt(size(x,2)))) : model.maxFeatures
       fittedmodel   = buildForest(x, yarray, model.nTrees, maxDepth=maxDepth, minGain=model.minGain, minRecords=model.minRecords, maxFeatures=maxFeatures, splittingCriterion=model.splittingCriterion, forceClassification=true, β=model.β)
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
    treePredictions  = Trees.predict(fittedModel, MMI.matrix(Xnew))
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(classes)
            predMatrix[n,c] = get(treePredictions[n],cl,0.0)
        end
    end
    predictions = [MMI.UnivariateFinite(classes, predMatrix[i,:])
                   for i in 1:nRecords]
    return predictions
end

# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(DecisionTreeRegressor,
    input_scitype    = MMI.Table(MMI.Missing, MMI.Known),           # also ok: MMI.Table(Union{MMI.Missing, MMI.Known}),
    target_scitype   = AbstractVector{<: MMI.Continuous},           # for a supervised model, what target?
    supports_weights = false,                                       # does the model support sample weights?
    descr            = "A simple Decision Tree for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.DecisionTreeRegressor"
    )
MMI.metadata_model(RandomForestRegressor,
    input_scitype    = MMI.Table(MMI.Missing, MMI.Known),
    target_scitype   = AbstractVector{<: MMI.Continuous},
    supports_weights = false,
    descr            = "A simple Random Forest ensemble for regression with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.RandomForestRegressor"
    )
MMI.metadata_model(DecisionTreeClassifier,
    input_scitype    = MMI.Table(MMI.Missing, MMI.Known),
    target_scitype   = AbstractVector{<: Union{MMI.Missing,MMI.Finite,MMI.Count}},
    supports_weights = false,
    descr            = "A simple Decision Tree for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.DecisionTreeClassifier"
    )
MMI.metadata_model(RandomForestClassifier,
    input_scitype    = MMI.Table(MMI.Missing, MMI.Known),                     
    target_scitype   = AbstractVector{<: Union{MMI.Missing,MMI.Finite,MMI.Count}},
    supports_weights = false,
    descr            = "A simple Random Forest ensemble for classification with support for Missing data, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Trees.RandomForestClassifier"
    )
