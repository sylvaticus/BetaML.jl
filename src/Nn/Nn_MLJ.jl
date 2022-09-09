"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Neural Networks models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here
using CategoricalArrays

export MultitargetNeuralNetworkRegressor, NeuralNetworkClassifier



# ------------------------------------------------------------------------------
# Model Structure declarations..
"""
    FeedfordwardNeuralNetwork

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) that can be used for both regression tasks or classification ones.

## Parameters:
$(FIELDS)

## Notes:
- data must be numerical
- the label can be a _n-records_ vector or a _n-records_ by _n-dimensions_ matrix (e.g. a one-hot-encoded data for classification), but the result is always a matrix.
  - For one-dimension regressions drop the unnecessary dimension with `dropdims(ŷ,dims=2)`
  - For classification tasks the columns should be interpreted as the probabilities for each categories
"""
Base.@kwdef mutable struct MultitargetNeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squaredCost`].
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = squaredCost
    "Derivative of the loss function [def: `dSquaredCost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dSquaredCost
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batchSize::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    optAlg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `trainingInfo`"
    cb::Function=trainingInfo
    "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
    "
    rng::AbstractRNG = Random.GLOBAL_RNG
end
"""

MMI.fit(model::FeedforwardNN, verbosity, X, y)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::MultitargetNeuralNetworkRegressor, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    if !(verbosity == 0 || verbosity == 10 || verbosity == 20 || verbosity == 30 || verbosity == 40) 
        error("Wrong verbosity level. Verbosity must be either 0, 10, 20, 30 or 40.")
    end
    mi = FeedforwardNN(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batchSize=m.batchSize, optAlg=m.optAlg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=Verbosity(verbosity))
    fit!(mi,x,y)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 MMI.predict(m::MultitargetNeuralNetworkRegressor, fitresult, Xnew) = predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(MultitargetNeuralNetworkRegressor,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype   = Union{AbstractVector{<: Union{MMI.Continuous,MMI.Count}},AbstractMatrix{<: Union{MMI.Continuous,MMI.Count}}},
    supports_weights = false,
    descr            = "A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) that can be used for both regression tasks or classification ones.",
    load_path        = "BetaML.Nn.MultitargetNeuralNetworkRegressor"
)

# ------------------------------------------------------------------------------

"""
NeuralNetworkClassifier

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for classification  problems.

## Parameters:
$(FIELDS)

## Notes:
- data must be numerical
- the label can be a _n-records_ vector or a _n-records_ by _n-dimensions_ matrix (e.g. a one-hot-encoded data for classification), but the result is always a matrix.
  - For one-dimension regressions drop the unnecessary dimension with `dropdims(ŷ,dims=2)`
  - For classification tasks the columns should be interpreted as the probabilities for each categories
"""
Base.@kwdef mutable struct NeuralNetworkClassifier <: MMI.Probabilistic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers. The last \"softmax\" layer is automatically added."
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `crossEntropy`].
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = crossEntropy
    "Derivative of the loss function [def: `dCrossEntropy`, i.e. the derivative of the cross-entropy]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dCrossEntropy
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batchSize::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    optAlg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `trainingInfo`"
    cb::Function=trainingInfo
    "The categories to represent as columns. [def: `nothing`, i.e. unique training values]."  
    categories::Union{Vector,Nothing} = nothing
    "How to handle categories not seens in training or not present in the provided `categories` array? \"error\" (default) rises an error, \"infrequent\" adds a specific column for these categories."
    handle_unknown::String = "error"
    "Which value during prediction to assign to this \"other\" category (i.e. categories not seen on training or not present in the provided `categories` array? [def: ` nothing`, i.e. typemax(Int64) for integer vectors and \"other\" for other types]. This setting is active only if `handle_unknown=\"infrequent\"` and in that case it MUST be specified if Y is neither integer or strings"
    other_categories_name = nothing
    "Random Number Generator [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG = Random.GLOBAL_RNG
end
"""

MMI.fit(model::NeuralNetworkClassifier, verbosity, X, y)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::NeuralNetworkClassifier, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    if !(verbosity == 0 || verbosity == 10 || verbosity == 20 || verbosity == 30 || verbosity == 40) 
        error("Wrong verbosity level. Verbosity must be either 0, 10, 20, 30 or 40.")
    end

    categories = deepcopy(m.categories)
    if categories == nothing
        #if occursin("CategoricalVector",string(typeof(y))) # to avoid dependency to CategoricalArrays or MLJBase 
        if typeof(y) <: CategoricalVector
            categories = levels(y)
        end
    end

    ohmod = OneHotEncoder(categories=categories,handle_unknown=m.handle_unknown,other_categories_name=m.other_categories_name)
    Y_oh = fit!(ohmod,y)

    nR,nD       = size(x)
    (nRy,nDy)   = size(Y_oh)         
    
    nR == nRy || error("X and Y have different number of records (rows)")

    if isnothing(m.layers)
        layers = nothing
    else
        layers = deepcopy(m.layers)
        push!(layers,VectorFunctionLayer(nDy,f=softmax))
    end
    mi = FeedforwardNN(;layers=layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batchSize=m.batchSize, optAlg=m.optAlg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=Verbosity(verbosity))
    fit!(mi,x,Y_oh)
    fitresults = (mi,ohmod)
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

function MMI.predict(m::NeuralNetworkClassifier, fitresult, Xnew) 
    nnmod, ohmod = fitresult
    yhat = predict(nnmod, MMI.matrix(Xnew))
    classes = parameters(ohmod).categories_applied
    predictions = MMI.UnivariateFinite(classes, yhat,pool=missing)
    #return yhat
    return predictions
end

 MMI.metadata_model(NeuralNetworkClassifier,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype = AbstractVector{<: Union{MMI.Multiclass,MMI.Finite,MMI.Count}},
    supports_weights = false,
    descr            = "A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) that can be used for both regression tasks or classification ones.",
    load_path        = "BetaML.Nn.NeuralNetworkClassifier"
)
