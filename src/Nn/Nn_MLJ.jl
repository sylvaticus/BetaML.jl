"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Neural Networks models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here
using CategoricalArrays

export NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor, NeuralNetworkClassifier


# Model Structure declarations..
"""
$(TYPEDEF)

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for regression of a single dimensional target.

## Parameters:
$(FIELDS)

## Notes:
- data must be numerical
- the label should be be a _n-records_ vector.
"""
Base.@kwdef mutable struct NeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squared_cost`]. Should always assume y and ŷ as matrices, even if the regression task is 1-D
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = squared_cost
    "Derivative of the loss function [def: `dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`"
    cb::Function=fitting_info
    "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
    "
    rng::AbstractRNG = Random.GLOBAL_RNG
end

"""
$(TYPEDSIGNATURES)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::NeuralNetworkRegressor, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    if !(verbosity == 0 || verbosity == 10 || verbosity == 20 || verbosity == 30 || verbosity == 40) 
        error("Wrong verbosity level. Verbosity must be either 0, 10, 20, 30 or 40.")
    end
    ndims(y) > 1 && error("The label should have only 1 dimensions. Use `MultitargetNeuralNetworkRegressor` or `NeuralNetworkClassifier` for multi_dimensional outputs.")
    mi = NeuralNetworkEstimator(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=Verbosity(verbosity))
    fit!(mi,x,y)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 MMI.predict(m::NeuralNetworkRegressor, fitresult, Xnew) = predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(NeuralNetworkRegressor,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype   = AbstractVector{<: Union{MMI.Continuous,MMI.Count}},
    supports_weights = false,
    load_path        = "BetaML.Nn.NeuralNetworkRegressor"
)

# ------------------------------------------------------------------------------
# Model Structure declarations..
"""
$(TYPEDEF)

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for regression of multiple dimensional targets.

## Parameters:
$(FIELDS)

## Notes:
- data must be numerical
- the label should be a _n-records_ by _n-dimensions_ matrix 
"""
Base.@kwdef mutable struct MultitargetNeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squared_cost`].  Should always assume y and ŷ as matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = squared_cost
    "Derivative of the loss function [def: `dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`"
    cb::Function=fitting_info
    "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
    "
    rng::AbstractRNG = Random.GLOBAL_RNG
end
"""
$(TYPEDSIGNATURES)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::MultitargetNeuralNetworkRegressor, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    if !(verbosity == 0 || verbosity == 10 || verbosity == 20 || verbosity == 30 || verbosity == 40) 
        error("Wrong verbosity level. Verbosity must be either 0, 10, 20, 30 or 40.")
    end
    ndims(y) > 1 || error("The label should have multiple dimensions. Use `NeuralNetworkRegressor` for single-dimensional outputs.")
    mi = NeuralNetworkEstimator(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=Verbosity(verbosity))
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
    load_path        = "BetaML.Nn.MultitargetNeuralNetworkRegressor"
)

# ------------------------------------------------------------------------------

"""
$(TYPEDEF)

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for classification  problems.

## Parameters:
$(FIELDS)

## Notes:
- data must be numerical
- the label should be a _n-records_ by _n-dimensions_ matrix (e.g. a one-hot-encoded data for classification), where the output columns should be interpreted as the probabilities for each categories.
"""
Base.@kwdef mutable struct NeuralNetworkClassifier <: MMI.Probabilistic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers. The last \"softmax\" layer is automatically added."
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `crossentropy`]. Should always assume y and ŷ as matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = crossentropy
    "Derivative of the loss function [def: `dcrossentropy`, i.e. the derivative of the cross-entropy]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dcrossentropy
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`"
    cb::Function=fitting_info
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
    mi = NeuralNetworkEstimator(;layers=layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=Verbosity(verbosity))
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
    load_path        = "BetaML.Nn.NeuralNetworkClassifier"
)
