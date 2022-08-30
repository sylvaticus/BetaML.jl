"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Neural Networks models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export FeedforwardNeuralNetwork



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
Base.@kwdef mutable struct FeedforwardNeuralNetwork <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    "Loss (cost) function [def: `squaredCost`]"
    loss::Union{Nothing,Function} = squaredCost
    "Derivative of the loss function [def: `nothing`, i.e. use autodiff]"
    dloss::Union{Function,Nothing}  = nothing
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
function MMI.fit(m::FeedforwardNeuralNetwork, verbosity, X, y)
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

 MMI.predict(m::FeedforwardNeuralNetwork, fitresult, Xnew) = predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(FeedforwardNeuralNetwork,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype   = Union{AbstractVector{<: Union{MMI.Continuous,MMI.Count}},AbstractMatrix{<: Union{MMI.Continuous,MMI.Count}}},
    supports_weights = false,
    descr            = "A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) that can be used for both regression tasks or classification ones.",
    load_path        = "BetaML.Nn.FeedforwardNeuralNetwork"
)