# [Api v2 - developer documentation (API implementation)](@id api_implementation)

Each model is a child of either `BetaMLSuperVisedModel` or `BetaMLSuperVisedModel`, both in turn child of `BetaMLModel`:

```
BetaMLSuperVisedModel   <: BetaMLModel
BetaMLUnsupervisedModel <: BetaMLModel
RandomForestEstimator                 <: BetaMLSuperVisedModel
```

The model struct is composed of the following elements:

```
mutable struct DecisionTreeEstimator <: BetaMLSupervisedModel
    hpar::DTHyperParametersSet   # Hyper-pharameters
    opt::BetaMLDefaultOptionsSet # Option sets, default or a specific one for the model
    par::DTLearnableParameters   # Model learnable parameters (needed for predictions)
    cres::T                      # Cached results
    trained::Bool                # Trained flag
    info                         # Complementary information, but not needed to make predictions
end
```

Each specific model hyperparameter set and learnable parameter set are childs of `BetaMLHyperParametersSet` and `BetaMLLearnedParametersSet` and, if a specific model option set is used, this would be child of `BetaMLOptionsSet`.

While hyperparameters are elements that control the learning process, i.e. would influence the model training and prediction, the options have a more general meaning and do not directly affect the training (they can do indirectly, like the rng). The default option set is implemented as:

```
Base.@kwdef mutable struct BetaMLDefaultOptionsSet
   "Cache the results of the fitting stage, as to allow predict(mod) [default: `true`]. Set it to `false` to save memory for large data."
   cache::Bool = true
   "An optional title and/or description for this model"
   descr::String = "" 
   "The verbosity level to be used in training or prediction (see [`Verbosity`](@ref)) [deafult: `STD`]
   "
   verbosity::Verbosity = STD
   "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
   "
   rng::AbstractRNG = Random.GLOBAL_RNG
end
```

Note that the user doesn't generally need to make a difference between an hyperparameter and an option, as both are provided as keyword arguments to the model constructor thanks to a model constructor like the following one:

```
function KMedoidsClusterer(;kwargs...)
    m = KMedoidsClusterer(KMeansMedoidsHyperParametersSet(),BetaMLDefaultOptionsSet(),KMeansMedoidsLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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
```

So, in order to implement a new model we need to:
- implement its struct and constructor
- implement the relative `ModelHyperParametersSet`, `ModelLearnedParametersSet` and eventually `ModelOptionsSet`.
- define `fit!(model, X, [y])`, `predict(model,X)` and eventually `inverse_predict(model,X)`.

