"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
    Api

The Api Module (currently v2)


This module includes the shared api trough the various BetaML submodules, i.e. names used by more than one submodule.

Modules are free to use other functions but these are defined here to avoid name conflicts and allows instead Multiple Dispatch to handle them.
For a user-prospective overall description of the BetaML API see the page `API V2` → [`Introduction for users`](@ref api_usage), while for the implementation of the API see the page `API V2` → [`For developers`](@ref api_implementation)


"""
module Api

using StableRNGs, DocStringExtensions, Random

import Base.show


export Verbosity, NONE, LOW, STD, HIGH, FULL,     
       FIXEDSEED, FIXEDRNG,
       BetaMLModel, BetaMLSupervisedModel, BetaMLUnsupervisedModel,
       BetaMLOptionsSet, BetaMLDefaultOptionsSet, BetaMLHyperParametersSet, BetaMLLearnableParametersSet,
       predict, inverse_predict, fit!, partition, info, reset!, learned



abstract type BetaMLModel end
abstract type BetaMLSupervisedModel <: BetaMLModel end
abstract type BetaMLUnsupervisedModel <: BetaMLModel end
abstract type BetaMLOptionsSet end
abstract type BetaMLHyperParametersSet end
abstract type BetaMLLearnableParametersSet end


"""

$(TYPEDEF)

Many models and functions accept a `verbosity` parameter.

Choose between: `NONE`, `LOW`, `STD` [default], `HIGH` and `FULL`.
"""
@enum Verbosity NONE=0 LOW=10 STD=20 HIGH=30 FULL=40

"""
    const FIXEDSEED

Fixed seed to allow reproducible results.
This is the seed used to obtain the same results under unit tests.

Use it with:
- `myAlgorithm(;rng=MyChoosenRNG(FIXEDSEED))`             # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(MyChoosenRNG(FIXEDSEED)))`        # always produce the same result (new rng object on each call)
"""
const FIXEDSEED = 123

"""
$(TYPEDEF)

Fixed ring to allow reproducible results

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`         # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG))`   # always produce the same result (new rng object on each function call)
"""
const FIXEDRNG  = StableRNG(FIXEDSEED)

"""

$(TYPEDEF)

A struct defining the options used by default by the algorithms that do not override it with their own option sets.

# Fields:
$(TYPEDFIELDS)

# Notes:
- even if a model doesn't override `BetaMLDefaultOptionsSet`, may not use all its options, for example deterministic models would not make use of the `rng` parameter. Passing such parameters in these cases would simply have no influence.

# Example:
```
julia> options = BetaMLDefaultOptionsSet(cache=false,descr="My model")
```
"""
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

"""

    fit!(m::BetaMLModel,X,[y])

Fit ("train") a `BetaMLModel` (i.e. learn the algorithm's parameters) based on data, either only features or features and labels.

Each specific model implements its own version of `fit!(m,X,[Y])`, but the usage is consistent across models.

# Notes:
- For online algorithms, i.e. models that support updating of the learned parameters with new data, `fit!` can be repeated as new data arrive, altought not all algorithms guarantee that training each record at the time is equivalent to train all the records at once.
- If the model has been trained while having the `cache` option set on `true` (by default) `fit!` returns `ŷ` instead of `nothing` effectively making it behave like a _fit-and-transform_ function. 

""" 
fit!(::BetaMLModel,X)  = nothing

"""
    predict(m::BetaMLModel,[X])

Predict new information (including transformation) based on a fitted `BetaMLModel`, eventually applied to new features when the algorithm generalises to new data.

# Notes:
- As a convenience, if the model has been trained while having the `cache` option set on `true` (by default) the predictions associated with the last training of the model is retained in the  model object and can be retrieved simply with `predict(m)`.
""" 
function predict(m::BetaMLModel)
   if m.fitted 
      return m.cres
   else
      if m.opt.verbosity > NONE
         @warn "Trying to predict an unfitted model. Run `fit!(model,X,[Y])` before!"
      end
      return nothing
   end
end

"""
    inverse_predict(m::BetaMLModel,X)

Given a model `m` that fitted on `x` produces `xnew`, it takes `xnew` to return (possibly an approximation of ) `x`.

For example, when `OneHotEncoder` is fitted with a subset of the possible categories and the ` handle_unknown` option is set on `infrequent`, `inverse_transform` will aggregate all the _other_ categories as specified in `other_categories_name`.

# Notes:
- Inplemented only in a few models.
""" 
inverse_predict(m::BetaMLModel,X) = nothing

function info(m::BetaMLModel)
   return m.info
end

"""
    parameters(m::BetaMLModel)

Returns the learned parameters of a model.
""" 
function parameters(m::BetaMLModel)
   return m.par
end

"""
    reset!(m::BetaMLModel)

Reset the parameters of a trained model.
""" 
function reset!(m::BetaMLModel)
   m.par     = nothing
   m.cres    = nothing 
   m.info    = Dict{Symbol,Any}()
   m.fitted  = false 
   return nothing
end

function show(io::IO, ::MIME"text/plain", m::BetaMLModel)
   if m.fitted == false
       print(io,"A $(typeof(m)) BetaMLModel (unfitted)")
   else
       print(io,"A $(typeof(m)) BetaMLModel (fitted)")
   end
end

function show(io::IO, m::BetaMLModel)
   m.opt.descr != "" && println(io,m.opt.descr)
   if m.fitted == false
      print(io,"A $(typeof(m)) BetaMLModel (unfitted)")
   else
      println(io,"A $(typeof(m)) BetaMLModel (unfitted)")
      print(io,m.info)
   end
end

partition()            = nothing

end

