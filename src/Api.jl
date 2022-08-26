
module Api

using StableRNGs, DocStringExtensions, Random

import Base.show

# Shared api trough the modules, i.e. names used by more than one module
# Modules are free to use other functions but these are defined here to avoid name conflicts
# and allows instead Multiple Dispatch to handle them

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

@enum Verbosity NONE=0 LOW=10 STD=20 HIGH=30 FULL=40

"""
    FIXEDSEED

Fixed seed to allow reproducible results.
This is the seed used to obtain the same results under unit tests.

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`             # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG)`        # always produce the same result (new rng object on each call)
"""
const FIXEDSEED = 123

"""
    FIXEDRNG

Fixed ring to allow reproducible results

Use it with:
- `myAlgorithm(;rng=FIXEDRNG)`         # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
- `myAlgorithm(;rng=copy(FIXEDRNG))`   # always produce the same result (new rng object on each function call)

"""
const FIXEDRNG  = StableRNG(FIXEDSEED)

"""
   BetaMLDefaultOptionsSet

A struct defining the options used by default by the algorithms that do not override it with their own option sets

## Parameters:
$(FIELDS)
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

Fit ("train") a BetaMLModel (i.e. learn the algorithm's parameters) based on data, either only features or features and labels
""" 
fit!(::BetaMLModel)  = nothing
"""
   predict(m::BetaMLModel,[X])

Predict new information (including transformation) based on a fitted BetaMLModel, eventually applied to new features when the algorithms generalise to new data.
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

Inplemented only in a few models.
""" 
inverse_predict(m::BetaMLModel,X) = nothing

function info(m::BetaMLModel)
   return m.info
end
"""
   learned(m::BetaMLModel)

Returns the learned parameters of a model.
""" 
function learned(m::BetaMLModel)
   return m.par
end

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

