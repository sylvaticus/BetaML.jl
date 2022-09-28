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
import JLD2


export Verbosity, NONE, LOW, STD, HIGH, FULL,     
       FIXEDSEED, FIXEDRNG,
       BetaMLModel, BetaMLSupervisedModel, BetaMLUnsupervisedModel,
       BetaMLOptionsSet, BetaMLDefaultOptionsSet, BetaMLHyperParametersSet, BetaMLLearnableParametersSet,
       AutoTuneMethod,
       predict, inverse_predict, fit!, fit_ex, info, reset!, reset_ex, parameters,hyperparameters, options, sethp!,
       model_save, model_load



abstract type BetaMLModel end
abstract type BetaMLSupervisedModel <: BetaMLModel end
abstract type BetaMLUnsupervisedModel <: BetaMLModel end
abstract type BetaMLOptionsSet end
abstract type BetaMLHyperParametersSet end
abstract type BetaMLLearnableParametersSet end
abstract type AutoTuneMethod end


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
   "0ption for hyper-parameters autotuning [def: `false`, i.e. not autotuning performed]. If activated, autotuning is performed on the first `fit!()` call. Controll auto-tuning trough the option `tunemethod` (see the model hyper-parameters)"
   autotune::Bool = false
   "The verbosity level to be used in training or prediction (see [`?Verbosity`](@ref Verbosity)) [deafult: `STD`]
   "
   verbosity::Verbosity = STD
   "Random Number Generator (see [`?FIXEDSEED`](@ref FIXEDSEED)) [deafult: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG = Random.GLOBAL_RNG
end

"""

    fit!(m::BetaMLModel,X,[y])

Fit ("train") a `BetaMLModel` (i.e. learn the algorithm's parameters) based on data, either only features or features and labels.

Each specific model implements its own version of `fit!(m,X,[Y])`, but the usage is consistent across models.

# Notes:
- For online algorithms, i.e. models that support updating of the learned parameters with new data, `fit!` can be repeated as new data arrive, altought not all algorithms guarantee that training each record at the time is equivalent to train all the records at once.
- If the model has been trained while having the `cache` option set on `true` (by default) `fit!` returns `ŷ` instead of `nothing` effectively making it behave like a _fit-and-transform_ function.
- In Python and other languages that don't allow the exclamation mark within the function name, use `fit_ex(⋅)` instead of `fit!(⋅)`

""" 
fit!(::BetaMLModel,args...;kargs...)  = nothing

fit_ex(m::BetaMLModel,args...;kargs...) = fit!(m,args...;kargs...) # version for Python interface that doesn't like the exclamation mark



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

"""
$(TYPEDSIGNATURES)

Return a string-keyed dictionary of "additional" information stored during model fitting.

"""
function info(m::BetaMLModel)
   return m.info
end

"""
    reset!(m::BetaMLModel)

Reset the parameters of a trained model.

Notes:
- In Python and other languages that don't allow the exclamation mark within the function name, use `reset_ex(⋅)` instead of `reset!(⋅)`
""" 
function reset!(m::BetaMLModel)
   m.par     = nothing
   m.cres    = nothing 
   m.info    = Dict{Symbol,Any}()
   m.fitted  = false 
   return nothing
end
reset_ex(m::BetaMLModel,args...;kargs...) = reset!(m,args...;kargs...) # version for Python interface that doesn't like the exclamation mark


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
      println(io,"A $(typeof(m)) BetaMLModel (fitted)")
      println(io,"Output of `info(model)`:")
      for (k,v) in info(m)
          print(io,"- ")
          print(io,k)
          print(io,":\t")
          println(io,v)
      end
   end
end

#partition()            = nothing

"""
    parameters(m::BetaMLModel)

Returns the learned parameters of a BetaML model.

!!! warning
    The returned object is a reference, so if it is modified, the relative object in the model will change too.
""" 
parameters(m::BetaMLModel)      = m.par

"""
    hyperparameters(m::BetaMLModel)

Returns the hyperparameters of a BetaML model. See also [`?options`](@ref options) for the parameters that do not directly affect learning.

!!! warning
    The returned object is a reference, so if it is modified, the relative object in the model will change too.

""" 
hyperparameters(m::BetaMLModel) = m.hpar

"""
$(TYPEDSIGNATURES)

Set the hyperparameters of model `m` as specified in the `hp` dictionary.
""" 
function sethp!(m::BetaMLModel,hp::Dict)
    hpobj = hyperparameters(m)
    for (k,v) in hp
        setproperty!(hpobj,Symbol(k),v)
    end
end

"""
    options(m::BetaMLModel)

Returns the non-learning related options of a BetaML model. See also [`?hyperparameters`](@ref hyperparameters) for the parameters that directly affect learning.

!!! warning
    The returned object is a reference, so if it is modified, the relative object in the model will change too.
""" 
options(m::BetaMLModel)         = m.opt

#function model_save(filename::AbstractString;names...)
#    JLD2.jldsave(filename;names...)
#end
"""
    model_save(filename::AbstractString,overwrite_file::Bool=false;kwargs...)

Allow to save one or more BetaML models (wheter fitted or not), eventually specifying a name for each of them.

# Parameters:
- `filename`: Name of the destination file
- `overwrite_file`: Wheter to overrite the file if it alreaxy exist or preserve it (for the objects different than the one that are going to be saved) [def: `false`, i.e. preserve the file]
- `kwargs`: model objects to be saved, eventually associated with a different name to save the mwith (e.g. `mod1Name=mod1,mod2`)  
# Notes:
- If an object with the given name already exists on the destination JLD2 file it will be ovenwritten.
- If the file exists, but it is not a JLD2 file and the option `overwrite_file` is set to `false`, an error will be raisen.
- Use the semicolon `;` to separate the filename from the model(s) to save
- For further options see the documentation of the [`JLD2`](https://juliaio.github.io/JLD2.jl/stable/) package

# Examples
```
julia> model_save("fittedModels.jl"; mod1Name=mod1,mod2)
```
"""
function model_save(filename,overwrite_file::Bool=false;kargs...)
    flag = overwrite_file ? "w" : "a+"
    JLD2.jldopen(filename, flag) do f
        for (k,v) in kargs
            ks = string(k)
            if ks in keys(f)
                delete!(f, ks)
            end
            f[ks] = v
        end
    end
end

"""
    model_load(filename::AbstractString)
    model_load(filename::AbstractString,args::AbstractString...)
    
Load from file one or more BetaML models (wheter fitted or not).

# Notes:
- If no model names to retrieve are specified it returns a dictionary keyed with the model names
- If multiple models are demanded, a tuple is returned
- For further options see the documentation of the function `load` of the [`JLD2`](https://juliaio.github.io/JLD2.jl/stable/) package

# Examples:
```
julia> models = model_load("fittedModels.jl"; mod1Name=mod1,mod2)
julia> mod1 = model_load("fittedModels.jl",mod1)
julia> (mod1,mod2) = model_load("fittedModels.jl","mod1", "mod2")
```
"""
const model_load = JLD2.load


end