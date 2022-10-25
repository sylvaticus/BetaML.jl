"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
    Utils module

Provide shared utility functions for various machine learning algorithms.

For the complete list of functions provided see below. The main ones are:

## Helper functions for logging
- Most BetAML functions accept a parameter `verbosity` that expect one of the element in the `Verbosity` enoum (`NONE`, `LOW`, `STD`, `HIGH` or `FULL`)
- Writing complex code and need to find where something is executed ? Use the macro [`@codelocation`](@ref)

## Stochasticity management
- Utils provide [`FIXEDSEED`], [`FIXEDRNG`] and [`generate_parallel_rngs`](@ref). All stochastic functions accept a `rng` paraemter. See the "Getting started" section in the tutorial for details.

## Data processing
- Various small and large utilities for helping processing the data, expecially before running a ML algorithm
- Includes [`getpermutations`](@ref), [`onehotencoder`](@ref), [`integerencoder`](@ref) (and [`integerdecoder`](@ref)), [`partition`](@ref), [`scale`](@ref) (and [`get_scalefactors`](@ref)), [`pca`](@ref), [`cross_validation`](@ref).
- Auto-tuning of hyperparameters is implemented in the supported models by specifying `autotune=true` and optionally overriding the `tunemethod` parameters (e.g. for different hyperparameters ranges or different resources available for the tuning). Autotuning is then implemented in the (first) `fit!` call. Provided autotuning methods:  [`SuccessiveHalvingSearch`](@ref) (default), [`GridSearch`](@ref)

## Samplers
- Utilities to sample from data (e.g. for neural network training or for cross-validation)
- Include the "generic" type [`SamplerWithData`](@ref), together with the sampler implementation [`KFold`](@ref) and the function [`batch`](@ref)

## Transformers
- Funtions that "transform" a single input (that can be also a vector or a matrix)
- Includes varios NN "activation" functions ([`relu`](@ref), [`celu`](@ref), [`sigmoid`](@ref), [`softmax`](@ref), [`pool1d`](@ref)) and their derivatives (`d[FunctionName]`), but also [`gini`](@ref), [`entropy`](@ref), [`variance`](@ref), [`BIC`](@ref bic), [`AIC`](@ref aic)

## Measures
- Several functions of a pair of parameters (often `y` and `ŷ`) to measure the goodness of `ŷ`, the distance between the two elements of the pair, ...
- Includes "classical" distance functions ([`l1_distance`](@ref), [`l2_distance`](@ref), [`l2squared_distance`](@ref) [`cosine_distance`](@ref)), "cost" functions for continuous variables ([`squared_cost`](@ref), [`mean_relative_error`](@ref)) and comparision functions for multui-class variables ([`crossentropy`](@ref), [`accuracy`](@ref), [`ConfMatrix`](@ref)).

"""
module Utils

using LinearAlgebra, Printf, Random, Statistics, Combinatorics, Zygote, CategoricalArrays, Random, DocStringExtensions

using ForceImport
@force using ..Api
using ..Api

export @codelocation, generate_parallel_rngs,
       reshape, makecolvector, makerowvector, makematrix, issortable, getpermutations,
#       onehotencoder, onehotdecoder, integerencoder, integerdecoder, # TODO: delete
       cols_with_missing, 
#      get_scalefactors, scale, scale!, # TODO: delete
       batch, partition, shuffle,
       pca,
       didentity, relu, drelu, elu, delu, celu, dcelu, plu, dplu,  #identity and rectify units
       dtanh, sigmoid, dsigmoid, softmax, dsoftmax, pool1d, softplus, dsoftplus, mish, dmish, # exp/trig based functions
       bic, aic,
       autojacobian,
       squared_cost, dsquared_cost, mse, crossentropy, dcrossentropy, class_counts, class_counts_with_labels, mean_dicts, mode, gini, entropy, variance,
       error, accuracy, relative_mean_error,
       ConfusionMatrix, ConfusionMatrixHyperParametersSet,
#       ConfMatrix, # TODO delete
       labels, scores, normalised_scores,
       cross_validation,
       AbstractDataSampler, SamplerWithData, KFold,
       autotune!, GridSearch, SuccessiveHalvingSearch, l2loss_by_cv,
       l1_distance,l2_distance, l2squared_distance, cosine_distance, lse, sterling,
       radial_kernel, polynomial_kernel,
       Scaler, MinMaxScaler, StandardScaler,
       ScalerHyperParametersSet, MinMaxScaler,StandardScaler,
       PCA, PCAHyperParametersSet,
       OneHotEncoder, OrdinalEncoder, OneHotEncoderHyperParametersSet,
       @threadsif,
       get_parametric_types, isinteger_bml

# Various functions that we add a method to
import Base.print, Base.println, Base.findfirst, Base.findall, Base.error, Random.shuffle, Base.show


include("Miscellaneous.jl")
include("Logging_utils.jl")
include("Processing.jl")
include("Stochasticity.jl")
include("Samplers.jl")
include("Transformers.jl")
include("Measures.jl")


end # end module
