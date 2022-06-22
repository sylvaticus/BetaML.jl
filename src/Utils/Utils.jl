

"""
  Utils.jl File

Machine Learning shared utility functions (Module BetaML.Utils)

`?BetaML.Utils` for documentation

- Part of [BetaML](https://github.com/sylvaticus/BetaML.jl)
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    Utils module

Provide shared utility functions for various machine learning algorithms.

For the complete list of functions provided see below. The main ones are:

## Helper functions for logging
- Most BetAML functions accept a parameter `verbosity` that expect one of the element in the `Verbosity` enoum (`NONE`, `LOW`, `STD`, `HIGH` or `FULL`)
- Writing complex code and need to find where something is executed ? Use the macro [`@codeLocation`](@ref)

## Stochasticity management
- Utils provide [`FIXEDSEED`], [`FIXEDRNG`] and [`generateParallelRngs`](@ref). All stochastic functions accept a `rng` paraemter. See the "Getting started" section in the tutorial for details.

## Data processing
- Various small and large utilities for helping processing the data, expecially before running a ML algorithm
- Includes [`getPermutations`](@ref), [`oneHotEncoder`](@ref), [`integerEncoder`](@ref) (and [`integerDecoder`](@ref)), [`partition`](@ref), [`scale`](@ref) (and [`getScaleFactors`](@ref)), [`pca`](@ref), [`crossValidation`](@ref)

## Samplers
- Utilities to sample from data (e.g. for neural network training or for cross-validation)
- Include the "generic" type [`SamplerWithData`](@ref), together with the sampler implementation [`KFold`](@ref) and the function [`batch`](@ref)

## Transformers
- Funtions that "transform" a single input (that can be also a vector or a matrix)
- Includes varios NN "activation" functions ([`relu`](@ref), [`celu`](@ref), [`sigmoid`](@ref), [`softmax`](@ref), [`pool1d`](@ref)) and their derivatives (`d[FunctionName]`), but also [`gini`](@ref), [`entropy`](@ref), [`variance`](@ref), [`BIC`](@ref bic), [`AIC`](@ref aic)

## Measures
- Several functions of a pair of parameters (often `y` and `ŷ`) to measure the goodness of `ŷ`, the distance between the two elements of the pair, ...
- Includes "classical" distance functions ([`l1_distance`](@ref), [`l2_distance`](@ref), [`l2²_distance`](@ref) [`cosine_distance`](@ref)), "cost" functions for continuous variables ([`squaredCost`](@ref), [`meanRelError`](@ref)) and comparision functions for multui-class variables ([`crossEntropy`](@ref), [`accuracy`](@ref), [`ConfusionMatrix`](@ref)).

# Imputers
- Imputers of missing values

"""
module Utils

using LinearAlgebra, Printf, Random, Statistics, Combinatorics, Zygote, CategoricalArrays, StableRNGs

using ForceImport
@force using ..Api

export Verbosity, NONE, LOW, STD, HIGH, FULL,
       FIXEDSEED, FIXEDRNG, @codeLocation, generateParallelRngs,
       reshape, makeColVector, makeRowVector, makeMatrix, issortable, getPermutations,
       oneHotEncoder, oneHotDecoder, integerEncoder, integerDecoder, colsWithMissing, getScaleFactors, scale, scale!, batch, partition, shuffle, pca,
       didentity, relu, drelu, elu, delu, celu, dcelu, plu, dplu,  #identity and rectify units
       dtanh, sigmoid, dsigmoid, softmax, dsoftmax, pool1d, softplus, dsoftplus, mish, dmish, # exp/trig based functions
       bic, aic,
       autoJacobian,
       squaredCost, dSquaredCost, mse, crossEntropy, dCrossEntropy, classCounts, classCountsWithLabels, meanDicts, mode, gini, entropy, variance,
       error, accuracy, meanRelError, ConfusionMatrix,
       crossValidation, AbstractDataSampler, SamplerWithData, KFold,
       l1_distance,l2_distance, l2²_distance, cosine_distance, lse, sterling,
       #normalFixedSd, logNormalFixedSd,
       radialKernel, polynomialKernel

# Various functions that we add a method to
import Base.print, Base.println, Base.findfirst, Base.findall, Base.error, Random.shuffle


#include("Miscelanneous.jl")
include("Logging_utils.jl")
include("Processing.jl")
include("Stochasticity.jl")
include("Samplers.jl")
include("Transformers.jl")
include("Measures.jl")


end # end module
