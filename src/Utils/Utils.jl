"""
  Utils.jl File

Machine Learning shared utility functions (Module BetaML.Utils)

`?BetaML.Utils` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Utils.jl) - [Julia Package](https://github.com/sylvaticus/Utils.jl)
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    Utils module

Provide shared utility functions for various machine learning algorithms. You don't usually need to import from this module, as each other module (Nn, Perceptron, Clusters,...) reexport it.

"""
module Utils

using LinearAlgebra, Printf, Random, Statistics, Combinatorics, Zygote, CategoricalArrays, StableRNGs

using ForceImport
@force using ..Api

export Verbosity, NONE, LOW, STD, HIGH, FULL,
       FIXEDSEED, FIXEDRNG, @codeLocation, generateParallelRngs,
       reshape, makeColVector, makeRowVector, makeMatrix, issortable, getPermutations,
       oneHotEncoder, integerEncoder, integerDecoder, colsWithMissing, getScaleFactors, scale, scale!, batch, partition, shuffle, pca,
       didentity, relu, drelu, elu, delu, celu, dcelu, plu, dplu,  #identity and rectify units
       dtanh, sigmoid, dsigmoid, softmax, dsoftmax, pool1d, softplus, dsoftplus, mish, dmish, # exp/trig based functions
       bic, aic,
       autoJacobian,
       squaredCost, dSquaredCost, crossEntropy, dCrossEntropy, classCounts, classCountsWithLabels, meanDicts, mode, gini, entropy, variance,
       error, accuracy, meanRelError, ConfusionMatrix,
       crossValidation, AbstractDataSampler, SamplerWithData, KFold,
       l1_distance,l2_distance, l2²_distance, cosine_distance, lse, sterling,
       #normalFixedSd, logNormalFixedSd,
       radialKernel, polynomialKernel

# Various functions that we add a method to
import Base.print, Base.println, Base.findfirst, Base.findall, Base.error, Random.shuffle


include("Miscelanneous.jl")
include("Logging_utils.jl")
include("Processing.jl")
include("Stochasticity.jl")
include("Samplers.jl")
include("Transformers.jl")
include("Measures.jl")


end # end module
