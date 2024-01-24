"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
    Perceptron module

Provide linear and kernel classifiers.

Provide the following supervised models:

- [`PerceptronClassifier`](@ref): Train data using the classical perceptron
- [`KernelPerceptronClassifier`](@ref): Train data using the kernel perceptron
- [`PegasosClassifier`](@ref): Train data using the pegasos algorithm


All algorithms are multiclass, with `PerceptronClassifier` and `PegasosClassifier` employing a one-vs-all strategy, while `KernelPerceptronClassifier` employs a _one-vs-one_ approach, and return a "probability" for each class in term of a dictionary for each record. Use `mode(ŷ)` to return a single class prediction per record.

These models are available in the MLJ framework as `PerceptronClassifier`,`KernelPerceptronClassifier` and `PegasosClassifier` respectivly.
"""
module Perceptron

using LinearAlgebra, Random, ProgressMeter, Reexport, CategoricalArrays, DocStringExtensions

using ForceImport
@force using ..Api
@force using ..Utils

import Base.show

# export perceptron, perceptronBinary, KernelPerceptronClassifier, KernelPerceptronClassifierBinary, pegasos, pegasosBinary, predict
export PerceptronClassifier, KernelPerceptronClassifier, PegasosClassifier
export PerceptronClassifierHyperParametersSet, KernelPerceptronClassifierHyperParametersSet, PegasosClassifierHyperParametersSet


include("Perceptron_classic.jl")
include("Perceptron_kernel.jl")
include("Perceptron_pegasos.jl")

end
