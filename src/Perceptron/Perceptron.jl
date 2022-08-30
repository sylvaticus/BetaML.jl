"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
    Perceptron module

Provide linear and kernel classifiers.

Provide the following supervised models:

- [`PerceptronClassic`](@ref): Train data using the classical perceptron
- [`KernelPerceptron`](@ref): Train data using the kernel perceptron
- [`Pegasos`](@ref): Train data using the pegasos algorithm


All algorithms are multiclass, with `PerceptronClassic` and `Pegasos` employing a one-vs-all strategy, while `KernelPerceptron` employs a _one-vs-one_ approach, and return a "probability" for each class in term of a dictionary for each record. Use `mode(ŷ)` to return a single class prediction per record.

These models are available in the MLJ framework as `PerceptronClassifier`,`KernelPerceptronClassifier` and `PegasosClassifier` respectivly.
"""
module Perceptron

using LinearAlgebra, Random, ProgressMeter, Reexport, CategoricalArrays, DocStringExtensions

using ForceImport
@force using ..Api
@force using ..Utils

import Base.show

export perceptron, perceptronBinary, kernelPerceptron, kernelPerceptronBinary, pegasos, pegasosBinary, predict
export PerceptronClassic, KernelPerceptron, Pegasos
export PerceptronClassicHyperParametersSet, KernelPerceptronHyperParametersSet, PegasosHyperParametersSet


include("Perceptron_classic.jl")
include("Perceptron_kernel.jl")
include("Perceptron_pegasos.jl")


# MLJ interface
include("Perceptron_MLJ.jl")

end
