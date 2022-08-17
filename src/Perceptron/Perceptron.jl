"""
    Perceptron.jl file

Implement the BetaML.Perceptron module

`?BetaML.Perceptron` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Perceptron.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/lmlj.jl/blob/master/notebooks/Perceptron.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FPerceptron.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""

"""
    Perceptron module

Provide linear and kernel classifiers.

See a [runnable example on myBinder](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FPerceptron.ipynb)

- [`perceptron`](@ref): Train data using the classical perceptron
- [`kernelPerceptron`](@ref): Train data using the kernel perceptron
- [`pegasos`](@ref): Train data using the pegasos algorithm
- [`predict`](@ref): Predict data using parameters from one of the above algorithms

All algorithms are multiclass, with `perceptron` and `pegasos` employing a one-vs-all strategy, while `kernelPerceptron` employs a _one-vs-one_ approach, and return a "probability" for each class in term of a dictionary for each record. Use `mode(ŷ)` to return a single class prediction per record.

The binary equivalent algorithms, accepting only `{-1,+1}` labels, are available as `peceptronBinary`, `kernelPerceptronBinary` and `pegasosBinary`. They are slighly faster as they don't need to be wrapped in the multi-class equivalent and return a more informative output.

The multi-class versions are available in the MLJ framework as `PerceptronClassifier`,`KernelPerceptronClassifier` and `PegasosClassifier` respectivly.
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
