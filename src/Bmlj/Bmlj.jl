"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
# MLJ interface for BetaML models

In this module we define the interface of several BetaML models. They can be used using the [MLJ framework](https://github.com/alan-turing-institute/MLJ.jl).

Note that MLJ models (whose name could be the same as the underlying BetaML model) are not exported. You can access them with `BetaML.Bmlj.ModelXYZ`.

"""
module Bmlj

using Random, LinearAlgebra, Statistics
using CategoricalArrays, DocStringExtensions


import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here


#@force using ..Nn
#import ..Api
using  ..Api
import ..Api: AutoTuneMethod, fit!
import ..BetaML
import ..Utils # can't using it as it exports some same-name models
import ..Perceptron
import ..Nn: AbstractLayer, ADAM, SGD, NeuralNetworkEstimator, OptimisationAlgorithm, DenseLayer, NN
import ..Utils: AbstractRNG, squared_cost, SuccessiveHalvingSearch, radial_kernel
import ..GMM

export mljverbosity_to_betaml_verbosity

"""
$(TYPEDSIGNATURES)

Convert any integer (short scale) to one of the defined betaml verbosity levels
Currently "steps" are 0, 1, 2 and 3
"""
function mljverbosity_to_betaml_verbosity(i::Integer)
    if i <= 0
        return NONE
    elseif i == 1
        return LOW
    elseif i == 2
        return STD
    elseif i == 3
        return HIGH
    else
        return FULL
    end
end

include("Perceptron_mlj.jl") # Perceptron-like algorithms
include("Trees_mlj.jl")      # Decision Trees and ensembles (Random Forests)
include("Clustering_mlj.jl") # Clustering (hard) algorithms
include("GMM_mlj.jl")        # GMM-based learners (clustering, fitter, regression) 
include("Imputation_mlj.jl") # Imputation models
include("Nn_mlj.jl")         # Neural network models
include("Utils_mlj.jl")      # Various transformers/encorders

end