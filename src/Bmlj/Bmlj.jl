"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for BetaML models

module Bmlj

using CategoricalArrays, DocStringExtensions
using Random

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here


#@force using ..Nn
#import ..Api
using  ..Api
import ..Utils # can't using it as it exports some same-name models
import ..Api: AutoTuneMethod, fit!
import ..Nn: AbstractLayer, ADAM, SGD, NeuralNetworkEstimator, OptimisationAlgorithm, DenseLayer, NN
import ..Utils: AbstractRNG, squared_cost, SuccessiveHalvingSearch, mljverbosity_to_betaml_verbosity


include("Nn_mlj.jl")
include("Utils_mlj.jl")

end