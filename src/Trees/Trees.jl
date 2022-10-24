"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
    BetaML.Trees module

Implement the [`DecisionTreeEstimator`](@ref) and [`RandomForestEstimator`](@ref) models (Decision Trees and Random Forests).

Both Decision Trees and Random Forests can be used for regression or classification problems, based on the type of the labels (numerical or not). The automatic selection can be overridden with the parameter `force_classification=true`, typically if labels are integer representing some categories rather than numbers. For classification problems the output of `predict` is a dictionary with the key being the labels with non-zero probabilitity and the corresponding value its probability; for regression it is a numerical value.

Please be aware that, differently from most other implementations, the Random Forest algorithm collects and averages the probabilities from the trees, rather than just repording the mode, i.e. no information is lost and the output of the forest classifier is still a PMF.

To retrieve the prediction with the highest probability use [`mode`](@ref) over the prediciton returned by the model. Most error/accuracy measures in the [`Utils`](@ref) BetaML module works diretly with this format.

Missing data and trully unordered types are supported on the features, both on training and on prediction.

The module provide the following functions. Use `?[type or function]` to access their full signature and detailed documentation:

Features are expected to be in the standard format (nRecords × nDimensions matrices) and the labels (either categorical or numerical) as a nRecords column vector.

Acknowlegdments: originally based on the [Josh Gordon's code](https://www.youtube.com/watch?v=LDRbO9a6XPU)
"""
module Trees

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays, DocStringExtensions
using AbstractTrees

using  ForceImport
@force using ..Api
@force using ..Utils

import Base.print
import Base.show
import Base.convert

export DecisionTreeEstimator, DTHyperParametersSet
# export AbstractDecisionNode,Leaf, DecisionNode, 
# export buildTree
#predictSingle # TODO: to remove

export RandomForestEstimator, RFHyperParametersSet
#export  Forest 
# export buildForest
# updateTreesWeights! # TODO:to remove

include("DecisionTrees.jl") # Decision Trees algorithm and API
include("RandomForests.jl") # Random Forests algorithm and API
include("Trees_MLJ.jl")     # MLJ interface


end # end module
