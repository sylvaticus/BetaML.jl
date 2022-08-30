"""
Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT.
"""

"""
    Clustering module (WIP)

(Hard) Clustering algorithms

Provide hard clustering methods using K-means and k-medoids. Please see also the [`GMM`](@ref) module for GMM-based soft clustering (i.e. where a probability distribution to be part of the various classes is assigned to each record instead of a single class), missing values imputation / collaborative filtering / reccomendation systems using clustering methods as backend.

The module provides the following models. Use `?[model]` to access their documentation:

- [`KMeansModel`](@ref): Classical KMean algorithm
- [`KMedoidsModel`](@ref kmeans): Kmedoids algorithm with configurable distance metric

"""
module Clustering

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays, DocStringExtensions
import Distributions

using  ForceImport
@force using ..Api
@force using ..Utils

import Base.print
import Base.show

export kmeans, kmedoids
export KMeansMedoidsHyperParametersSet, KMeansModel, KMedoidsModel 

include("Clustering_hard.jl") # K-means and k-medoids
# MLJ interface
include("Clustering_MLJ.jl")

end

