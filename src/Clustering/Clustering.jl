"""
Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT.
"""

"""
    Clustering module (WIP)

(Hard) Clustering algorithms

Provide hard clustering methods using K-means and k-medoids. Please see also the [`GMM`](@ref) module for GMM-mased soft clustering, missing values imputation / collaborative filtering / reccomendation systems using clustering methods as backend.

The module provides the following functions. Use `?[function]` to access their full signature and detailed documentation:

- [`initRepresentatives(X,K;initStrategy,Z₀)`](@ref initRepresentatives): Initialisation strategies for Kmean and Kmedoids
- [`kmeans(X,K;dist,initStrategy,Z₀)`](@ref kmeans): Classical KMean algorithm
- [`kmedoids(X,K;dist,initStrategy,Z₀)`](@ref kmedoids): Kmedoids algorithm
"""
module Clustering

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays, DocStringExtensions
import Distributions

using  ForceImport
@force using ..Api
@force using ..Utils

import Base.print
import Base.show

export initRepresentatives, kmeans, kmedoids

include("Clustering_hard.jl") # K-means and k-medoids
# MLJ interface
include("Clustering_MLJ.jl")

end

