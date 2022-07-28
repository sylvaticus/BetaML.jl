"""
  Clustering.jl file

Clustering and collaborative filtering (via clustering) algorithms

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Clustering.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/Clustering.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FClustering.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 4](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2004%20-%20Unsupervised%20Learning/Unit%2004%20-%20Unsupervised%20Learning.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)
"""

"""
    Clustering module (WIP)

Provide clustering methods and missing values imputation / collaborative filtering / reccomendation systems using clustering methods as backend.

The module provides the following functions. Use `?[function]` to access their full signature and detailed documentation:

- [`initRepresentatives(X,K;initStrategy,Z₀)`](@ref initRepresentatives): Initialisation strategies for Kmean and Kmedoids
- [`kmeans(X,K;dist,initStrategy,Z₀)`](@ref kmeans): Classical KMean algorithm
- [`kmedoids(X,K;dist,initStrategy,Z₀)`](@ref kmedoids): Kmedoids algorithm
- [`gmm(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance,initStrategy)`](@ref gmm): gmm algorithm over GMM
- [`predictMissing(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance)`](@ref predictMissing): Impute mixing values ("matrix completion") using gmm as backbone. Note that this can be used for collaborative filtering / reccomendation systems often with better results than traditional algorithms as k-nearest neighbors (KNN)

{Spherical|Diagonal|Full}Gaussian mixtures for `gmm` / `predictMissing` are already provided. User defined mixtures can be used defining a struct as subtype of `Mixture` and implementing for that mixture the following functions:
- `initMixtures!(mixtures, X; minVariance, minCovariance, initStrategy)`
- `lpdf(m,x,mask)` (for the e-step)
- `updateParameters!(mixtures, X, pₙₖ; minVariance, minCovariance)` (the m-step)

"""
module Clustering

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays
import Distributions

using  ForceImport
@force using ..Api
@force using ..Utils

import Base.print
import Base.show

export initRepresentatives, kmeans, kmedoids, gmm, predictMissing, AbstractMixture,
       KMeansModel, KMedoidsModel, GMMClusterModel

abstract type AbstractMixture end
include("Clustering_hard.jl") # K-means and k-medoids
include("Clustering_gmm.jl")
include("Mixtures.jl")
# MLJ interface
include("Clustering_MLJ.jl")

end

