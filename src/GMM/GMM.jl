"""
  GMM.jl file

Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT.
"""

"""
    GMM module

Generative (Gaussian) Mixed Model learners (supervised/unsupervised)

Provides clustering/collaborative filtering (via clustering) / missing values imputation / collaborative filtering / reccomendation systems, regressor and fitter using Generative Gaussiam Model (probabilistic). 

The module provides the following functions. Use `?[function]` to access their full signature and detailed documentation:

- [`gmm(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance,initStrategy)`](@ref gmm): gmm algorithm over GMM
- [`predictMissing(X,K;p₀,mixtures,tol,verbosity,minVariance,minCovariance)`](@ref predictMissing): Impute mixing values ("matrix completion") using gmm as backbone. Note that this can be used for collaborative filtering / reccomendation systems often with better results than traditional algorithms as k-nearest neighbors (KNN)

{Spherical|Diagonal|Full} Gaussian mixtures are already provided. User defined mixtures can be used defining a struct as subtype of `AbstractMixture` and implementing for that mixture the following functions:
- `initMixtures!(mixtures, X; minVariance, minCovariance, initStrategy)`
- `lpdf(m,x,mask)` (for the e-step)
- `updateParameters!(mixtures, X, pₙₖ; minVariance, minCovariance)` (the m-step)

"""
module GMM

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays
import Distributions

using  ForceImport
@force using ..Api
@force using ..Utils
@force using ..Clustering

import Base.print
import Base.show

export gmm, predictMissing, AbstractMixture,
       GMMClusterModel

abstract type AbstractMixture end

include("GMM_clustering.jl")
include("Mixtures.jl")
include("GMM_regression.jl")
# MLJ interface
include("GMM_MLJ.jl")

end

