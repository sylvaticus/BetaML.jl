"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
    GMM module 

Generative (Gaussian) Mixed Model learners (supervised/unsupervised)

Provides clustering and regressors using  (Generative) Gaussiam Mixture Model (probabilistic).

Collaborative filtering / missing values imputation / reccomendation systems based on GMM is available in the [`Imputation`](@ref BetaML.Imputation) module.

The module provides the following models. Use `?[model]` to access their documentation:

- [`GMMClusterer`](@ref): soft-clustering using GMM
- [`GMMRegressor1`](@ref): regressor using GMM as back-end (first algorithm)
- [`GMMRegressor1`](@ref): regressor using GMM as back-end (second algorithm)

All the algorithms works with arbitrary mixture distribution, altought only {Spherical|Diagonal|Full} Gaussian mixtures has been implemented. User defined mixtures can be used defining a struct as subtype of `AbstractMixture` and implementing for that mixture the following functions:
- `init_mixtures!(mixtures, X; minimum_variance, minimum_covariance, initialisation_strategy)`
- `lpdf(m,x,mask)` (for the e-step)
- `update_parameters!(mixtures, X, pₙₖ; minimum_variance, minimum_covariance)` (the m-step)
- `npar(mixtures::Array{T,1})` (for the BIC/AIC computation)


All the GMM-based algorithms works only with numerical data, but accepts also Missing one.

The `GMMClusterer` algorithm reports the `BIC` and the `AIC` in its `info(model)`, but some metrics of the clustered output are also available, for example the [`silhouette`](@ref) score.
"""
module GMM

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays, DocStringExtensions
import Distributions

using  ForceImport
@force using ..Api
@force using ..Utils
@force using ..Clustering

import Base.print
import Base.show

#export gmm, 
export AbstractMixture,
       GMMClusterer,
       GMMRegressor1, GMMRegressor2,
       GMMHyperParametersSet

abstract type AbstractMixture end

include("GMM_clustering.jl")
include("Mixtures.jl")
include("GMM_regression.jl")

# MLJ interface
include("GMM_MLJ.jl")

end

