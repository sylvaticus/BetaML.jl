"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
    GMM module 

Generative (Gaussian) Mixture Model learners (supervised/unsupervised)

Provides clustering and regressors using  (Generative) Gaussian Mixture Model (probabilistic).

Collaborative filtering / missing values imputation / recommendation systems based on GMM is available in the `Imputation` module.

The module provides the following models. Use `?[model]` to access their documentation:

- [`GaussianMixtureClusterer`](@ref): soft-clustering using GMM
- [`GaussianMixtureRegressor`](@ref): regressor using GMM as back-end (mixtures fitted on the combined X and Y matrix)
- [`GaussianMixtureRegressor2`](@ref): regressor using GMM as back-end (mixtures fitted on the X matrix only)

All the algorithms work with arbitrary mixture distributions, although only {Spherical|Diagonal|Full} Gaussian mixtures have been implemented. User defined mixtures can be used by defining a struct as subtype of `AbstractMixture` and implementing for that mixture the following functions:
- `init_mixtures!(mixtures, X; minimum_variance, minimum_covariance, initialisation_strategy)`
- `lpdf(m,x,mask)` (for the e-step)
- `update_parameters!(mixtures, X, pₙₖ; minimum_variance, minimum_covariance)` (the m-step)
- `npar(mixtures::Array{T,1})` (for the BIC/AIC computation)


All the GMM-based algorithms work only with numerical data, but also accept Missing ones.

The `GaussianMixtureClusterer` algorithm reports the `BIC` and the `AIC` in its `info(model)`, but some metrics of the clustered output are also available, for example the [`silhouette`](@ref) score.
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
       GaussianMixtureClusterer,
       GaussianMixtureRegressor2, GaussianMixtureRegressor,
       GaussianMixture_hp

abstract type AbstractMixture end

include("GMM_clustering.jl")
include("Mixtures.jl")
include("GMM_regression.jl")

end

