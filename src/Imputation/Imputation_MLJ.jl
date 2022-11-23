"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for imputers models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here

export SimpleImputer,GaussianMixtureImputer, RandomForestImputer, GeneralImputer

"""
$(TYPEDEF)

Impute missing values using feature (column) mean, with optional record normalisation (using l-`norm` norms), from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:
```julia
julia> using MLJ

julia> X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4] |> table ;

julia> modelType                   = @load SimpleImputer pkg = "BetaML" verbosity=0
BetaML.Imputation.SimpleImputer

julia> model                       = modelType(norm=1)
SimpleImputer(
  statistic = Statistics.mean, 
  norm = 1)

julia> (fitResults, cache, report) = MLJ.fit(model, 0, X);

julia> X_full                      = transform(model, fitResults, X) |> MLJ.matrix
9×2 Matrix{Float64}:
 1.0        10.5
 1.5         0.295466
 1.8         8.0
 1.7        15.0
 3.2        40.0
 0.280952    1.69524
 3.3        38.0
 0.0750839  -2.3
 5.2        -2.4
```


"""
mutable struct SimpleImputer <: MMI.Unsupervised
    "The descriptive statistic of the column (feature) to use as imputed value [def: `mean`]"
    statistic::Function
    "Normalise the feature mean by l-`norm` norm of the records [default: `nothing`]. Use it (e.g. `norm=1` to use the l-1 norm) if the records are highly heterogeneus (e.g. quantity exports of different countries)."
    norm::Union{Nothing,Int64}
end
SimpleImputer(;
    statistic::Function              = mean,
    norm::Union{Nothing,Int64}       = nothing,
) = SimpleImputer(statistic,norm)

"""
$(TYPEDEF)

Impute missing values using a probabilistic approach (Gaussian Mixture Models) fitted using the Expectation-Maximisation algorithm, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example :
```julia
julia> using MLJ

julia> X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4] |> table ;

julia> modelType                   = @load GaussianMixtureImputer pkg = "BetaML" verbosity=0
BetaML.Imputation.GaussianMixtureImputer

julia> model                       = modelType(initialisation_strategy="grid")
GaussianMixtureImputer(
  n_classes = 3, 
  initial_probmixtures = Float64[], 
  mixtures = BetaML.GMM.DiagonalGaussian{Float64}[BetaML.GMM.DiagonalGaussian{Float64}(nothing, nothing), BetaML.GMM.DiagonalGaussian{Float64}(nothing, nothing), BetaML.GMM.DiagonalGaussian{Float64}(nothing, nothing)], 
  tol = 1.0e-6, 
  minimum_variance = 0.05, 
  minimum_covariance = 0.0, 
  initialisation_strategy = "grid", 
  rng = Random._GLOBAL_RNG())

julia> (fitResults, cache, report) = MLJ.fit(model, 0, X);

julia> X_full                      = transform(model, fitResults, X) |> MLJ.matrix
9×2 Matrix{Float64}:
 1.0      10.5
 1.5      14.7366
 1.8       8.0
 1.7      15.0
 3.2      40.0
 2.51842  15.1747
 3.3      38.0
 2.47412  -2.3
 5.2      -2.4
```

"""
mutable struct GaussianMixtureImputer <: MMI.Unsupervised
    "Number of mixtures (latent classes) to consider [def: 3]"
    n_classes::Int64
    "Initial probabilities of the categorical distribution (n_classes x 1) [default: `[]`]"
    initial_probmixtures::Vector{Float64}
    """An array (of length `n_classes``) of the mixtures to employ (see the [`?GMM`](@ref GMM) module in BetaML).
    Each mixture object can be provided with or without its parameters (e.g. mean and variance for the gaussian ones). Fully qualified mixtures are useful only if the `initialisation_strategy` parameter is  set to \"gived\"`
    This parameter can also be given symply in term of a _type_. In this case it is automatically extended to a vector of `n_classes`` mixtures of the specified type.
    Note that mixing of different mixture types is not currently supported and that currently implemented mixtures are `SphericalGaussian`, `DiagonalGaussian` and `FullGaussian`.
    [def: `DiagonalGaussian`]"""
    mixtures::Union{Type,Vector{<: AbstractMixture}}
    "Tolerance to stop the algorithm [default: 10^(-6)]"
    tol::Float64
    "Minimum variance for the mixtures [default: 0.05]"
    minimum_variance::Float64
    "Minimum covariance for the mixtures with full covariance matrix [default: 0]. This should be set different than minimum_variance."
    minimum_covariance::Float64
    """
    The computation method of the vector of the initial mixtures.
    One of the following:
    - "grid": using a grid approach
    - "given": using the mixture provided in the fully qualified `mixtures` parameter
    - "kmeans": use first kmeans (itself initialised with a "grid" strategy) to set the initial mixture centers [default]
    Note that currently "random" and "shuffle" initialisations are not supported in gmm-based algorithms.
    """
    initialisation_strategy::String
    "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG
end
function GaussianMixtureImputer(;
    n_classes      = 3,
    initial_probmixtures  = Float64[],
    mixtures      = DiagonalGaussian, #[DiagonalGaussian() for i in 1:n_classes],
    tol           = 10^(-6),
    minimum_variance   = 0.05,
    minimum_covariance = 0.0,
    initialisation_strategy  = "kmeans",
    rng           = Random.GLOBAL_RNG,
)
    if typeof(mixtures) <: UnionAll
        mixtures = [mixtures() for i in 1:n_classes]
    end
    return GaussianMixtureImputer(n_classes,initial_probmixtures,mixtures, tol, minimum_variance, minimum_covariance,initialisation_strategy,rng)
end

"""
$(TYPEDEF)

Impute missing values using Random Forests, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:
```julia
julia> using MLJ

julia> X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4] |> table ;

julia> modelType                   = @load RandomForestImputer pkg = "BetaML" verbosity=0
BetaML.Imputation.RandomForestImputer

julia> model                       = modelType(n_trees=40)
RandomForestImputer(
  n_trees = 40, 
  max_depth = nothing, 
  min_gain = 0.0, 
  min_records = 2, 
  max_features = nothing, 
  forced_categorical_cols = Int64[], 
  splitting_criterion = nothing, 
  recursive_passages = 1, 
  rng = Random._GLOBAL_RNG())

julia> (fitResults, cache, report) = MLJ.fit(model, 0, X);

julia> X_full                      = transform(model, fitResults, X) |> MLJ.matrix
9×2 Matrix{Float64}:
 1.0    10.5
 1.5    10.3333
 1.8     8.0
 1.7    15.0
 3.2    40.0
 2.415   8.6545
 3.3    38.0
 3.72   -2.3
 5.2    -2.4
```

"""
mutable struct RandomForestImputer <: MMI.Unsupervised
    "Number of (decision) trees in the forest [def: `30`]"
    n_trees::Int64
    "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `nothing`, i.e. no limits]"
    max_depth::Union{Nothing,Int64}
    "The minimum information gain to allow for a node's partition [def: `0`]"
    min_gain::Float64
    "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
    min_records::Int64
    "The maximum number of (random) features to consider at each partitioning [def: `nothing`, i.e. square root of the data dimension]"
    max_features::Union{Nothing,Int64}
    "Specify the positions of the integer columns to treat as categorical instead of cardinal. [Default: empty vector (all numerical cols are treated as cardinal by default and the others as categorical)]"
    forced_categorical_cols::Vector{Int64}
    "Either `gini`, `entropy` or `variance`. This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `nothing`, i.e. `gini` for categorical labels (classification task) and `variance` for numerical labels(regression task)]. It can be an anonymous function."
    splitting_criterion::Union{Nothing,Function}
    "Define the times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`]."
    recursive_passages::Int64                  
    "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG
end
RandomForestImputer(;
    n_trees                 = 30, 
    max_depth               = nothing,
    min_gain                = 0.0,
    min_records             = 2,
    max_features            = nothing,
    forced_categorical_cols  = Int64[],
    splitting_criterion     = nothing,
    recursive_passages      = 1,
    #multiple_imputations    = 1,
    rng                    = Random.GLOBAL_RNG,
) = RandomForestImputer(n_trees, max_depth, min_gain, min_records, max_features, forced_categorical_cols, splitting_criterion, recursive_passages, rng)

"""
$(TYPEDEF)

Impute missing values using arbitrary learning models, from the Beta Machine Learning Toolkit (BetaML).

Impute missing values using a vector (one per column) of arbitrary learning models (classifiers/regressors, not necessarily from BetaML) that:
- implement the interface `m = Model([options])`, `train!(m,X,Y)` and `predict(m,X)`;
- accept missing data in the feature matrix.
(default to Random Forests)


# Hyperparameters:
$(TYPEDFIELDS)

# Example :
```julia
julia> using MLJ

julia> X = ["a" 10.5;"a" missing; "b" 8; "b" 15; "c" 40; missing missing; "c" 38; missing -2.3; "c" -2.4] |> table ;

julia> modelType                   = @load GeneralImputer pkg = "BetaML" verbosity=0
GeneralImputer

julia> model                       = modelType(estimators=[BetaML.DecisionTreeEstimator(),BetaML.RandomForestEstimator(n_trees=40)],recursive_passages=2)
GeneralImputer(
  estimators = BetaMLSupervisedModel[DecisionTreeEstimator - A Decision Tree model (unfitted), RandomForestEstimator - A 40 trees Random Forest model (unfitted)], 
  recursive_passages = 2, 
  rng = Random._GLOBAL_RNG())

julia> (fitResults, cache, report) = MLJ.fit(model, 0, X);

julia> X_full                      = transform(model, fitResults, X) |> MLJ.matrix
9×2 Matrix{Any}:
 "a"  10.5
 "a"  10.5
 "b"   8
 "b"  15
 "c"  40
 "a"  10.5
 "c"  38
 "c"  -2.3
 "c"  -2.4
```
"""
mutable struct GeneralImputer <: MMI.Unsupervised
    "A D-dimensions vector of regressor or classifier models (and eventually their respective options/hyper-parameters) to be used to impute the various columns of the matrix [default: `nothing`, i.e. use random forests]."
    estimators::Union{Vector,Nothing}
    "Define the times to go trough the various columns to impute their data. Useful when there are data to impute on multiple columns. The order of the first passage is given by the decreasing number of missing values per column, the other passages are random [default: `1`]."
    recursive_passages::Int64                  
    "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG
end
GeneralImputer(;
    estimators               = nothing,
    recursive_passages    = 1,
    #multiple_imputations  = 1,
    rng                  = Random.GLOBAL_RNG,
) = GeneralImputer(estimators, recursive_passages, rng)


# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(m::SimpleImputer, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    mod = FeatureBasedImputer(
        statistic = m.statistic,
        norm      = m.norm,
        verbosity = verbosity,
    )
    fit!(mod,x)
    #fitResults = MMI.table(predict(mod))
    fitResults = mod
    cache      = nothing
    report     = info(mod)
    return (fitResults, cache, report)
end

function MMI.fit(m::GaussianMixtureImputer, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    #=if m.mixtures == :diag_gaussian
        mixtures = [DiagonalGaussian() for i in 1:m.n_classes]
    elseif m.mixtures == :full_gaussian
        mixtures = [FullGaussian() for i in 1:m.n_classes]
    elseif m.mixtures == :spherical_gaussian
        mixtures = [SphericalGaussian() for i in 1:m.n_classes]
    else
        error("Usupported mixture. Supported mixtures are either `:diag_gaussian`, `:full_gaussian` or `:spherical_gaussian`.")
    end
    =#

    mod = GMMImputer(
        n_classes      = m.n_classes,
        initial_probmixtures  = m.initial_probmixtures,
        mixtures      = m.mixtures,
        tol           = m.tol,
        minimum_variance   = m.minimum_variance,
        minimum_covariance = m.minimum_covariance,
        initialisation_strategy  = m.initialisation_strategy,
        verbosity     = verbosity,
        rng           = m.rng
    )
    fit!(mod,x)
    #fitResults = MMI.table(predict(mod))
    fitResults = mod
    cache      = nothing
    report     = info(mod)

    return (fitResults, cache, report)  
end

function MMI.fit(m::RandomForestImputer, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    mod = RFImputer(
        n_trees                 = m.n_trees, 
        max_depth               = m.max_depth,
        min_gain                = m.min_gain,
        min_records             = m.min_records,
        max_features            = m.max_features,
        forced_categorical_cols  = m.forced_categorical_cols,
        splitting_criterion     = m.splitting_criterion,
        verbosity              = verbosity,
        recursive_passages      = m.recursive_passages,
        #multiple_imputations    = m.multiple_imputations,
        rng                    = m.rng,
    )
    fit!(mod,x)
    #if m.multiple_imputations == 1
    #    fitResults = MMI.table(predict(mod))
    #else
    #    fitResults = MMI.table.(predict(mod))
    #end
    fitResults = mod
    cache      = nothing
    report     = info(mod)
    return (fitResults, cache, report)
end

function MMI.fit(m::GeneralImputer, verbosity, X)
    x          = MMI.matrix(X) # convert table to matrix
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    mod =  UniversalImputer(
        estimators             = m.estimators,
        verbosity              = verbosity,
        recursive_passages      = m.recursive_passages,
        #multiple_imputations    = m.multiple_imputations,
        rng                    = m.rng,
    )
    fit!(mod,x)
    #if m.multiple_imputations == 1
    #    fitResults = MMI.table(predict(mod))
    #else
    #    fitResults = MMI.table.(predict(mod))
    #end
    fitResults = mod
    cache      = nothing
    report     = info(mod)
    return (fitResults, cache, report)
end

# ------------------------------------------------------------------------------
# Transform functions...

""" transform(m, fitResults, X) - Given a trained imputator model fill the missing data of some new observations"""
function MMI.transform(m::Union{SimpleImputer,GaussianMixtureImputer,RandomForestImputer,GeneralImputer}, fitResults, X)
    x   = MMI.matrix(X) # convert table to matrix
    mod = fitResults
    return MMI.table(predict(mod,x))
end


# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(SimpleImputer,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
	load_path        = "BetaML.Imputation.SimpleImputer"
)

MMI.metadata_model(GaussianMixtureImputer,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Missing}),
    output_scitype   = MMI.Table(MMI.Continuous),     # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
	load_path        = "BetaML.Imputation.GaussianMixtureImputer"
)

MMI.metadata_model(RandomForestImputer,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    output_scitype   = MMI.Table(MMI.Known),          # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
	load_path        = "BetaML.Imputation.RandomForestImputer"
)
MMI.metadata_model(GeneralImputer,
    input_scitype    = MMI.Table(Union{MMI.Missing, MMI.Known}),
    output_scitype   = MMI.Table(MMI.Known),          # for an unsupervised, what output?
    supports_weights = false,                         # does the model support sample weights?
	load_path        = "BetaML.Imputation.GeneralImputer"
)
