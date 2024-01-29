"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Decision Trees/Random Forests models

export PerceptronClassifier, KernelPerceptronClassifier, PegasosClassifier


# ------------------------------------------------------------------------------
# Model Structure declarations..
"""
$(TYPEDEF)

The classical perceptron algorithm using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:
```julia
julia> using MLJ

julia> X, y        = @load_iris;

julia> modelType   = @load PerceptronClassifier pkg = "BetaML"
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
BetaML.Perceptron.PerceptronClassifier

julia> model       = modelType()
PerceptronClassifier(
  initial_coefficients = nothing, 
  initial_constant = nothing, 
  epochs = 1000, 
  shuffle = true, 
  force_origin = false, 
  return_mean_hyperplane = false, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);
[ Info: Training machine(PerceptronClassifier(initial_coefficients = nothing, …), …).
*** Avg. error after epoch 2 : 0.0 (all elements of the set has been correctly classified)
julia> est_classes = predict(mach, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt8, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>1.0, versicolor=>2.53e-34, virginica=>0.0)
 UnivariateFinite{Multiclass{3}}(setosa=>1.0, versicolor=>1.27e-18, virginica=>1.86e-310)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>2.77e-57, versicolor=>1.1099999999999999e-82, virginica=>1.0)
 UnivariateFinite{Multiclass{3}}(setosa=>3.09e-22, versicolor=>4.03e-25, virginica=>1.0)
```

"""
mutable struct PerceptronClassifier <: MMI.Probabilistic
   "N-classes by D-dimensions matrix of initial linear coefficients [def: `nothing`, i.e. zeros]"
   initial_coefficients::Union{Matrix{Float64},Nothing} 
   "N-classes vector of initial contant terms [def: `nothing`, i.e. zeros]"
   initial_constant::Union{Vector{Float64},Nothing} 
   "Maximum number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
   epochs::Int64
   "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
   shuffle::Bool
   "Whether to force the parameter associated with the constant term to remain zero [def: `false`]"
   force_origin::Bool
   "Whether to return the average hyperplane coefficients instead of the final ones  [def: `false`]"
   return_mean_hyperplane::Bool
   "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG
end
PerceptronClassifier(;
  initial_coefficients=nothing,
  initial_constant=nothing,
  epochs=1000,
  shuffle=true,
  force_origin=false,
  return_mean_hyperplane=false,
  rng = Random.GLOBAL_RNG,
  ) = PerceptronClassifier(initial_coefficients,initial_constant,epochs,shuffle,force_origin,return_mean_hyperplane,rng)

"""
$(TYPEDEF)

The kernel perceptron algorithm using one-vs-one for multiclass, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:
```julia
julia> using MLJ

julia> X, y        = @load_iris;

julia> modelType   = @load KernelPerceptronClassifier pkg = "BetaML"
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
BetaML.Perceptron.KernelPerceptronClassifier

julia> model       = modelType()
KernelPerceptronClassifier(
  kernel = BetaML.Utils.radial_kernel, 
  epochs = 100, 
  initial_errors = nothing, 
  shuffle = true, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);

julia> est_classes = predict(mach, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt8, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>0.665, versicolor=>0.245, virginica=>0.09)
 UnivariateFinite{Multiclass{3}}(setosa=>0.665, versicolor=>0.245, virginica=>0.09)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>0.09, versicolor=>0.245, virginica=>0.665)
 UnivariateFinite{Multiclass{3}}(setosa=>0.09, versicolor=>0.665, virginica=>0.245)
```

"""
mutable struct KernelPerceptronClassifier <: MMI.Probabilistic
    "Kernel function to employ. See `?radial_kernel` or `?polynomial_kernel` (once loaded the BetaML package) for details or check `?BetaML.Utils` to verify if other kernels are defined (you can alsways define your own kernel) [def: [`radial_kernel`](@ref)]"
    kernel::Function
    "Maximum number of epochs, i.e. passages trough the whole training sample [def: `100`]"
    epochs::Int64
    "Initial distribution of the number of errors errors [def: `nothing`, i.e. zeros]. If provided, this should be a nModels-lenght vector of nRecords integer values vectors , where nModels is computed as `(n_classes  * (n_classes - 1)) / 2`"
    initial_errors::Union{Nothing,Vector{Vector{Int64}}}
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool
    "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG
end
KernelPerceptronClassifier(;
    kernel=BetaML.Utils.radial_kernel,
    epochs=100,
    initial_errors = nothing,
    shuffle=true,
    rng = Random.GLOBAL_RNG,
    ) = KernelPerceptronClassifier(kernel,epochs,initial_errors,shuffle,rng)
"""
$(TYPEDEF)

The gradient-based linear "pegasos" classifier using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML).

# Hyperparameters:
$(TYPEDFIELDS)

# Example:
```julia
julia> using MLJ

julia> X, y        = @load_iris;

julia> modelType   = @load PegasosClassifier pkg = "BetaML" verbosity=0
BetaML.Perceptron.PegasosClassifier

julia> model       = modelType()
PegasosClassifier(
  initial_coefficients = nothing, 
  initial_constant = nothing, 
  learning_rate = BetaML.Perceptron.var"#71#73"(), 
  learning_rate_multiplicative = 0.5, 
  epochs = 1000, 
  shuffle = true, 
  force_origin = false, 
  return_mean_hyperplane = false, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);

julia> est_classes = predict(mach, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt8, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>0.817, versicolor=>0.153, virginica=>0.0301)
 UnivariateFinite{Multiclass{3}}(setosa=>0.791, versicolor=>0.177, virginica=>0.0318)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>0.254, versicolor=>0.5, virginica=>0.246)
 UnivariateFinite{Multiclass{3}}(setosa=>0.283, versicolor=>0.51, virginica=>0.207)
```
"""
mutable struct PegasosClassifier <: MMI.Probabilistic
    "N-classes by D-dimensions matrix of initial linear coefficients [def: `nothing`, i.e. zeros]"
   initial_coefficients::Union{Matrix{Float64},Nothing} 
   "N-classes vector of initial contant terms [def: `nothing`, i.e. zeros]"
   initial_constant::Union{Vector{Float64},Nothing} 
   "Learning rate [def: (epoch -> 1/sqrt(epoch))]"
   learning_rate::Function
   "Multiplicative term of the learning rate [def: `0.5`]"       
   learning_rate_multiplicative::Float64
   "Maximum number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
   epochs::Int64
   "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
   shuffle::Bool
   "Whether to force the parameter associated with the constant term to remain zero [def: `false`]"
   force_origin::Bool
   "Whether to return the average hyperplane coefficients instead of the final ones  [def: `false`]"
   return_mean_hyperplane::Bool
   "A Random Number Generator to be used in stochastic parts of the code [deafult: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG
end
PegasosClassifier(;
  initial_coefficients=nothing,
  initial_constant=nothing,
  learning_rate = (t -> 1/sqrt(t)),
  learning_rate_multiplicative = 0.5,
  epochs=1000,
  shuffle=true,
  force_origin=false,
  return_mean_hyperplane=false,
  rng = Random.GLOBAL_RNG,
  ) = PegasosClassifier(initial_coefficients,initial_constant,learning_rate,learning_rate_multiplicative,epochs,shuffle,force_origin,return_mean_hyperplane,rng)

# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(model::PerceptronClassifier, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 allClasses = levels(y)
 typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
 verbosity = mljverbosity_to_betaml_verbosity(verbosity)
 #initial_coefficients  = length(model.initial_coefficients) == 0 ? zeros(size(x,2)) : model.initial_coefficients
 fitresult = BetaML.Perceptron.perceptron(x, y; θ=model.initial_coefficients, θ₀=model.initial_constant, T=model.epochs, nMsgs=0, shuffle=model.shuffle, force_origin=model.force_origin, return_mean_hyperplane=model.return_mean_hyperplane,rng=model.rng, verbosity=verbosity)
 cache=nothing
 report=nothing
 return (fitresult,allClasses), cache, report
end

function MMI.fit(model::KernelPerceptronClassifier, verbosity, X, y)
 x          = MMI.matrix(X)                     # convert table to matrix
 allClasses = levels(y)
 typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
 verbosity = mljverbosity_to_betaml_verbosity(verbosity)
 #initial_errors   = length(model.initial_errors) == 0 ? zeros(Int64,length(y)) : model.initial_errors
 fitresult  = BetaML.Perceptron.kernel_perceptron_classifier(x, y; K=model.kernel, T=model.epochs, α=model.initial_errors, nMsgs=0, shuffle=model.shuffle,rng=model.rng, verbosity=verbosity)
 cache      = nothing
 report     = nothing
 return (fitresult,allClasses), cache, report
end

function MMI.fit(model::PegasosClassifier, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 allClasses = levels(y)
 typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
 verbosity = mljverbosity_to_betaml_verbosity(verbosity)
 #initial_coefficients  = length(model.initial_coefficients) == 0 ? zeros(size(x,2)) : model.initial_coefficients
 fitresult = BetaML.Perceptron.pegasos(x, y; θ=model.initial_coefficients,θ₀=model.initial_constant, λ=model.learning_rate_multiplicative,η=model.learning_rate, T=model.epochs, nMsgs=0, shuffle=model.shuffle, force_origin=model.force_origin, return_mean_hyperplane=model.return_mean_hyperplane,rng=model.rng, verbosity=verbosity)
 cache=nothing
 report=nothing
 return (fitresult,allClasses), cache, report
end

# ------------------------------------------------------------------------------
# Predict functions....
function MMI.predict(model::Union{PerceptronClassifier,PegasosClassifier}, fitresult, Xnew)
    fittedModel      = fitresult[1]
    #classes          = CategoricalVector(fittedModel.classes)
    classes          = fittedModel.classes
    allClasses       = fitresult[2] # as classes do not includes classes unsees at training time
    nLevels          = length(allClasses)
    nRecords         = MMI.nrows(Xnew)
    modelPredictions = BetaML.Perceptron.predict(MMI.matrix(Xnew), fittedModel.θ, fittedModel.θ₀, fittedModel.classes)
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(allClasses)
            predMatrix[n,c] = get(modelPredictions[n],cl,0.0)
        end
    end
    #predictions = [MMI.UnivariateFinite(classes, predMatrix[i,:])
    #               for i in 1:nRecords]
    predictions = MMI.UnivariateFinite(allClasses,predMatrix,pool=missing)
    return predictions
end

function MMI.predict(model::KernelPerceptronClassifier, fitresult, Xnew)
    fittedModel      = fitresult[1]
    #classes          = CategoricalVector(fittedModel.classes)
    classes          = fittedModel.classes
    allClasses       = fitresult[2] # as classes do not includes classes unsees at training time
    nLevels          = length(allClasses)
    nRecords         = MMI.nrows(Xnew)
    #ŷtrain = Perceptron.predict([10 10; 2.2 2.5],model.x,model.y,model.α, model.classes,K=model.K)
    modelPredictions = BetaML.Perceptron.predict(MMI.matrix(Xnew), fittedModel.x, fittedModel.y, fittedModel.α, fittedModel.classes, K=fittedModel.K)
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(allClasses)
            predMatrix[n,c] = get(modelPredictions[n],cl,0.0)
        end
    end
    #predictions = [MMI.UnivariateFinite(classes, predMatrix[i,:])
    #              for i in 1:nRecords]
    #predictions = MMI.UnivariateFinite(classes, predMatrix)
    predictions = MMI.UnivariateFinite(allClasses,predMatrix,pool=missing)
    #predictions4 = MMI.UnivariateFinite(modelPredictions,pool=classes,ordered=false)
    #predictions = MMI.UnivariateFinite(modelPredictions,pool=fittedModel.classes)
    return predictions
end

# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(PerceptronClassifier,
    input_scitype = Union{
        MMI.Table(MMI.Infinite),
        AbstractMatrix{<: MMI.Infinite},
    },
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
	load_path        = "BetaML.Bmlj.PerceptronClassifier"
)

MMI.metadata_model(KernelPerceptronClassifier,
    input_scitype = Union{
        MMI.Table(MMI.Infinite),
        AbstractMatrix{<: MMI.Infinite},
    },
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
	load_path        = "BetaML.Bmlj.KernelPerceptronClassifier"
)

MMI.metadata_model(PegasosClassifier,
    input_scitype = Union{
        MMI.Table(MMI.Infinite),
        AbstractMatrix{<: MMI.Infinite},
    },
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
	load_path        = "BetaML.Bmlj.PegasosClassifier"
)
