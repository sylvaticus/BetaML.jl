"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Decision Trees/Random Forests models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export LinearPerceptron, KernelPerceptron, Pegasos


# ------------------------------------------------------------------------------
# Model Structure declarations..
"The classical perceptron algorithm using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML)."
mutable struct LinearPerceptron <: MMI.Probabilistic
   initialθ::Union{Matrix{Float64},Nothing} 
   initialθ₀::Union{Vector{Float64},Nothing} 
   maxEpochs::Int64
   shuffle::Bool
   force_origin::Bool
   return_mean_hyperplane::Bool
   rng::AbstractRNG
end
LinearPerceptron(;
  initialθ=nothing,
  initialθ₀=nothing,
  maxEpochs=1000,
  shuffle=false,
  force_origin=false,
  return_mean_hyperplane=false,
  rng = Random.GLOBAL_RNG,
  ) = LinearPerceptron(initialθ,initialθ₀,maxEpochs,shuffle,force_origin,return_mean_hyperplane,rng)

"The kernel perceptron algorithm using one-vs-one for multiclass, from the Beta Machine Learning Toolkit (BetaML)."
mutable struct KernelPerceptron <: MMI.Probabilistic
     K::Function
     maxEpochs::Int64
     initialα::Union{Nothing,Vector{Vector{Int64}}}
     shuffle::Bool
     rng::AbstractRNG
end
KernelPerceptron(;
    K=radialKernel,
    maxEpochs=100,
    initialα = nothing,
    shuffle=false,
    rng = Random.GLOBAL_RNG,
    ) = KernelPerceptron(K,maxEpochs,initialα,shuffle,rng)

"The gradient-based linear \"pegasos\" classifier using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML)."
mutable struct Pegasos <: MMI.Probabilistic
   initialθ::Union{Matrix{Float64},Nothing} 
   initialθ₀::Union{Vector{Float64},Nothing} 
   λ::Float64
   η::Function
   maxEpochs::Int64
   shuffle::Bool
   force_origin::Bool
   return_mean_hyperplane::Bool
   rng::AbstractRNG
end
Pegasos(;
  initialθ=nothing,
  initialθ₀=nothing,
  λ = 0.5,
  η = (t -> 1/sqrt(t)),
  maxEpochs=1000,
  shuffle=false,
  force_origin=false,
  return_mean_hyperplane=false,
  rng = Random.GLOBAL_RNG,
  ) = Pegasos(initialθ,initialθ₀,λ,η,maxEpochs,shuffle,force_origin,return_mean_hyperplane,rng)

# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(model::LinearPerceptron, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 allClasses = levels(y)
 #initialθ  = length(model.initialθ) == 0 ? zeros(size(x,2)) : model.initialθ
 fitresult = perceptron(x, y; θ=model.initialθ, θ₀=model.initialθ₀, T=model.maxEpochs, nMsgs=0, shuffle=model.shuffle, force_origin=model.force_origin, return_mean_hyperplane=model.return_mean_hyperplane,rng=model.rng)
 cache=nothing
 report=nothing
 return (fitresult,allClasses), cache, report
end

function MMI.fit(model::KernelPerceptron, verbosity, X, y)
 x          = MMI.matrix(X)                     # convert table to matrix
 allClasses = levels(y)
 #initialα   = length(model.initialα) == 0 ? zeros(Int64,length(y)) : model.initialα
 fitresult  = kernelPerceptron(x, y; K=model.K, T=model.maxEpochs, α=model.initialα, nMsgs=0, shuffle=model.shuffle,rng=model.rng)
 cache      = nothing
 report     = nothing
 return (fitresult,allClasses), cache, report
end

function MMI.fit(model::Pegasos, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 allClasses = levels(y)
 #initialθ  = length(model.initialθ) == 0 ? zeros(size(x,2)) : model.initialθ
 fitresult = pegasos(x, y; θ=model.initialθ,θ₀=model.initialθ₀, λ=model.λ,η=model.η, T=model.maxEpochs, nMsgs=0, shuffle=model.shuffle, force_origin=model.force_origin, return_mean_hyperplane=model.return_mean_hyperplane,rng=model.rng)
 cache=nothing
 report=nothing
 return (fitresult,allClasses), cache, report
end

# ------------------------------------------------------------------------------
# Predict functions....

function MMI.predict(model::Union{LinearPerceptron,Pegasos}, fitresult, Xnew)
    fittedModel      = fitresult[1]
    #classes          = CategoricalVector(fittedModel.classes)
    classes          = fittedModel.classes
    allClasses       = fitresult[2] # as classes do not includes classes unsees at training time
    nLevels          = length(allClasses)
    nRecords         = MMI.nrows(Xnew)
    modelPredictions = Perceptron.predict(MMI.matrix(Xnew), fittedModel.θ, fittedModel.θ₀, fittedModel.classes)
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

function MMI.predict(model::KernelPerceptron, fitresult, Xnew)
    fittedModel      = fitresult[1]
    #classes          = CategoricalVector(fittedModel.classes)
    classes          = fittedModel.classes
    allClasses       = fitresult[2] # as classes do not includes classes unsees at training time
    nLevels          = length(allClasses)
    nRecords         = MMI.nrows(Xnew)
    #ŷtrain = Perceptron.predict([10 10; 2.2 2.5],model.x,model.y,model.α, model.classes,K=model.K)
    modelPredictions = Perceptron.predict(MMI.matrix(Xnew), fittedModel.x, fittedModel.y, fittedModel.α, fittedModel.classes, K=fittedModel.K)
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

MMI.metadata_model(LinearPerceptron,
    input_scitype    = MMI.Table(MMI.Infinite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "The classical perceptron algorithm using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Perceptron.LinearPerceptron"
)

MMI.metadata_model(KernelPerceptron,
    input_scitype    = MMI.Table(MMI.Infinite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "The kernel perceptron algorithm using one-vs-one for multiclass, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Perceptron.KernelPerceptron"
)

MMI.metadata_model(Pegasos,
    input_scitype    = MMI.Table(MMI.Infinite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "The gradient-based linear \"pegasos\" classifier using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Perceptron.Pegasos"
)
