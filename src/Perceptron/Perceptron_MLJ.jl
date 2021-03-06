# MLJ interface for Decision Trees/Random Forests models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repoeat it here

export PerceptronClassifier, KernelPerceptronClassifier, PegasosClassifier


# ------------------------------------------------------------------------------
# Model Structure declarations..

mutable struct PerceptronClassifier <: MMI.Probabilistic
   initialθ::Vector{Float64}
   initialθ₀::Float64
   maxEpochs::Int64
   shuffle::Bool
   forceOrigin::Bool
   returnMeanHyperplane::Bool
   rng::AbstractRNG
end
PerceptronClassifier(;
  initialθ=Float64[],
  initialθ₀=0.0,
  maxEpochs=1000,
  shuffle=false,
  forceOrigin=false,
  returnMeanHyperplane=false,
  rng = Random.GLOBAL_RNG,
  ) = PerceptronClassifier(initialθ,initialθ₀,maxEpochs,shuffle,forceOrigin,returnMeanHyperplane,rng)

#kernelPerceptron(x, y; K=radialKernel, T=100, α=zeros(Int64,length(y)), nMsgs=0, shuffle=false)
 mutable struct KernelPerceptronClassifier <: MMI.Probabilistic
     K::Function
     maxEpochs::Int64
     initialα::Vector{Int64}
     shuffle::Bool
     rng::AbstractRNG
 end
 KernelPerceptronClassifier(;
    K=radialKernel,
    maxEpochs=100,
    initialα = Int64[],
    shuffle=false,
    rng = Random.GLOBAL_RNG,
    ) = KernelPerceptronClassifier(K,maxEpochs,initialα,shuffle,rng)

# pegasos(x, y; θ=zeros(size(x,2)),θ₀=0.0, λ=0.5,η= (t -> 1/sqrt(t)), T=1000, nMsgs=10, shuffle=false,forceOrigin=false,returnMeanHyperplane=false
mutable struct PegasosClassifier <: MMI.Probabilistic
   initialθ::Vector{Float64}
   initialθ₀::Float64
   λ::Float64
   η::Function
   maxEpochs::Int64
   shuffle::Bool
   forceOrigin::Bool
   returnMeanHyperplane::Bool
   rng::AbstractRNG
end
PegasosClassifier(;
  initialθ=Float64[],
  initialθ₀=0.0,
  λ = 0.5,
  η = (t -> 1/sqrt(t)),
  maxEpochs=1000,
  shuffle=false,
  forceOrigin=false,
  returnMeanHyperplane=false,
  rng = Random.GLOBAL_RNG,
  ) = PegasosClassifier(initialθ,initialθ₀,λ,η,maxEpochs,shuffle,forceOrigin,returnMeanHyperplane,rng)

# ------------------------------------------------------------------------------
# Fit functions...

function MMI.fit(model::PerceptronClassifier, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 initialθ  = length(model.initialθ) == 0 ? zeros(size(x,2)) : model.initialθ
 fitresult = perceptron(x, y; θ=initialθ, θ₀=model.initialθ₀, T=model.maxEpochs, nMsgs=0, shuffle=model.shuffle, forceOrigin=model.forceOrigin, returnMeanHyperplane=model.returnMeanHyperplane,rng=model.rng)
 cache=nothing
 report=nothing
 return fitresult, cache, report
end

function MMI.fit(model::KernelPerceptronClassifier, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 initialα  = length(model.initialα) == 0 ? zeros(Int64,length(y)) : model.initialα
 fitresult = kernelPerceptron(x, y; K=model.K, T=model.maxEpochs, α=initialα, nMsgs=0, shuffle=model.shuffle,rng=model.rng)
 cache=nothing
 report=nothing
 return fitresult, cache, report
end

function MMI.fit(model::PegasosClassifier, verbosity, X, y)
 x = MMI.matrix(X)                     # convert table to matrix
 initialθ  = length(model.initialθ) == 0 ? zeros(size(x,2)) : model.initialθ
 fitresult = pegasos(x, y; θ=initialθ,θ₀=model.initialθ₀, λ=model.λ,η=model.η, T=model.maxEpochs, nMsgs=0, shuffle=model.shuffle, forceOrigin=model.forceOrigin, returnMeanHyperplane=model.returnMeanHyperplane,rng=model.rng)
 cache=nothing
 report=nothing
 return fitresult, cache, report
end

# ------------------------------------------------------------------------------
# Predict functions....

function MMI.predict(model::Union{PerceptronClassifier,PegasosClassifier}, fitresult, Xnew)
    fittedModel      = fitresult
    #classes          = CategoricalVector(fittedModel.classes)
    classes          = fittedModel.classes
    nLevels          = length(classes)
    nRecords         = MMI.nrows(Xnew)
    modelPredictions = Perceptron.predict(MMI.matrix(Xnew), fittedModel.θ, fittedModel.θ₀, fittedModel.classes)
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(classes)
            predMatrix[n,c] = get(modelPredictions[n],cl,0.0)
        end
    end
    #predictions = [MMI.UnivariateFinite(classes, predMatrix[i,:])
    #               for i in 1:nRecords]
    predictions = MMI.UnivariateFinite(fittedModel.classes,predMatrix,pool=missing)
    return predictions
end

function MMI.predict(model::KernelPerceptronClassifier, fitresult, Xnew)
    fittedModel      = fitresult
    #classes          = CategoricalVector(fittedModel.classes)
    classes          = fittedModel.classes
    nLevels          = length(classes)
    nRecords         = MMI.nrows(Xnew)
    #ŷtrain = Perceptron.predict([10 10; 2.2 2.5],model.x,model.y,model.α, model.classes,K=model.K)
    modelPredictions = Perceptron.predict(MMI.matrix(Xnew), fittedModel.x, fittedModel.y, fittedModel.α, fittedModel.classes, K=fittedModel.K)
    predMatrix       = zeros(Float64,(nRecords,nLevels))
    # Transform the predictions from a vector of dictionaries to a matrix
    # where the rows are the PMF of each record
    for n in 1:nRecords
        for (c,cl) in enumerate(classes)
            predMatrix[n,c] = get(modelPredictions[n],cl,0.0)
        end
    end
    #predictions = [MMI.UnivariateFinite(classes, predMatrix[i,:])
    #              for i in 1:nRecords]
    #predictions = MMI.UnivariateFinite(classes, predMatrix)
    predictions = MMI.UnivariateFinite(fittedModel.classes,predMatrix,pool=missing)
    #predictions4 = MMI.UnivariateFinite(modelPredictions,pool=classes,ordered=false)
    #predictions = MMI.UnivariateFinite(modelPredictions,pool=fittedModel.classes)
    return predictions
end

# ------------------------------------------------------------------------------
# Model metadata for registration in MLJ...

MMI.metadata_model(PerceptronClassifier,
    input_scitype    = MMI.Table(MMI.Infinite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "The classical perceptron algorithm using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Perceptron.PerceptronClassifier"
)

MMI.metadata_model(KernelPerceptronClassifier,
    input_scitype    = MMI.Table(MMI.Infinite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "The kernel perceptron algorithm using one-vs-one for multiclass, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Perceptron.KernelPerceptronClassifier"
)

MMI.metadata_model(PegasosClassifier,
    input_scitype    = MMI.Table(MMI.Infinite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "The gradient-based linear \"pegasos\" classifier using one-vs-all for multiclass, from the Beta Machine Learning Toolkit (BetaML).",
	load_path        = "BetaML.Perceptron.PegasosClassifier"
)
