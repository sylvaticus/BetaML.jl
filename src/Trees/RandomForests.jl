      
# ------------------------------------------------------------------------------
# TYPE HIERARCHY AND DEFINITIONS

"""
    Forest{Ty}

Type representing a Random Forest.

Individual trees are stored in the array `trees`. The "type" of the forest is given by the type of the labels on which it has been trained.

# Struct members:
- `trees`:        The individual Decision Trees
- `isRegression`: Whether the forest is to be used for regression jobs or classification
- `oobData`:      For each tree, the rows number if the data that have _not_ being used to train the specific tree
- `oobError`:     The out of bag error (if it has been computed)
- `weights`:      A weight for each tree depending on the tree's score on the oobData (see [`buildForest`](@ref))
"""
mutable struct Forest{Ty} <: BetaMLLearnableParametersSet
    trees::Array{Union{AbstractDecisionNode,Leaf{Ty}},1}
    isRegression::Bool
    oobData::Array{Array{Int64,1},1}
    oobError::Float64
    weights::Array{Float64,1}
end

# Api V2..
Base.@kwdef mutable struct RFHyperParametersSet <: BetaMLHyperParametersSet
    nTrees::Int64                               = 30
    maxDepth::Union{Nothing,Int64}              = nothing
    minGain::Float64                            = 0.0
    minRecords::Int64                           = 2
    maxFeatures::Union{Nothing,Int64}           = nothing
    forceClassification::Bool                   = false
    splittingCriterion::Union{Nothing,Function} = nothing
    beta::Float64                               = 0.0
    oob::Bool                                   = false
end

#=
Base.@kwdef mutable struct RFOptionsSet <: BetaMLOptionsSet
    rng                  = Random.GLOBAL_RNG
    verbosity::Verbosity = STD
end
=#

mutable struct RFModel <: BetaMLSupervisedModel
    hpar::RFHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,Forest} #TODO: Forest contain info that is actualy in report. Currently we duplicate, we should just remofe them from par by making a dedicated struct instead of Forest
    cres
    fitted::Bool
    info::Dict{Symbol,Any}
end

function RFModel(;kwargs...)
m              = RFModel(RFHyperParametersSet(),BetaMLDefaultOptionsSet(),nothing,nothing,false,Dict{Symbol,Any}())
thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
for (kw,kwv) in kwargs
    found = false
    for f in thisobjfields
        fobj = getproperty(m,f)
        if kw in fieldnames(typeof(fobj))
            setproperty!(fobj,kw,kwv)
            found = true
        end
    end
    found || error("Keyword \"$kw\" is not part of this model.")
end
return m
end

# ------------------------------------------------------------------------------
# MODEL ALGORITHMS AND TRAINING

"""
   buildForest(x, y, nTrees; maxDepth, minGain, minRecords, maxFeatures, splittingCriterion, forceClassification)

Builds (define and train) a "forest" of Decision Trees.


# Parameters:
See [`buildTree`](@ref). The function has all the parameters of `bildTree` (with the `maxFeatures` defaulting to `√D` instead of `D`) plus the following parameters:
- `nTrees`: Number of trees in the forest [def: `30`]
- `β`: Parameter that regulate the weights of the scoring of each tree, to be (optionally) used in prediction (see later) [def: `0`, i.e. uniform weigths]
- `oob`: Whether to coompute the out-of-bag error, an estimation of the generalization accuracy [def: `false`]
- `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Output:
- The function returns a Forest object (see [`Forest`](@ref)).
- The forest weights default to array of ones if `β ≤ 0` and the oob error to `+Inf` if `oob` == `false`.

# Notes :
- Each individual decision tree is built using bootstrap over the data, i.e. "sampling N records with replacement" (hence, some records appear multiple times and some records do not appear in the specific tree training). The `maxFeature` injects further variability and reduces the correlation between the forest trees.
- The predictions of the "forest" (using the function `predict()`) are then the aggregated predictions of the individual trees (from which the name "bagging": **b**oostrap **agg**regat**ing**).
- This function optionally reports a weight distribution of the performances of eanch individual trees, as measured using the records he has not being trained with. These weights can then be (optionally) used in the `predict` function. The parameter `β ≥ 0` regulate the distribution of these weights: larger is `β`, the greater the importance (hence the weights) attached to the best-performing trees compared to the low-performing ones. Using these weights can significantly improve the forest performances (especially using small forests), however the correct value of β depends on the problem under exam (and the chosen caratteristics of the random forest estimator) and should be cross-validated to avoid over-fitting.
- Note that this function uses multiple threads if these are available. You can check the number of threads available with `Threads.nthreads()`. To set the number of threads in Julia either set the environmental variable `JULIA_NUM_THREADS` (before starting Julia) or start Julia with the command line option `--threads` (most integrated development editors for Julia already set the number of threads to 4).
"""
function buildForest(x, y::AbstractArray{Ty,1}, nTrees=30; maxDepth = size(x,1), minGain=0.0, minRecords=2, maxFeatures=Int(round(sqrt(size(x,2)))), forceClassification=false, splittingCriterion = (Ty <: Number && !forceClassification) ? variance : gini, β=0, oob=false,rng = Random.GLOBAL_RNG) where {Ty}
    # Force what would be a regression task into a classification task
    if forceClassification && Ty <: Number
        y = string.(y)
    end
    trees            = Array{Union{AbstractDecisionNode,Leaf{Ty}},1}(undef,nTrees)
    notSampledByTree = Array{Array{Int64,1},1}(undef,nTrees) # to later compute the Out of Bag Error

    errors = Float64[]

    jobIsRegression = (forceClassification || !(eltype(y) <: Number )) ? false : true # we don't need the tertiary operator here, but it is more clear with it...
    (N,D) = size(x)

    masterSeed = rand(rng,100:9999999999999) ## Some RNG have problems with very small seed. Also, the master seed has to be computed _before_ generateParallelRngs
    rngs = generateParallelRngs(rng,Threads.nthreads())

    #for i in 1:nTrees # for easier debugging/profiling...
    Threads.@threads for i in 1:nTrees
        tsrng = rngs[Threads.threadid()] # Thread safe random number generator
        Random.seed!(tsrng,masterSeed+i*10)
        toSample = rand(tsrng, 1:N,N)
        notToSample = setdiff(1:N,toSample)
        bootstrappedx = x[toSample,:] # "boosted is different than "bootstrapped": https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f
        bootstrappedy = y[toSample]
        #controlx = x[notToSample,:]
        #controly = y[notToSample]
        tree = buildTree(bootstrappedx, bootstrappedy; maxDepth = maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, splittingCriterion = splittingCriterion, forceClassification=forceClassification, rng = tsrng)
        #ŷ = predict(tree,controlx)
        trees[i] = tree
        notSampledByTree[i] = notToSample
    end

    weights = ones(Float64,nTrees)
    if β > 0
        weights = updateTreesWeights!(Forest{Ty}(trees,jobIsRegression,notSampledByTree,0.0,weights), x, y, β=β, rng=rng)
    end
    oobE = +Inf
    if oob
        oobE = oobError(Forest{Ty}(trees,jobIsRegression,notSampledByTree,0.0,weights),x,y,rng=rng)
    end
    return Forest{Ty}(trees,jobIsRegression,notSampledByTree,oobE,weights)
end

# API V2
function fit!(m::RFModel,x,y::AbstractArray{Ty,1}) where {Ty}

    if m.fitted
        @warn "This model has already been fitted and it doesn't support multiple training. This training will override the previous one(s)"
    end

    # Setting default parameters that depends from the data...
    maxDepth    = m.hpar.maxDepth    == nothing ?  size(x,1) : m.hpar.maxDepth
    maxFeatures = m.hpar.maxFeatures == nothing ?  Int(round(sqrt(size(x,2)))) : m.hpar.maxFeatures
    splittingCriterion = m.hpar.splittingCriterion == nothing ? ( (Ty <: Number && !m.hpar.forceClassification) ? variance : gini) : m.hpar.splittingCriterion
    # Setting schortcuts to other hyperparameters/options....
    minGain             = m.hpar.minGain
    minRecords          = m.hpar.minRecords
    forceClassification = m.hpar.forceClassification
    nTrees              = m.hpar.nTrees
    β                   = m.hpar.beta
    oob                 = m.hpar.oob
    cache               = m.opt.cache
    rng                 = m.opt.rng
    verbosity           = m.opt.verbosity
    
    forest = buildForest(x, y, nTrees; maxDepth = maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, forceClassification=forceClassification, splittingCriterion = splittingCriterion, β=β, oob=false,  rng = rng)
    m.par = forest

    m.cres = cache ? predictSingle.(Ref(forest),eachrow(x),rng=rng) : nothing
    
    if oob
        m.par.oobError = oobError(m.par,x,y;rng = rng) 
    end

    m.fitted = true
    
    m.info[:fitted_records]             = size(x,1)
    m.info[:dimensions]                 = maxFeatures
    m.info[:jobIsRegression]            = m.par.isRegression ? 1 : 0
    m.info[:oobE]                       = m.par.oobError
    depths = vcat([transpose([computeDepths(tree)[1],computeDepths(tree)[2]]) for tree in m.par.trees]...)
    (m.info[:avgAvgDepth],m.info[:avgMmaxDepth]) = mean(depths,dims=1)[1], mean(depths,dims=1)[2]
    return true
end

# ------------------------------------------------------------------------------
# MODEL PREDICTIONS 

# Optionally a weighted mean of tree's prediction is used if the parameter `weights` is given.
"""
predictSingle(forest,x)

Predict the label of a single feature record. See [`predict`](@ref).
"""
function predictSingle(forest::Forest{Ty}, x; rng = Random.GLOBAL_RNG) where {Ty}
    trees   = forest.trees
    weights = forest.weights
    predictions  = predictSingle.(trees,Ref(x),rng=rng)
    if eltype(predictions) <: AbstractDict   # categorical
        #weights = 1 .- treesErrors # back to the accuracy
        return meanDicts(predictions,weights=weights)
    else
        #weights = exp.( - treesErrors)
        return dot(predictions,weights)/sum(weights)
    end
end


"""
  [predict(forest,x)](@id forest_prediction)

Predict the labels of a feature dataset.

For each record of the dataset and each tree of the "forest", recursivelly traverse the tree to find the prediction most opportune for the given record.
If the labels the tree has been trained with are numeric, the prediction is also numeric (the mean of the different trees predictions, in turn the mean of the labels of the training records ended in that leaf node).
If the labels were categorical, the prediction is a dictionary with the probabilities of each item and in such case the probabilities of the different trees are averaged to compose the forest predictions. This is a bit different than most other implementations where the mode instead is reported.

In the first case (numerical predictions) use `meanRelError(ŷ,y)` to assess the mean relative error, in the second case you can use `accuracy(ŷ,y)`.
"""
function predict(forest::Forest{Ty}, x;rng = Random.GLOBAL_RNG) where {Ty}
    predictions = predictSingle.(Ref(forest),eachrow(x),rng=rng)
    return predictions
end

# API V2...
function predict(m::RFModel,x)
    return predictSingle.(Ref(m.par),eachrow(x),rng=m.opt.rng)
end

# ------------------------------------------------------------------------------
# OTHER (MODEL OPTIONAL PARTS, INFO, VISUALISATION,...)

"""
   updateTreesWeights!(forest,x,y;β)

Update the weights of each tree (to use in the prediction of the forest) based on the error of the individual tree computed on the records on which it has not been trained.
As training a forest is expensive, this function can be used to "just" upgrade the trees weights using different betas, without retraining the model.
"""
function updateTreesWeights!(forest::Forest{Ty},x,y;β=50,rng = Random.GLOBAL_RNG) where {Ty}
    trees            = forest.trees
    notSampledByTree = forest.oobData
    jobIsRegression  = forest.isRegression
    weights          = Float64[]
    for (i,tree) in enumerate(trees)
        yoob = y[notSampledByTree[i]]
        if length(yoob) > 0
            ŷ = predict(tree,x[notSampledByTree[i],:],rng=rng)
            if jobIsRegression
                push!(weights,exp(- β*meanRelError(ŷ,yoob)))
            else
                push!(weights,accuracy(ŷ,yoob)*β)
            end
        else  # there has been no data that has not being used for this tree, because by a (rare!) chance all the sampled data for this tree was on a different row
            push!(weights,forest.weights[i])
        end
    end
    forest.weights = weights
    return weights
end

"""
   oobError(forest,x,y;rng)

Comute the Out-Of-Bag error, an estimation of the validation error.

This function is called at time of train the forest if the parameter `oob` is `true`, or can be used later to get the oob error on an already trained forest.
The oob error reported is the mismatching error for classification and the relative mean error for regression. 
"""
function oobError(forest::Forest{Ty},x,y;rng = Random.GLOBAL_RNG) where {Ty}
    trees            = forest.trees
    jobIsRegression  = forest.isRegression
    notSampledByTree = forest.oobData
    weights          = forest.weights
    B                = length(trees)
    N                = size(x,1)

    if jobIsRegression
        ŷ = Array{Float64,1}(undef,N)
    else
        ŷ = Array{Dict{Ty,Float64},1}(undef,N)
    end
    # Rarelly a given n has been visited by al lthe trees of the forest, so there is no trees available to compute the oob error
    # This serves as a mask to remove this n from the computation of the oob error
    nMask = fill(true,N)
    for (n,x) in enumerate(eachrow(x))
        unseenTreesBools  = in.(n,notSampledByTree)
        if sum(unseenTreesBools) == 0 # this particular record has been visited by all trees of the forest
            nMask[n] = false
            continue
        end
        unseenTrees = trees[(1:B)[unseenTreesBools]]
        unseenTreesWeights = weights[(1:B)[unseenTreesBools]]
        ŷi   = predictSingle(Forest{Ty}(unseenTrees,jobIsRegression,forest.oobData,0.0,unseenTreesWeights),x,rng=rng)
        if !jobIsRegression && Ty <: Number # we are in the ugly case where we want integers but we have dict of Strings, need to convert
            ŷi   = Dict(map((k,v) -> parse(Int,k)=>v, keys(ŷi), values(ŷi)))
        end
        ŷ[n] = ŷi
    end
    if jobIsRegression
        return meanRelError(ŷ[nMask],y[nMask],normDim=false,normRec=false)
    else
        return error(ŷ[nMask],y[nMask])
    end
end

function show(io::IO, ::MIME"text/plain", m::RFModel)
    if m.fitted == false
        print(io,"RFModel - A $(m.hpar.nTrees) trees Random Forest model (unfitted)")
    else
        job = m.info[:jobIsRegression] == 1 ? "regressor" : "classifier"
        print(io,"RFModel - A $(m.hpar.nTrees) trees Random Forest $job (fitted on $(m.info[:fitted_records]) records)")
    end
end

function show(io::IO, m::RFModel)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"RFModel - A $(m.hpar.nTrees) trees Random Forest model (unfitted)")
    else
        job = m.info[:jobIsRegression] == 1 ? "regressor" : "classifier"
        println(io,"RFModel - A $(m.hpar.nTrees) trees Random Forest $job (fitted on $(m.info[:fitted_records]) records)")
        println(io,m.info)
    end
end