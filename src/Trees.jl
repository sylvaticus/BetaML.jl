"""
  Trees.jl File

Decision Trees implementation (Module BetaML.Trees)

`?BetaML.Trees` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Trees.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    BetaML.Trees module

Implement the functionality required to build a Decision Tree or a whole Random Forest, predict data and assess its performances.

Both Decision Trees and Random Forests can be used for regression or classification problems, based on the type of the labels (numerical or not). You can override the automatic selection with the parameter `forceClassification=true`, typically if your labels are integer representing some categories rather than numbers. For classification problems the output of `predictSingle` is a dictionary with the key being the labels with non-zero probabilitity and the corresponding value its proobability; for regression it is a numerical value.

Please be aware that, differently from most other implementations, the Random Forest algorithm collects and averages the probabilities from the trees, rather than just repording the mode, i.e. no information is lost and the output of the forest classifier is still a PMF.

Missing data on features are supported, both on training and on prediction.

The module provide the following functions. Use `?[type or function]` to access their full signature and detailed documentation:

# Model definition and training:

- `buildTree(xtrain,ytrain)`: Build a single Decision Tree
- `buildForest(xtrain,ytrain)`: Build a "forest" of Decision Trees


# Model predictions and assessment:

- `predict(tree or forest, x)`: Return the prediction given the feature matrix
- `oobError(forest,x,y)`: Return the out-of-bag error estimate
- `Utils.accuracy(ŷ,y))`: Categorical output accuracy
- `Utils.meanRelError(ŷ,y,p)`: L-p norm based error

Features are expected to be in the standard format (nRecords × nDimensions matrices) and the labels (either categorical or numerical) as a nRecords column vector.

Acknowlegdments: originally based on the [Josh Gordon's code](https://www.youtube.com/watch?v=LDRbO9a6XPU)
"""
module Trees

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays

using  ForceImport
@force using ..Api
@force using ..Utils


export buildTree, buildForest, updateTreesWeights!, predictSingle, predict, print
import Base.print

export AbstractDecisionNode,Leaf, DecisionNode, Forest  # cancelalble

"""
   Question

A question used to partition a dataset.

This struct just records a 'column number' and a 'column value' (e.g., Green).
"""
abstract type AbstractQuestion end
struct Question{Tx} <: AbstractQuestion
    column::Int64
    value::Tx
end

abstract type AbstractNode end
abstract type AbstractDecisionNode <: AbstractNode end
abstract type AbstractLeaf <: AbstractNode end

"""
   Leaf(y,depth)

A tree's leaf (terminal) node.

# Constructor's arguments:
- `y`: The labels assorciated to each record (either numerical or categorical)
- `depth`: The nodes's depth in the tree

# Struct members:
- `predictions`: Either the relative label's count (i.e. a PMF) or the mean
- `depth`: The nodes's depth in the tree
"""
struct Leaf{Ty} <: AbstractLeaf
    predictions::Union{Number,Dict{Ty,Float64}}
    depth::Int64
    function Leaf(y::AbstractArray{Ty,1},depth::Int64) where {Ty}
        if eltype(y) <: Number
            rawPredictions = y
            predictions    = mean(rawPredictions)
        else
            rawPredictions = classCounts(y)
            total = sum(values(rawPredictions))
            predictions = Dict{Ty,Float64}()
            [predictions[k] = rawPredictions[k] / total for k in keys(rawPredictions)]
        end
        new{Ty}(predictions,depth)
    end
end

struct TempNode
    trueBranch::Bool
    parentNode::AbstractDecisionNode
    depth::Int64
    x
    y
end

"""

A Decision Node asks a question.

This holds a reference to the question, and to the two child nodes.
"""


"""
   DecisionNode(question,trueBranch,falseBranch, depth)

A tree's non-terminal node.

# Constructor's arguments and struct members:
- `question`: The question asked in this node
- `trueBranch`: A reference to the "true" branch of the trees
- `falseBranch`: A reference to the "false" branch of the trees
- `depth`: The nodes's depth in the tree
"""
mutable struct DecisionNode{Tx} <: AbstractDecisionNode
    # Note that a decision node is indeed type unstable, as it host other decision nodes whose X type could be different (different X features can have different type)
    question::Question{Tx}
    trueBranch::Union{Nothing,AbstractNode}
    falseBranch::Union{Nothing,AbstractNode}
    depth::Int64
    pTrue::Float64
    function DecisionNode(question::Question{Tx},trueBranch::Union{Nothing,AbstractNode},falseBranch::Union{Nothing,AbstractNode}, depth,pTrue) where {Tx}
        return new{Tx}(question,trueBranch,falseBranch, depth,pTrue)
    end
end

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
mutable struct Forest{Ty}
    trees::Array{Union{AbstractDecisionNode,Leaf{Ty}},1}
    isRegression::Bool
    oobData::Array{Array{Int64,1},1}
    oobError::Float64
    weights::Array{Float64,1}
end

function print(question::Question)
    condition = "=="
    if isa(question.value, Number)
        condition = ">="
    end
    print("Is col $(question.column) $condition $(question.value) ?")
end

"""

   match(question, x)

Return a dicotomic answer of a question when applied to a given feature record.

It compares the feature value in the given record to the value stored in the
question.
Numerical features are compared in terms of disequality (">="), while categorical features are compared in terms of equality ("==").
"""
function match(question::Question{Tx}, x) where {Tx}
    val = x[question.column]
    if Tx <: Number
        return val >= question.value
    else
        return val == question.value
    end
end

"""
   partition(question,x)

Dicotomically partitions a dataset `x` given a question.

For each row in the dataset, check if it matches the question. If so, add it to 'true rows', otherwise, add it to 'false rows'.
Rows with missing values on the question column are assigned randomly proportionally to the assignment of the non-missing rows.
"""
function partition(question::Question{Tx},x,mCols;sorted=false,rng = Random.GLOBAL_RNG) where {Tx}
    N = size(x,1)

    trueIdx = fill(false,N);

    if  in(question.column,mCols) # do we have missings in this col ?
        missingIdx = fill(false,N)
        nFalse = 0
        @inbounds for (rIdx,row) in enumerate(eachrow(x))
            if(ismissing(row[question.column]))
                missingIdx[rIdx] = true
            elseif match(question,row)
                trueIdx[rIdx] = true
            else
                nFalse += 1
            end
        end
        # Assigning missing rows randomly proportionally to non-missing rows
        p = sum(trueIdx)/(sum(trueIdx)+nFalse)
        @inbounds for rIdx in 1:N
            if missingIdx[rIdx]
                r = rand(rng)
                if r <= p
                    trueIdx[rIdx] = true
                end
            end
        end
    else
        if sorted
            #val = x[question.column]
            idx = searchsorted(x[:,question.column], question.value)
            if Tx <: Number
                trueIdx[first(idx):end] .= true
            else
                trueIdx[idx] .= true
            end
        else
            @inbounds for (rIdx,row) in enumerate(eachrow(x))
                if match(question,row)
                    trueIdx[rIdx] = true
                end
            end
        end

    end
    return trueIdx
end

"""

   infoGain(left, right, parentUncertainty; splittingCriterion)

Compute the information gain of a specific partition.

Compare the "information gain" my measuring the difference betwwen the "impurity" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items.

# Parameters:
- `leftY`:  Child #1 labels
- `rightY`: Child #2 labels
- `parentUncertainty`: "Impurity" of the labels of the parent node
- `splittingCriterion`: Metric to adopt to determine the "impurity" (see below)

You can use your own function as the metric. We provide the following built-in metrics:
- `gini` (categorical)
- `entropy` (categorical)
- `variance` (numerical)

"""
function infoGain(leftY, rightY, parentUncertainty; splittingCriterion=gini)
    p = size(leftY,1) / (size(leftY,1) + size(rightY,1))
    return parentUncertainty - p * splittingCriterion(leftY) - (1 - p) * splittingCriterion(rightY)
end

"""
   findBestSplit(x,y;maxFeatures,splittingCriterion)

Find the best possible split of the database.

Find the best question to ask by iterating over every feature / value and calculating the information gain.

# Parameters:
- `x`: The feature dataset
- `y`: The labels dataset
- `maxFeatures`: Maximum number of (random) features to look up for the "best split"
- `splittingCriterion`: The metric to define the "impurity" of the labels
- `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

"""
function findBestSplit(x,y::AbstractArray{Ty,1}, mCols;maxFeatures,splittingCriterion=gini,rng = Random.GLOBAL_RNG) where {Ty}
    bestGain           = 0.0  # keep track of the best information gain
    bestQuestion       = Question(1,1.0) # keep train of the feature / value that produced it
    currentUncertainty = splittingCriterion(y)
    (N,D)  = size(x)  # number of columns (the last column is the label)

    featuresToConsider = (maxFeatures >= D) ? (1:D) : shuffle(rng, 1:D)[1:maxFeatures]

    for d in featuresToConsider      # for each feature (we consider only maxFeatures features randomly)
        values = Set(skipmissing(x[:,d]))  # unique values in the column
        sortable = Utils.issortable(x[:,d])
        if(sortable)
            sortIdx = sortperm(x[:,d])
            sortedx = x[sortIdx,:]
            sortedy = y[sortIdx]
        else
            sortIdx = 1:N
            sortedx = x
            sortedy = y
        end

        for val in values  # for each value
            question = Question(d, val)
            # try splitting the dataset
            #println(question)
            trueIdx = partition(question,sortedx,mCols,sorted=sortable,rng=rng)
            # Skip this split if it doesn't divide the
            # dataset.
            if all(trueIdx) || ! any(trueIdx)
                continue
            end
            # Calculate the information gain from this split
            gain = infoGain(sortedy[trueIdx], sortedy[map(!,trueIdx)], currentUncertainty, splittingCriterion=splittingCriterion)
            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= bestGain
                bestGain, bestQuestion = gain, question
            end
        end
    end
    return bestGain, bestQuestion
end


"""

   buildTree(x, y, depth; maxDepth, minGain, minRecords, maxFeatures, splittingCriterion, forceClassification)

Builds (define and train) a Decision Tree.

Given a dataset of features `x` and the corresponding dataset of labels `y`, recursivelly build a decision tree by finding at each node the best question to split the data untill either all the dataset is separated or a terminal condition is reached.
The given tree is then returned.

# Parameters:
- `x`: The dataset's features (N × D)
- `y`: The dataset's labels (N × 1)
- `maxDepth`: The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `N`, i.e. no limits]
- `minGain`: The minimum information gain to allow for a node's partition [def: `0`]
- `minRecords`:  The minimum number of records a node must holds to consider for a partition of it [def: `2`]
- `maxFeatures`: The maximum number of (random) features to consider at each partitioning [def: `D`, i.e. look at all features]
- `splittingCriterion`: Either `gini`, `entropy` or `variance` (see [`infoGain`](@ref) ) [def: `gini` for categorical labels (classification task) and `variance` for numerical labels(regression task)]
- `forceClassification`: Weather to force a classification task even if the labels are numerical (typically when labels are integers encoding some feature rather than representing a real cardinal measure) [def: `false`]
- `rng`: Random Number Generator ((see [`FIXEDSEED`](@ref))) [deafult: `Random.GLOBAL_RNG`]

# Notes:

Missing data (in the feature dataset) are supported.
"""
function buildTree(x, y::AbstractArray{Ty,1}; maxDepth = size(x,1), minGain=0.0, minRecords=2, maxFeatures=size(x,2), forceClassification=false, splittingCriterion = (Ty <: Number && !forceClassification) ? variance : gini, mCols=nothing, rng = Random.GLOBAL_RNG) where {Ty}


    #println(depth)
    # Force what would be a regression task into a classification task
    if forceClassification && Ty <: Number
        y = string.(y)
    end

    if(mCols == nothing) mCols = colsWithMissing(x) end


    nodes = TempNode[]
    depth = 1

    # Deciding if the root node is a Leaf itself or not

    # Check if this branch has still the minimum number of records required and we are reached the maxDepth allowed. In case, declare it a leaf
    if size(x,1) <= minRecords || depth >= maxDepth return Leaf(y, depth) end

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = findBestSplit(x,y,mCols;maxFeatures=maxFeatures,splittingCriterion=splittingCriterion,rng=rng)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain <= minGain  return Leaf(y, depth)  end

    trueIdx  = partition(question,x,mCols,rng=rng)
    rootNode = DecisionNode(question,nothing,nothing,1,sum(trueIdx)/length(trueIdx))

    push!(nodes,TempNode(true,rootNode,depth+1,x[trueIdx,:],y[trueIdx]))
    push!(nodes,TempNode(false,rootNode,depth+1,x[map(!,trueIdx),:],y[map(!,trueIdx)]))

    while length(nodes) > 0
        thisNode = pop!(nodes)

        # Check if this branch has still the minimum number of records required, that we didn't reached the maxDepth allowed and that there is still a gain in splitting. In case, declare it a leaf
        isLeaf = false
        if size(thisNode.x,1) <= minRecords || thisNode.depth >= maxDepth
            isLeaf = true
        else
            # Try partitioing the dataset on each of the unique attribute,
            # calculate the information gain,
            # and return the question that produces the highest gain.
            gain, question = findBestSplit(thisNode.x,thisNode.y,mCols;maxFeatures=maxFeatures,splittingCriterion=splittingCriterion,rng=rng)
            if gain <= minGain
                isLeaf = true
            end
        end
        if isLeaf
            newNode = Leaf(thisNode.y, thisNode.depth)
        else
            trueIdx = partition(question,thisNode.x,mCols,rng=rng)
            newNode = DecisionNode(question,nothing,nothing,thisNode.depth,sum(trueIdx)/length(trueIdx))
            push!(nodes,TempNode(true,newNode,thisNode.depth+1,thisNode.x[trueIdx,:],thisNode.y[trueIdx]))
            push!(nodes,TempNode(false,newNode,thisNode.depth+1,thisNode.x[map(!,trueIdx),:],thisNode.y[map(!,trueIdx)]))
        end
        thisNode.trueBranch ? (thisNode.parentNode.trueBranch = newNode) : (thisNode.parentNode.falseBranch = newNode)
    end

    return rootNode

end


"""
  print(node)

Print a Decision Tree (textual)

"""
function print(node::AbstractNode, rootDepth="")

    depth     = node.depth
    fullDepth = rootDepth*string(depth)*"."
    spacing   = ""
    if depth  == 1
        println("*** Printing Decision Tree: ***")
    else
        spacing = join(["\t" for i in 1:depth],"")
    end

    # Base case: we've reached a leaf
    if typeof(node) <: Leaf
        println("  $(node.predictions)")
        return
    end

    # Print the question at this node
    print("\n$spacing$fullDepth ")
    print(node.question)
    print("\n")

    # Call this function recursively on the true branch
    print(spacing * "--> True :")
    print(node.trueBranch, fullDepth)

    # Call this function recursively on the false branch
    print(spacing * "--> False:")
    print(node.falseBranch, fullDepth)
end


"""
predictSingle(tree,x)

Predict the label of a single feature record. See [`predict`](@ref).

"""
function predictSingle(node::Union{DecisionNode{Tx},Leaf{Ty}}, x;rng = Random.GLOBAL_RNG) where {Tx,Ty}
    # Base case: we've reached a leaf
    if typeof(node) <: Leaf
        return node.predictions
    end
    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.

    # If the feature on which to base prediction is missing, we follow the true branch with a probability equal to the share of true
    # records over all the records during this node training..
    if ismissing(x[node.question.column])
        r = rand(rng)
        return (node.pTrue >= r) ? predictSingle(node.trueBranch,x,rng=rng) : predictSingle(node.falseBranch,x,rng=rng)
    end
    if match(node.question,x)
        return predictSingle(node.trueBranch,x,rng=rng)
    else
        return predictSingle(node.falseBranch,x,rng=rng)
    end
end

"""
   predict(tree,x)

Predict the labels of a feature dataset.

For each record of the dataset, recursivelly traverse the tree to find the prediction most opportune for the given record.
If the labels the tree has been trained with are numeric, the prediction is also numeric.
If the labels were categorical, the prediction is a dictionary with the probabilities of each item.

In the first case (numerical predictions) use `meanRelError(ŷ,y)` to assess the mean relative error, in the second case you can use `accuracy(ŷ,y)`.
"""
function predict(tree::Union{DecisionNode{Tx}, Leaf{Ty}}, x; rng = Random.GLOBAL_RNG) where {Tx,Ty}
    predictions = predictSingle.(Ref(tree),eachrow(x),rng=rng)
    return predictions
end

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
   oobError(forest,x,y)

Comute the Out-Of-Bag error, an estimation of the validation error.

This function is called at time of train the forest if the parameter `oob` is `true`, or can be used later to get the oob error on an already trained forest.
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

    for (n,x) in enumerate(eachrow(x))
        unseenTreesBools  = in.(n,notSampledByTree)
        unseenTrees = trees[(1:B)[unseenTreesBools]]
        unseenTreesWeights = weights[(1:B)[unseenTreesBools]]
        ŷ[n] = predictSingle(Forest{Ty}(unseenTrees,jobIsRegression,forest.oobData,0.0,unseenTreesWeights),x,rng=rng)
    end
    if jobIsRegression
        return meanRelError(ŷ,y)
    else
        return error(ŷ,y)
    end
end

#=
# maybe nice to implement.. when I'll have time..
function tune(model::AbstractNode,xtrain,ytrain,xval,yval,parameters;loss=(ŷ,y)->meanRelError(ŷ,y,normRec=false),repetitions=5,rng=Random.GLOBAL_RNG)
    ## We start with an infinitely high error
    bestError       = +Inf

    compLock        = ReentrantLock()

    ## Generate one random number generator per thread
    masterSeed = rand(rng,100:9999999999999) ## Some RNG have problems with very small seed. Also, the master seed has to be computed _before_ generateParallelRngs
    rngs = generateParallelRngs(rng,Threads.nthreads())

    ## We loop over all possible hyperparameter combinations...
    parLengths = (length(maxDepthRange),length(maxFeaturesRange),length(minRecordsRange),length(nTreesRange),length(βRange))
    Threads.@threads for ij in CartesianIndices(parLengths) ## This to avoid many nested for loops
           (maxDepth,maxFeatures,minRecords,nTrees,β)   = (maxDepthRange[Tuple(ij)[1]], maxFeaturesRange[Tuple(ij)[2]], minRecordsRange[Tuple(ij)[3]], nTreesRange[Tuple(ij)[4]], βRange[Tuple(ij)[5]]) ## The specific hyperparameters of this nested loop
           tsrng = rngs[Threads.threadid()] ## The random number generator is specific for each thread..
           joinedIndx = LinearIndices(parLengths)[ij]
           ## And here we make the seeding depending on the id of the loop, not the thread: hence we get the same results indipendently of the number of threads
           Random.seed!(tsrng,masterSeed+joinedIndx*10)
           totAttemptError = 0.0
           ## We run several repetitions with the same hyperparameter combination to account for stochasticity...
           for r in 1:repetitions
              if model == "DecisionTree"
                 ## Here we train the Decition Tree model
                 myTrainedModel = buildTree(xtrain,ytrain, maxDepth=maxDepth,maxFeatures=maxFeatures,minRecords=minRecords,rng=tsrng)
              else
                 ## Here we train the Random Forest model
                 myTrainedModel = buildForest(xtrain,ytrain,nTrees,maxDepth=maxDepth,maxFeatures=maxFeatures,minRecords=minRecords,β=β,rng=tsrng)
              end
              ## Here we make prediciton with this trained model and we compute its error
              ŷval   = predict(myTrainedModel, xval,rng=tsrng)
              rmeVal = meanRelError(ŷval,yval,normRec=false)
              totAttemptError += rmeVal
           end
           avgAttemptedDepthError = totAttemptError / repetitions
           begin
               lock(compLock) ## This step can't be run in parallel...
               try
                   ## Select this specific combination of hyperparameters if the error is the lowest
                   if avgAttemptedDepthError < bestRme
                     bestRme         = avgAttemptedDepthError
                     bestMaxDepth    = maxDepth
                     bestMaxFeatures = maxFeatures
                     bestNTrees      = nTrees
                     bestβ           = β
                     bestMinRecords  = minRecords
                   end
               finally
                   unlock(compLock)
               end
           end
    end
    return (bestRme,bestMaxDepth,bestMaxFeatures,bestMinRecords,bestNTrees,bestβ)
end
=#





# MLJ interface
include("Trees_MLJ.jl")

end # end module
