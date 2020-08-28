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

Both Decision Trees and Random Forests can be used for regression or classification problems, based on the type of the labels (numerical or not). You can override the automatic selection with the parameter `forceClassification=true`, typically if your labels are integer representing some categories rather than numbers.

Missing data on features are supported.

Based originally on the [Josh Gordon's code](https://www.youtube.com/watch?v=LDRbO9a6XPU)

The module provide the following functions. Use `?[type or function]` to access their full signature and detailed documentation:

# Model definition and training:

- `buildTree`: Build a single Decision Tree
- `buildForest`: Build a "forest" of Decision Trees


# Model predictions and assessment:

- `predict`: Return the prediction given the feature matrix
- `Utils.accuracy(tree or forest)`: Categorical output accuracy
- `Utils.mearRelError(tree or forest,p)`: L-p norm based error

Features are expected to be in the standard format (nRecords × nDimensions matrices) and the labels (either categorical or numerical) as a nRecords column vector.
"""
module Trees

using LinearAlgebra, Random, Statistics, Reexport
@reexport using ..Utils

export buildTree, buildForest, computeTreesWeights, oobEstimation, predictSingle, predict, print
import Base.print

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

function print(question::Question)
    condition = "=="
    if isa(question.value, Number)
        condition = ">="
    end
    print("Is col $(question.column) $condition $(question.value) ?")
end

abstract type AbstractNode end
abstract type AbstractDecisionNode <: AbstractNode end
#abstract type AbstractDecisionNodeTy{Ty} <: AbstractDecisionNode end
abstract type AbstractLeaf <: AbstractNode end


struct Tree{Ty}
    nodes::Array{AbstractNode,1}
    type::DataType
end

"""
   Leaf(y,depth)

A tree's leaf (terminal) node.

# Constructor's arguments:
- `y`: The labels assorciated to each record (either numerical or categorical)
- `depth`: The nodes's depth in the tree

# Struct members:
- `rawPredictions`: Either the label's count or the numerical labels of the members of the node
- `predictions`: Either the relative label's count (i.e. a PMF) or the mean
- `depth`: The nodes's depth in the tree
"""
struct Leaf{Ty} <: AbstractLeaf
    predictions::Union{Number,Dict{Ty,Float64}}
    depth::Int64
    function Leaf(y::Array{Ty,1},rIds,depth::Int64) where {Ty}
        thisNodeY = @view y[rIds]
        if eltype(y) <: Number
            rawPredictions = thisNodeY
            predictions    = mean(rawPredictions)
        else
            rawPredictions = classCounts(thisNodeY)
            total = sum(values(rawPredictions))
            predictions = Dict{Ty,Float64}()
            [predictions[k] = rawPredictions[k] / total for k in keys(rawPredictions)]
        end
        new{Ty}(predictions,depth)
    end
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
    trueNodeId::Int64
    falseNodeId::Int64
    depth::Int64
    function DecisionNode(question::Question{Tx},trueNodeId,falseNodeId,depth) where {Tx}
        return new{Tx}(question,trueNodeId,falseNodeId,depth)
    end
end


"""

   match(question, x)

Return a dicotomic answer of a question when applied to a given feature record.

It compares the feature value in the given record to the value stored in the
question.
Numerical features are compared in terms of disequality (">="), while categorical features are compared in terms of equality ("==").
"""
function match(question::Question{Tx}, val::Tx) where {Tx}
    if Tx <: Number
    #if isa(val, Number) # or isa(val, AbstractFloat) to consider "numeric" only floats
        return val >= question.value
    else
        return val == question.value
    end
end

"""
   partition(question,x,xids)

Dicotomically partitions a dataset `x` given a question.

For each of the rows xids in the dataset x, check if it matches the question. If so, add the row id to 'true rows', otherwise, add it to 'false rows'.
Rows with missing values on the question column are assigned randomply proportionally to the assignment of the non-missing rows.
"""
function partition(question::Question{Tx},x,rIds) where {Tx}
    N = size(x,1)

    trueRIds = Int64[]; falseRIds = Int64[]; missingRIds = Int64[]
    for rid in rIds
        value = x[rid,question.column]
        if(ismissing(value))
            push!(missingRIds,rid)
        elseif match(question,value)
            push!(trueRIds,rid)
        else
            push!(falseRIds,rid)
        end
    end
    # Assigning missing rows randomly proportionally to non-missing rows
    p = length(trueRIds)/(length(trueRIds)+length(falseRIds))

    for mRId in missingRIds
        r = rand()
        if r <= p
            push!(trueRIds,mRId)
        else
            push!(falseRIds,mRId)
        end
    end
    return trueRIds, falseRIds
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

"""
function findBestSplit(x,y::AbstractArray{Ty,1},rIds;maxFeatures,splittingCriterion=gini) where {Ty}
    bestGain           = 0  # keep track of the best information gain
    bestQuestion       = nothing  # keep train of the feature / value that produced it
    currentUncertainty = splittingCriterion(y[rIds])
    D  = size(x,2)  # number of columns (the last column is the label)

    for d in shuffle(1:D)[1:maxFeatures]      # for each feature (we consider only maxFeatures features randomly)
        values = Set(skipmissing(@view x[rIds,d]))  # unique values in the column
        for val in values  # for each value
            question = Question(d, val)
            # try splitting the dataset
            #println(question)
            trueRIds, falseRIds = partition(question,x,rIds)
            # Skip this split if it doesn't divide the
            # dataset.
            if length(trueRIds) == 0 || length(falseRIds) == 0
                continue
            end
            # Calculate the information gain from this split
            gain = infoGain(y[trueRIds], y[falseRIds], currentUncertainty, splittingCriterion=splittingCriterion)
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
- `depth`: The current tree's depth. Used when calling the function recursively [def: `1`]
- `maxDepth`: The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `N`, i.e. no limits]
- `minGain`: The minimum information gain to allow for a node's partition [def: `0`]
- `minRecords`:  The minimum number of records a node must holds to consider for a partition of it [def: `2`]
- `maxFeatures`: The maximum number of (random) features to consider at each partitioning [def: `D`, i.e. look at all features]
- `splittingCriterion`: Either `gini`, `entropy` or `variance` (see [`infoGain`](@ref) ) [def: `gini` for categorical labels (classification task) and `variance` for numerical labels(regression task)]
- `forceClassification`: Weather to force a classification task even if the labels are numerical (typically when labels are integers encoding some feature rather than representing a real cardinal measure) [def: `false`]

# Notes:

Missing data (in the feature dataset) are supported.
"""
function buildTree(x, y::Array{Ty,1}, depth=1; maxDepth = size(x,1), minGain=0.0, minRecords=2, maxFeatures=size(x,2), forceClassification=false, splittingCriterion = (Ty <: Number && !forceClassification) ? variance : gini) where {Ty}

    tree = Tree{Ty}(Array{Union{AbstractDecisionNode,Leaf{Ty}},1}[],Ty)
    N    = size(x,1)

    if forceClassification && Ty <: Number
        y = string.(y)
    end

    idsToTreat = Tuple{Array{Int64,1},Int64,Bool}[] # feature ids, parent nodeid, branch type

    currentId = 1
    depth = 1

    # creating the root node
    gain, question = findBestSplit(x,y,1:N;maxFeatures=maxFeatures,splittingCriterion=splittingCriterion)

    if size(x,1) <= minRecords || depth >= maxDepth || gain <= minGain
        push!(tree.nodes, Leaf(y, depth))
    else
        trueRIds, falseRIds = partition(question,x,1:N)
        push!(tree.nodes, DecisionNode(question,0,0,depth))
        push!(idsToTreat,  (trueRIds, length(tree.nodes), true))  # feature ids, parent nodeid, branch type
        push!(idsToTreat,  (falseRIds, length(tree.nodes), false)) # feature ids, parent nodeid, branch type
    end

    while length(idsToTreat) > 0
        workingSet = pop!(idsToTreat)
        thisNodeRIds = workingSet[1]
        #nodex = @view x[workingSet[1],:]
        #nodey = @view y[workingSet[1]]
        thisDepth =  tree.nodes[workingSet[2]].depth + 1

        # Check if this branch has still the minimum number of records required, that we didn't reached the maxDepth allowed and that there is still a gain in splitting. In case, declare it a leaf
        isLeaf = false
        if length(thisNodeRIds) <= minRecords || thisDepth >= maxDepth
            isLeaf = true
        else
            # Try partitioing the dataset on each of the unique attribute,
            # calculate the information gain,
            # and return the question that produces the highest gain.
            gain, question = findBestSplit(x,y,thisNodeRIds;maxFeatures=maxFeatures,splittingCriterion=splittingCriterion)
            if gain <= minGain
                isLeaf = true
            end
        end

        if isLeaf
            push!(tree.nodes, Leaf(y, thisNodeRIds, thisDepth))
        else
            trueRIds, falseRIds = partition(question,x,thisNodeRIds) # get the records ids matching or not the "best possible" question for this node
            push!(tree.nodes, DecisionNode(question,0,0,thisDepth))   # add the node to the array. We don't yeat know the ids of the childs
            push!(idsToTreat,  (trueRIds, length(tree.nodes), true))   # feature ids, parent nodeid, branch type
            push!(idsToTreat,  (falseRIds, length(tree.nodes), false)) # feature ids, parent nodeid, branch type
        end

        # Whatever Leaf or DecisionNode, assign the ids of this node as the child id to the parent
        if workingSet[3] # if it is a "true" branch
            tree.nodes[workingSet[2]].trueNodeId = length(tree.nodes)
        else
            tree.nodes[workingSet[2]].falseNodeId = length(tree.nodes)
        end
    end

    # Return the tree
    return tree
end

#= TODO: need to think on how to print the tree without recursion..
"""
  print(tree)

Print a Decision Tree (textual)

"""
function print(tree::Tree)

    depth     = 1
    fullDepth = rootDepth*string(depth)*"."
    spacing   = ""
    rootDepth = ""
    nId = 1

    while true
        node = tree.nodes[nID]
        depth = node.depth
        if depth  == 1
            println("*** Printing Decision Tree: ***")
        else
            spacing = join(["\t" for i in 1:depth],"")
        end

        # Base case: we've reached a leaf
        if typeof(node) <: Leaf
            println("  $(node.predictions)")
            continue
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
end
=#

"""
predictSingle(tree,x)

Predict the label of a single feature record. See [`predict`](@ref).

"""
function predictSingle(tree::Tree{Ty}, x) where {Ty}

    N = size(x,1)
    nId = 1
    while true
        node = tree.nodes[nId]
        # Base case: we've reached a leaf
        if typeof(node) <: Leaf
            return node.predictions
        end

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if match(node.question,x[node.question.column])
            nId = node.trueNodeId
        else
            nId = node.falseNodeId
        end
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
function predict(tree::Tree{Ty}, x) where {Ty}
    predictions = predictSingle.(Ref(tree),eachrow(x))
    return predictions
end

"""
   buildForest(x, y, nTrees; maxDepth, minGain, minRecords, maxFeatures, splittingCriterion, forceClassification)

Builds (define and train) a "forest" of Decision Trees.


# Parameters:
See [`buildTree`](@ref). The function has all the parameters of `bildTree` (with the `maxFeatures` defaulting to `√D` instead of `D`) plus the following parameters:
- `nTrees`: Number of trees in the forest [def: `30`]
- `β`: Parameter that regulate the weights of the scoring of each tree, to be (optionally) used in prediction (see later) [def: `0`, i.e. uniform weigths]
- `oob`: Wheter to report the out-of-bag error, an estimation of the generalization accuracy [def: `false`]

# Output:
The function returns a named touple with the following elements:
- `forest`: the forest ityself (array of Trees)
- `weights`: the per-tree weight based on their accuracy [def: to array of ones if `β ≤ 0`]
- `oob`:   the estimate of the oob error [def: to `+Inf` if `oob` == `false`]

# Notes :
- Each individual decision tree is built using bootstrap over the data, i.e. "sampling N records with replacement" (hence, some records appear multiple times and some records do not appear in the specific tree training). The `maxFeature` injects further variability and reduces the correlation between the forest trees.
- The predictions of the "forest" (using the function `predict()`) are then the aggregated predictions of the individual trees (from which the name "bagging": **b**oostrap **agg**regat**ing**).
- This function optionally reports a weight distribution of the performances of eanch individual trees, as measured using the records he has not being trained with. These weights can then be (optionally) used in the `predict` function. The parameter `β ≥ 0` regulate the distribution of these weights: larger is `β`, the greater the importance (hence the weights) attached to the best-performing trees compared to the low-performing ones. Using these weights can significantly improve the forest performances (especially using small forests), however the correct value of β depends on the problem under exam (and the chosen caratteristics of the random forest estimator) and should be cross-validated to avoid over-fitting.
- Note that this function uses muiltiple threads if these are available. You can check the number of threads available with `Threads.nthreads()`. To set the number of threads in Julia either set the environmental variable `JULIA_NUM_THREADS` (before starting Julia) or start Julia with the command line option `--threads` (most integrated development editors for Julia already set the number of threads to 4).
"""
function buildForest(x, y::Array{Ty,1}, nTrees=30; maxDepth = size(x,1), minGain=0.0, minRecords=2, maxFeatures=Int(round(sqrt(size(x,2)))), forceClassification=false, splittingCriterion = (Ty <: Number && !forceClassification) ? variance : gini, β=0, oob=false) where {Ty}
    # Force what would be a regression task into a classification task
    if forceClassification && Ty <: Number
        y = string.(y)
    end
    forest           = Array{Tree{Ty},1}(undef,nTrees)
    notSampledByTree = Array{Array{Int64,1},1}(undef,nTrees) # to later compute the Out of Bag Error

    errors = Float64[]

    #jobIsRegression = (forceClassification || !(eltype(y) <: Number ) ? false : true # we don't need the tertiary operator here, but it is more clear with it...
    (N,D) = size(x)

    Threads.@threads for i in 1:nTrees
        toSample = rand(1:N,N)
        notToSample = setdiff(1:N,toSample)
        bootstrappedx = x[toSample,:] # "boosted is different than "bootstrapped": https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f
        bootstrappedy = y[toSample]
        #controlx = x[notToSample,:]
        #controly = y[notToSample]
        tree = buildTree(bootstrappedx, bootstrappedy; maxDepth = maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, splittingCriterion = splittingCriterion, forceClassification=forceClassification)
        #ŷ = predict(tree,controlx)
        forest[i] = tree
        notSampledByTree[i] = notToSample
    end

    weigths = ones(Float64,nTrees)
    if β > 0
        weigths = computeTreesWeights(forest, notSampledByTree, x, y, forceClassification=forceClassification, β=β)
    end
    oobE = +Inf
    if oob
        oobE = oobError(forest,notSampledByTree,x,y,forceClassification = forceClassification)
    end
    return (forest=forest,weights=weigths,oobError=oobE)
end

"""
predictSingle(forest,x;weights)

Predict the label of a single feature record. See [`predict`](@ref).
Optionally a weighted mean of tree's prediction is used if the parameter `weights` is given.

"""
function predictSingle(forest::Array{Tree{Ty},1}, x;weights=ones(length(forest))) where {Ty}
    predictions  = predictSingle.(forest,Ref(x))
    if eltype(predictions) <: AbstractDict   # categorical
        #weights = 1 .- treesErrors # back to the accuracy
        return meanDicts(predictions,weights=weights)
    else
        #weights = exp.( - treesErrors)
        return dot(predictions,weights)/sum(weights)
    end
end

"""
   predict(forest,x;weights)

Predict the labels of a feature dataset.

For each record of the dataset and each tree of the "forest", recursivelly traverse the tree to find the prediction most opportune for the given record.
If the labels the tree has been trained with are numeric, the prediction is also numeric (the mean of the different trees predictions, in turn the mean of the labels of the training records ended in that leaf node).
If the labels were categorical, the prediction is a dictionary with the probabilities of each item and in such case the probabilities of the different trees are averaged to compose the forest predictions. This is a bit different than most other implementations where the mode instead is reported.

In the first case (numerical predictions) use `meanRelError(ŷ,y)` to assess the mean relative error, in the second case you can use `accuracy(ŷ,y)`.

Optionally a weighted mean of tree's prediction is used if the parameter `weights` is given.

"""
function predict(forest::Array{Tree{Ty},1}, x; weights=ones(length(forest))) where {Ty}
    predictions = predictSingle.(Ref(forest),eachrow(x); weights = weights)
    return predictions
end


"""
   computeTreesWeights(forest,notSampledByTree,x,y;forceClassification,β)

Compute the weights of each tree (to use in the prediction of the forest) based on the error of the individual tree computed on the records on which it has not been trained.

"""
function computeTreesWeights(forest::Array{Tree{Ty},1},notSampledByTree::Array{Array{Int64,1},1},x,y;forceClassification = false,β=50) where {Ty}
    weights = Float64[]
    jobIsRegression = (forceClassification || !(Ty <: Number )) ? false : true # we don't need the tertiary operator here, but it is more clear with it...
    for (i,tree) in enumerate(forest)
        yoob = y[notSampledByTree[i]]
        ŷ = predict(tree,x[notSampledByTree[i],:])
        if jobIsRegression
            push!(weights,exp(- β*meanRelError(ŷ,yoob)))
        else
            push!(weights,accuracy(ŷ,yoob)*β)
        end
    end
    return weights
end

function oobError(forest::Array{Tree{Ty},1},notSampledByTree::Array{Array{Int64,1},1},x,y;forceClassification = false) where {Ty}
    jobIsRegression = (forceClassification || !(Ty <: Number )) ? false : true # we don't need the tertiary operator here, but it is more clear with it...
    B = length(forest)
    N = size(x,1)
    if jobIsRegression
        ŷ = Array{Float64,1}(undef,N)
    else
        ŷ = Array{Dict{Ty,Float64},1}(undef,N)
    end

    for (n,x) in enumerate(eachrow(x))
        unseenTrees  = in.(n,notSampledByTree)
        unseenForest = forest[(1:B)[unseenTrees]]
        ŷ[n] = predictSingle(unseenForest,x)
    end
    if jobIsRegression
        return meanRelError(ŷ,y)
    else
        return error(ŷ,y)
    end
end
end
