
# ------------------------------------------------------------------------------
# TYPE HIERARCHY AND DEFINITIONS

abstract type AbstractNode end
abstract type AbstractDecisionNode <: AbstractNode end
abstract type AbstractLeaf <: AbstractNode end

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
            rawPredictions = classCountsWithLabels(y)
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

# Avi v2..
Base.@kwdef mutable struct DTHyperParametersSet <: BetaMLHyperParametersSet
    maxDepth::Union{Nothing,Int64}              = nothing
    minGain::Float64                            = 0.0
    minRecords::Int64                           = 2
    maxFeatures::Union{Nothing,Int64}           = nothing
    forceClassification::Bool                   = false
    splittingCriterion::Union{Nothing,Function} = nothing
end
#=
Base.@kwdef mutable struct DTOptionsSet <: BetaMLOptionsSet
    rng                  = Random.GLOBAL_RNG
    verbosity::Verbosity = STD
end
=#

Base.@kwdef mutable struct DTLearnableParameters <: BetaMLLearnableParametersSet
    tree::Union{Nothing,AbstractNode} = nothing
end


mutable struct DTModel <: BetaMLSupervisedModel
    hpar::DTHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,DTLearnableParameters}
    cres
    fitted::Bool
    info::Dict{Symbol,Any}
end

function DTModel(;kwargs...)
    m              = DTModel(DTHyperParametersSet(),BetaMLDefaultOptionsSet(),DTLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
          end
        end
    end
    return m
end

# ------------------------------------------------------------------------------
# MODEL ALGORITHMS AND TRAINING
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

# API V2
function fit!(m::DTModel,x,y::AbstractArray{Ty,1}) where {Ty}

    if m.fitted
        @warn "This model has already been fitted (trained) and it doesn't support multiple fitting. This fitting will override the previous one(s)"
    end

    # Setting default parameters that depends from the data...
    maxDepth    = m.hpar.maxDepth    == nothing ?  size(x,1) : m.hpar.maxDepth
    maxFeatures = m.hpar.maxFeatures == nothing ?  size(x,2) : m.hpar.maxFeatures
    splittingCriterion = m.hpar.splittingCriterion == nothing ? ( (Ty <: Number && !m.hpar.forceClassification) ? variance : gini) : m.hpar.splittingCriterion
    # Setting schortcuts to other hyperparameters/options....
    minGain             = m.hpar.minGain
    minRecords          = m.hpar.minRecords
    forceClassification = m.hpar.forceClassification
    cache               = m.opt.cache
    rng                 = m.opt.rng
    verbosity           = m.opt.verbosity

    tree = buildTree(x, y; maxDepth = maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, forceClassification=forceClassification, splittingCriterion = splittingCriterion, mCols=nothing, rng = rng)

    m.par = DTLearnableParameters(tree)
    m.cres = cache ? predictSingle.(Ref(tree),eachrow(x),rng=rng) : nothing

    m.fitted = true

    jobIsRegression = (forceClassification || ! (Ty <: Number) ) ? false : true
    
    m.info[:fittedRecords]             = size(x,1)
    m.info[:dimensions]                 = size(x,2)
    m.info[:jobIsRegression]            = jobIsRegression ? 1 : 0
    (m.info[:avgDepth],m.info[:maxDepth]) = computeDepths(m.par.tree)
    return true
end

# ------------------------------------------------------------------------------
# MODEL PREDICTIONS 

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
If the labels the tree has been fitted with are numeric, the prediction is also numeric.
If the labels were categorical, the prediction is a dictionary with the probabilities of each item.

In the first case (numerical predictions) use `meanRelError(ŷ,y)` to assess the mean relative error, in the second case you can use `accuracy(ŷ,y)`.
"""
function predict(tree::Union{DecisionNode{Tx}, Leaf{Ty}}, x; rng = Random.GLOBAL_RNG) where {Tx,Ty}
    predictions = predictSingle.(Ref(tree),eachrow(x),rng=rng)
    return predictions
end

# API V2...
function predict(m::DTModel,x)
    return predictSingle.(Ref(m.par.tree),eachrow(x),rng=m.opt.rng)
end

# ------------------------------------------------------------------------------
# OTHER (MODEL OPTIONAL PARTS, INFO, VISUALISATION,...)

function computeDepths(node::AbstractNode)
    leafDepths = Int64[]
    nodeQueue = AbstractNode[]
    push!(nodeQueue,node)
    while length(nodeQueue) > 0
      thisNode = pop!(nodeQueue)
      if(typeof(thisNode)  <: AbstractLeaf )
        push!(leafDepths, thisNode.depth)
      else
        push!(nodeQueue, thisNode.trueBranch)
        push!(nodeQueue, thisNode.falseBranch)
      end
    end
    return (mean(leafDepths),maximum(leafDepths))
end

function show(io::IO,question::Question)
    condition = "=="
    if isa(question.value, Number)
        condition = ">="
    end
    print(io, "Is col $(question.column) $condition $(question.value) ?")
end

"""
  print(node)

Print a Decision Tree (textual)

"""
function _printNode(node::AbstractNode, rootDepth="")

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
    _printNode(node.trueBranch, fullDepth)

    # Call this function recursively on the false branch
    print(spacing * "--> False:")
    _printNode(node.falseBranch, fullDepth)
end


function show(io::IO, ::MIME"text/plain", m::DTModel)
    if m.fitted == false
        print(io,"DTModel - A Decision Tree model (unfitted)")
    else
        job = m.info[:jobIsRegression] == 1 ? "regressor" : "classifier"
        print(io,"DTModel - A Decision Tree $job (fitted on $(m.info[:fittedRecords]) records)")
    end
end

function show(io::IO, m::DTModel)
    if m.fitted == false
        print(io,"DTModel - A Decision Tree model (unfitted)")
    else
        job = m.info[:jobIsRegression] == 1 ? "regressor" : "classifier"
        println(io,"DTModel - A Decision Tree $job (fitted on $(m.info[:fittedRecords]) records)")
        println(io,m.info)
        _printNode(m.par.tree)
    end
end


include("AbstractTrees.jl")