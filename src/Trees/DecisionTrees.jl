"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

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
            rawPredictions = class_counts_with_labels(y)
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
"""

$(TYPEDEF)

Hyperparameters for [`DecisionTreeEstimator`](@ref) (Decision Tree).

## Parameters:
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct DTHyperParametersSet <: BetaMLHyperParametersSet
    "The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `nothing`, i.e. no limits]"
    max_depth::Union{Nothing,Int64}              = nothing
    "The minimum information gain to allow for a node's partition [def: `0`]"
    min_gain::Float64                            = 0.0
    "The minimum number of records a node must holds to consider for a partition of it [def: `2`]"
    min_records::Int64                           = 2
    "The maximum number of (random) features to consider at each partitioning [def: `nothing`, i.e. look at all features]"
    max_features::Union{Nothing,Int64}           = nothing
    "Whether to force a classification task even if the labels are numerical (typically when labels are integers encoding some feature rather than representing a real cardinal measure) [def: `false`]"
    force_classification::Bool                   = false
    "This is the name of the function to be used to compute the information gain of a specific partition. This is done by measuring the difference betwwen the \"impurity\" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items. [def: `nothing`, i.e. `gini` for categorical labels (classification task) and `variance` for numerical labels(regression task)]. Either `gini`, `entropy`, `variance` or a custom function. It can also be an anonymous function."
    splitting_criterion::Union{Nothing,Function} = nothing
    """
    The method - and its parameters - to employ for hyperparameters autotuning.
    See [`SuccessiveHalvingSearch`](@ref) for the default method.
    To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and eventually change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
    """
    tunemethod::AutoTuneMethod                  = SuccessiveHalvingSearch(hpranges=Dict("max_depth" =>[5,10,nothing], "min_gain"=>[0.0, 0.1, 0.5], "min_records"=>[2,3,5],"max_features"=>[nothing,5,10,30]),multithreads=true)
end


Base.@kwdef mutable struct DTLearnableParameters <: BetaMLLearnableParametersSet
    tree::Union{Nothing,AbstractNode} = nothing
end

"""
$(TYPEDEF)

A Decision Tree classifier and regressor (supervised).

Decision Tree works by finding the "best" question to split the fitting data (according to the metric specified by the parameter `splitting_criterion` on the associated labels) untill either all the dataset is separated or a terminal condition is reached. 

For the parameters see [`?DTHyperParametersSet`](@ref DTHyperParametersSet) and [`?BetaMLDefaultOptionsSet`](@ref BetaMLDefaultOptionsSet).

# Notes:
- Online fitting (re-fitting with new data) is not supported
- Missing data (in the feature dataset) is supported.

"""
mutable struct DecisionTreeEstimator <: BetaMLSupervisedModel
    hpar::DTHyperParametersSet
    opt::BetaMLDefaultOptionsSet
    par::Union{Nothing,DTLearnableParameters}
    cres
    fitted::Bool
    info::Dict{Symbol,Any}
end

function DecisionTreeEstimator(;kwargs...)
    m              = DecisionTreeEstimator(DTHyperParametersSet(),BetaMLDefaultOptionsSet(),DTLearnableParameters(),nothing,false,Dict{Symbol,Any}())
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

   infoGain(left, right, parentUncertainty; splitting_criterion)

Compute the information gain of a specific partition.

Compare the "information gain" my measuring the difference betwwen the "impurity" of the labels of the parent node with those of the two child nodes, weighted by the respective number of items.

# Parameters:
- `leftY`:  Child #1 labels
- `rightY`: Child #2 labels
- `parentUncertainty`: "Impurity" of the labels of the parent node
- `splitting_criterion`: Metric to adopt to determine the "impurity" (see below)

You can use your own function as the metric. We provide the following built-in metrics:
- `gini` (categorical)
- `entropy` (categorical)
- `variance` (numerical)

"""
function infoGain(leftY, rightY, parentUncertainty; splitting_criterion=gini)
    p = size(leftY,1) / (size(leftY,1) + size(rightY,1))
    return parentUncertainty - p * splitting_criterion(leftY) - (1 - p) * splitting_criterion(rightY)
end

"""
   findBestSplit(x,y;max_features,splitting_criterion)

Find the best possible split of the database.

Find the best question to ask by iterating over every feature / value and calculating the information gain.

# Parameters:
- `x`: The feature dataset
- `y`: The labels dataset
- `max_features`: Maximum number of (random) features to look up for the "best split"
- `splitting_criterion`: The metric to define the "impurity" of the labels
- `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

"""
function findBestSplit(x,y::AbstractArray{Ty,1}, mCols;max_features,splitting_criterion=gini,rng = Random.GLOBAL_RNG) where {Ty}
    bestGain           = 0.0  # keep track of the best information gain
    bestQuestion       = Question(1,1.0) # keep train of the feature / value that produced it
    currentUncertainty = splitting_criterion(y)
    (N,D)  = size(x)  # number of columns (the last column is the label)

    featuresToConsider = (max_features >= D) ? (1:D) : shuffle(rng, 1:D)[1:max_features]

    for d in featuresToConsider      # for each feature (we consider only max_features features randomly)
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
            gain = infoGain(sortedy[trueIdx], sortedy[map(!,trueIdx)], currentUncertainty, splitting_criterion=splitting_criterion)
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

   buildTree(x, y, depth; max_depth, min_gain, min_records, max_features, splitting_criterion, force_classification)

Builds (define and train) a Decision Tree.

!!! warning
    This function is deprecated and will possibly be removed in BetaML 0.9.
    Use [`DecisionTreeEstimator`](@ref) instead. 

Given a dataset of features `x` and the corresponding dataset of labels `y`, recursivelly build a decision tree by finding at each node the best question to split the data untill either all the dataset is separated or a terminal condition is reached.
The given tree is then returned.

# Parameters:
- `x`: The dataset's features (N × D)
- `y`: The dataset's labels (N × 1)
- `max_depth`: The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `N`, i.e. no limits]
- `min_gain`: The minimum information gain to allow for a node's partition [def: `0`]
- `min_records`:  The minimum number of records a node must holds to consider for a partition of it [def: `2`]
- `max_features`: The maximum number of (random) features to consider at each partitioning [def: `D`, i.e. look at all features]
- `splitting_criterion`: Either `gini`, `entropy` or `variance`[def: `gini` for categorical labels (classification task) and `variance` for numerical labels(regression task)]
- `force_classification`: Whether to force a classification task even if the labels are numerical (typically when labels are integers encoding some feature rather than representing a real cardinal measure) [def: `false`]
- `rng`: Random Number Generator ((see [`FIXEDSEED`](@ref))) [deafult: `Random.GLOBAL_RNG`]

# Notes:

Missing data (in the feature dataset) are supported.
"""
function buildTree(x, y::AbstractArray{Ty,1}; max_depth = size(x,1), min_gain=0.0, min_records=2, max_features=size(x,2), force_classification=false, splitting_criterion = (Ty <: Number && !force_classification) ? variance : gini, mCols=nothing, rng = Random.GLOBAL_RNG) where {Ty}


    #println(depth)
    # Force what would be a regression task into a classification task
    if force_classification && Ty <: Number
        y = string.(y)
    end

    if(mCols == nothing) mCols = cols_with_missing(x) end


    nodes = TempNode[]
    depth = 1

    # Deciding if the root node is a Leaf itself or not

    # Check if this branch has still the minimum number of records required and we are reached the max_depth allowed. In case, declare it a leaf
    if size(x,1) <= min_records || depth >= max_depth return Leaf(y, depth) end

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = findBestSplit(x,y,mCols;max_features=max_features,splitting_criterion=splitting_criterion,rng=rng)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain <= min_gain  return Leaf(y, depth)  end

    trueIdx  = partition(question,x,mCols,rng=rng)
    rootNode = DecisionNode(question,nothing,nothing,1,sum(trueIdx)/length(trueIdx))

    push!(nodes,TempNode(true,rootNode,depth+1,x[trueIdx,:],y[trueIdx]))
    push!(nodes,TempNode(false,rootNode,depth+1,x[map(!,trueIdx),:],y[map(!,trueIdx)]))

    while length(nodes) > 0
        thisNode = pop!(nodes)

        # Check if this branch has still the minimum number of records required, that we didn't reached the max_depth allowed and that there is still a gain in splitting. In case, declare it a leaf
        isLeaf = false
        if size(thisNode.x,1) <= min_records || thisNode.depth >= max_depth
            isLeaf = true
        else
            # Try partitioing the dataset on each of the unique attribute,
            # calculate the information gain,
            # and return the question that produces the highest gain.
            gain, question = findBestSplit(thisNode.x,thisNode.y,mCols;max_features=max_features,splitting_criterion=splitting_criterion,rng=rng)
            if gain <= min_gain
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

"""
$(TYPEDSIGNATURES)

Fit a [`DecisionTreeEstimator`](@ref) to the data

"""
function fit!(m::DecisionTreeEstimator,x,y::AbstractArray{Ty,1}) where {Ty}

    if m.fitted
        @warn "This model has already been fitted (trained) and it doesn't support multiple fitting. This fitting will override the previous one(s)"
    else
        autotune!(m,(x,y))
    end

    # Setting default parameters that depends from the data...
    max_depth    = m.hpar.max_depth    == nothing ?  size(x,1) : m.hpar.max_depth
    max_features = m.hpar.max_features == nothing ?  size(x,2) : m.hpar.max_features
    splitting_criterion = m.hpar.splitting_criterion == nothing ? ( (Ty <: Number && !m.hpar.force_classification) ? variance : gini) : m.hpar.splitting_criterion
    # Setting schortcuts to other hyperparameters/options....
    min_gain             = m.hpar.min_gain
    min_records          = m.hpar.min_records
    force_classification = m.hpar.force_classification
    cache               = m.opt.cache
    rng                 = m.opt.rng
    verbosity           = m.opt.verbosity

    tree = buildTree(x, y; max_depth = max_depth, min_gain=min_gain, min_records=min_records, max_features=max_features, force_classification=force_classification, splitting_criterion = splitting_criterion, mCols=nothing, rng = rng)

    m.par = DTLearnableParameters(tree)
    m.cres = cache ? predictSingle.(Ref(tree),eachrow(x),rng=rng) : nothing

    m.fitted = true

    jobIsRegression = (force_classification || ! (Ty <: Number) ) ? false : true
    
    m.info[:fitted_records]             = size(x,1)
    m.info[:dimensions]                 = size(x,2)
    m.info[:jobIsRegression]            = jobIsRegression ? 1 : 0
    (m.info[:avgDepth],m.info[:max_depth]) = computeDepths(m.par.tree)
    return cache ? m.cres : nothing
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

!!! warning
    This function is deprecated and will possibly be removed in BetaML 0.9.
    Use [`DecisionTreeEstimator`](@ref) and the associated `predict(m::Model,x)` function instead.

For each record of the dataset, recursivelly traverse the tree to find the prediction most opportune for the given record.
If the labels the tree has been fitted with are numeric, the prediction is also numeric.
If the labels were categorical, the prediction is a dictionary with the probabilities of each item.

In the first case (numerical predictions) use `relative_mean_error(ŷ,y)` to assess the mean relative error, in the second case you can use `accuracy(ŷ,y)`.
"""
function predict(tree::Union{DecisionNode{Tx}, Leaf{Ty}}, x; rng = Random.GLOBAL_RNG) where {Tx,Ty}
    predictions = predictSingle.(Ref(tree),eachrow(x),rng=rng)
    return predictions
end

# API V2...
"""
$(TYPEDSIGNATURES)

Predict the labels associated to some feature data using a trained [`DecisionTreeEstimator`](@ref)

"""
function predict(m::DecisionTreeEstimator,x)
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


function show(io::IO, ::MIME"text/plain", m::DecisionTreeEstimator)
    if m.fitted == false
        print(io,"DecisionTreeEstimator - A Decision Tree model (unfitted)")
    else
        job = m.info[:jobIsRegression] == 1 ? "regressor" : "classifier"
        print(io,"DecisionTreeEstimator - A Decision Tree $job (fitted on $(m.info[:fitted_records]) records)")
    end
end

function show(io::IO, m::DecisionTreeEstimator)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        print(io,"DecisionTreeEstimator - A Decision Tree model (unfitted)")
    else
        job = m.info[:jobIsRegression] == 1 ? "regressor" : "classifier"
        println(io,"DecisionTreeEstimator - A Decision Tree $job (fitted on $(m.info[:fitted_records]) records)")
        println(io,m.info)
        _printNode(m.par.tree)
    end
end


include("AbstractTrees.jl")