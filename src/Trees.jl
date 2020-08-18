module Trees

using LinearAlgebra, Random, Statistics, Reexport
@reexport using ..Utils

export buildTree, predict, print

# Decision trees:

# based on https://www.youtube.com/watch?v=LDRbO9a6XPU
# https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
import Base.print

"""A Question is used to partition a dataset.

This class just records a 'column number' (e.g., 0 for Color) and a
'column value' (e.g., Green). The 'match' method is used to compare
the feature value in an example to the feature value stored in the
question. See the demo below.
"""
mutable struct Question
    column
    value
end

function print(question::Question)
    condition = "=="
    if isa(question.value, Number)
        condition = ">="
    end
    print("Is col $(question.column) $condition $(question.value) ?")
end

"""A Leaf node classifies data.

This holds a dictionary of class (e.g., "Apple") -> number of times
it appears in the rows from the training data that reach this leaf.
"""
mutable struct Leaf
    rawPredictions
    predictions
    depth
    function Leaf(y,depth)
        if eltype(y) <: Number
            rawPredictions = y
            predictions     = mean(rawPredictions)
        else
            rawPredictions = classCounts(y)
            total = sum(values(rawPredictions))
            predictions = Dict{eltype(y),Float64}()
            [predictions[k] = rawPredictions[k] / total for k in keys(rawPredictions)]
        end
        return new(rawPredictions,predictions,depth)
    end
end

"""A Decision Node asks a question.

This holds a reference to the question, and to the two child nodes.
"""
mutable struct DecisionNode
    question
    true_branch
    false_branch
    depth
    function DecisionNode(question,true_branch,false_branch, depth)
        return new(question,true_branch,false_branch, depth)
    end
end


"""
Compare the feature value in an example to the
feature value in this question.
"""
function match(question, example)
    val = example[question.column]
    #println(val)
    #println(question.value)
    if isa(val, Number) # or isa(val, AbstractFloat) to consider "numeric" only floats
        return val >= question.value
    else
        return val == question.value
    end
end

"""Partitions a dataset.

For each row in the dataset, check if it matches the question. If
so, add it to 'true rows', otherwise, add it to 'false rows'.

Rows with missing values on the question column are assigned randomply proportionally to the assignment of the non-missing rows.
"""
function partition(question::Question,x)
    N = size(x,1)
    trueIdx = fill(false,N); falseIdx = fill(false,N); missingIdx = fill(false,N)
    for (rIdx,row) in enumerate(eachrow(x))
        #println(row)
        if(ismissing(row[question.column]))
            missingIdx[rIdx] = true
        elseif match(question,row)
            trueIdx[rIdx] = true
        else
            falseIdx[rIdx] = true
        end
    end
    # Assigning missing rows randomly proportionally to non-missing rows
    p = sum(trueIdx)/(sum(trueIdx)+sum(falseIdx))
    r = rand(N)
    for rIdx in 1:N
        if missingIdx[rIdx]
            if r[rIdx] <= p
                trueIdx[rIdx] = true
            else
                falseIdx[rIdx] = true
            end
        end
    end
    return trueIdx, falseIdx
end



"""Information Gain.

The uncertainty of the starting node, minus the weighted impurity of
two child nodes.
"""
function infoGain(left, right, parentUncertainty; splittingCriterion)
    p = size(left,1) / (size(left,1) + size(right,1))
    popvar(x) = var(x,corrected=false)
    critFunction = giniImpurity
    if splittingCriterion == "gini"
        critFunction = giniImpurity
    elseif splittingCriterion == "entropy"
        critFunction = entropy
    elseif splittingCriterion == "variance"
        critFunction = popvar
    else
        @error "Splitting criterion not supported"
    end
    return parentUncertainty - p * critFunction(left) - (1 - p) * critFunction(right)
end

"""Find the best question to ask by iterating over every feature / value
and calculating the information gain."""
function findBestSplit(x,y;maxFeatures,splittingCriterion)

    bestGain           = 0  # keep track of the best information gain
    bestQuestion       = nothing  # keep train of the feature / value that produced it
    if splittingCriterion == "gini"
        currentUncertainty = giniImpurity(y)
    elseif splittingCriterion == "entropy"
        currentUncertainty = entropy(y)
    elseif splittingCriterion == "variance"
        currentUncertainty = var(y,corrected=false)
    else
        @error "Splitting criterion not defined"
    end
    D  = size(x,2)  # number of columns (the last column is the label)

    for d in shuffle(1:D)[1:maxFeatures]      # for each feature (we consider only maxFeatures features randomly)
        values = Set(skipmissing(x[:,d]))  # unique values in the column
        for val in values  # for each value
            question = Question(d, val)
            # try splitting the dataset
            #println(question)
            trueIdx, falseIdx = partition(question,x)
            # Skip this split if it doesn't divide the
            # dataset.
            if sum(trueIdx) == 0 || sum(falseIdx) == 0
                continue
            end
            # Calculate the information gain from this split
            gain = infoGain(y[trueIdx], y[falseIdx], currentUncertainty, splittingCriterion=splittingCriterion)
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

"""Builds the tree.

Rules of recursion: 1) Believe that it works. 2) Start by checking
for the base case (no further information gain). 3) Prepare for
giant stack traces.

criterion{“gini”, “entropy”,"mse"}

"""
function buildTree(x, y, depth=1; maxDepth = size(x,1), minGain=0.0, minRecords=2, maxFeatures=size(x,2), splittingCriterion = eltype(y) <: Number ? "variance" : "gini", forceClassification=false)

    # Force what would be a regression task into a classification task
    if forceClassification && eltype(y) <: Number
        y = string.(y)
    end

    # Check if this branch has still the minimum number of records required and we are reached the maxDepth allowed. In case, declare it a leaf
    if size(x,1) <= minRecords || depth >= maxDepth return Leaf(y, depth) end

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = findBestSplit(x,y;maxFeatures=maxFeatures,splittingCriterion=splittingCriterion)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain <= minGain  return Leaf(y, depth)  end

    # If we reach here, we have found a useful feature / value
    # to partition on.
    trueIdx, falseIdx = partition(question,x)

    # Recursively build the true branch.
    true_branch = buildTree(x[trueIdx,:], y[trueIdx], depth+1, maxDepth=maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, splittingCriterion=splittingCriterion, forceClassification=forceClassification)

    # Recursively build the false branch.
    false_branch = buildTree(x[falseIdx,:], y[falseIdx], depth+1, maxDepth=maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, splittingCriterion=splittingCriterion, forceClassification=forceClassification)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return DecisionNode(question, true_branch, false_branch, depth)
end


"""World's most elegant tree printing function."""
function print(node::Union{Leaf,DecisionNode}, rootDepth="")

    depth     = node.depth
    fullDepth = rootDepth*string(depth)*"."
    spacing   = ""
    if depth  == 1
        println("*** Printing Decision Tree: ***")
    else
        spacing = join(["\t" for i in 1:depth],"")
    end

    # Base case: we've reached a leaf
    if typeof(node) == Leaf
        println("  $(node.predictions)")
        return
    end

    # Print the question at this node
    print("\n$spacing$fullDepth ")
    print(node.question)
    print("\n")

    # Call this function recursively on the true branch
    print(spacing * "--> True :")
    print(node.true_branch, fullDepth)

    # Call this function recursively on the false branch
    print(spacing * "--> False:")
    print(node.false_branch, fullDepth)
end


"""See the 'rules of recursion' above."""
function predictSingle(node::Union{DecisionNode,Leaf}, x)
    # Base case: we've reached a leaf
    if typeof(node) == Leaf
        return node.predictions
    end
    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if match(node.question,x)
        return predictSingle(node.true_branch,x)
    else
        return predictSingle(node.false_branch,x)
    end
end

function predict(tree::Union{DecisionNode, Leaf}, x)
    predictions = predictSingle.(Ref(tree),eachrow(x))
end



end
