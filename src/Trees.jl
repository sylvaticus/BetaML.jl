module Trees

using LinearAlgebra, Random,  Reexport
@reexport using ..Utils

export buildTree, predict, printPredictions, print


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
    function Leaf(y)
        T = eltype(y)
        if T <: Number
            rawPredictions = y
            prediction     = mean(rawPredictions)
        else
            rawPredictions = classCounts(y)
            total = sum(values(rawPredictions))
            predictions = Dict{T,Float64}()
            [predictions[k] = rawPredictions[k] / total for k in keys(rawPredictions)]
        end
        return new(rawPredictions,predictions)
    end
end

"""A Decision Node asks a question.

This holds a reference to the question, and to the two child nodes.
"""
mutable struct DecisionNode
    question
    true_branch
    false_branch
    function DecisionNode(question,true_branch,false_branch)
        return new(question,true_branch,false_branch)
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

"""Counts the number of each type of example in a dataset."""
function classCounts(y)
    T = eltype(y)
    counts = Dict{T,Int64}()  # a dictionary of label -> count.
    for label in y
        if !(label in keys(counts))
            counts[label] = 1
        else
            counts[label] += 1
        end
    end
    return counts
end

"""Calculate the Gini Impurity for a list of rows.

There are a few different ways to do this, I thought this one was
the most concise. See:
https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
"""
function gini(rows)
    counts = classCounts(rows)
    impurity = 1
    for lbl in keys(counts)
        prob_of_lbl = counts[lbl] / float(size(rows,1))
        impurity -= prob_of_lbl^2
    end
    return impurity
end

"""Information Gain.

The uncertainty of the starting node, minus the weighted impurity of
two child nodes.
"""
function infoGain(left, right, current_uncertainty)
    p = size(left,1) / (size(left,1) + size(right,1))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
end

"""Find the best question to ask by iterating over every feature / value
and calculating the information gain."""
function findBestSplit(x,y;maxFeatures,splittingCriterion)

    best_gain           = 0  # keep track of the best information gain
    best_question       = nothing  # keep train of the feature / value that produced it
    current_uncertainty = gini(y)
    D                   = size(x,2)  # number of columns (the last column is the label)

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
            gain = infoGain(y[trueIdx], y[falseIdx], current_uncertainty)
            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain
                best_gain, best_question = gain, question
            end
        end
    end
    return best_gain, best_question
end

"""Builds the tree.

Rules of recursion: 1) Believe that it works. 2) Start by checking
for the base case (no further information gain). 3) Prepare for
giant stack traces.

criterion{“gini”, “entropy”,"mse"}

"""
function buildTree(x, y, depth=1; maxDepth = size(x,1), minGain=0.0, minRecords=2, maxFeatures=size(x,2), splittingCriterion="entropy")

    # Check if this branch has still the minimum number of records required and we are reached the maxDepth allowed. In case, declare it a leaf
    if size(x,1) <= minRecords || depth >= maxDepth return Leaf(y) end

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = findBestSplit(x,y;maxFeatures=maxFeatures,splittingCriterion=splittingCriterion)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain <= minGain  return Leaf(y)  end

    # If we reach here, we have found a useful feature / value
    # to partition on.
    trueIdx, falseIdx = partition(question,x)

    # Recursively build the true branch.

    true_branch = buildTree(x[trueIdx,:], y[trueIdx], depth+1, maxDepth=maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, splittingCriterion=splittingCriterion)

    # Recursively build the false branch.
    false_branch = buildTree(x[falseIdx,:], y[falseIdx], depth+1, maxDepth=maxDepth, minGain=minGain, minRecords=minRecords, maxFeatures=maxFeatures, splittingCriterion=splittingCriterion)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return DecisionNode(question, true_branch, false_branch)
end


"""World's most elegant tree printing function."""
function print(node::Union{Leaf,DecisionNode}, spacing="")

    # Base case: we've reached a leaf
    if typeof(node) == Leaf
        println("$spacing  Predict $(node.predictions)")
        return
    end

    # Print the question at this node
    print(spacing)
    print(node.question)
    print("\n")

    # Call this function recursively on the true branch
    println(spacing * "--> True:")
    print(node.true_branch, spacing * "  ")

    # Call this function recursively on the false branch
    println(spacing * "--> False:")
    print(node.false_branch, spacing * "  ")
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
