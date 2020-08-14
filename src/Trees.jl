module Trees

using LinearAlgebra, Random,  Reexport
@reexport using ..Utils

export fitTrees, predictTrees

function fitTrees(X,Y)

end

function predictTrees(X,trees)

end





# based on https://www.youtube.com/watch?v=LDRbO9a6XPU
# https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb

training_data = [
    "Green"  3.0 "Apple";
    "Yellow" 3.0 "Apple";
    "Red"    1.0 "Grape";
    "Red"    1.0 "Grape";
    "Yellow" 3.0 "Lemon";
    "Yellow" missing "Lemon";
]

mutable struct Question
    column
    value
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
function partition(data,question::Question)
    (N,D) = size(data)
    true_rows, false_rows, missing_rows = Array{Any,2}(undef,0,D), Array{Any,2}(undef,0,D), Array{Any,2}(undef,0,D)
    for row in eachrow(data)
        #println(row)
        if(ismissing(row[question.column]))
            missing_rows = vcat(missing_rows,permutedims(row))
        elseif match(question,row)
            true_rows = vcat(true_rows,permutedims(row))
        else
            false_rows = vcat(false_rows,permutedims(row))
        end
    end
    # Assigning missing rows randomly proportionally to non-missing rows
    p = size(true_rows,1)/(size(true_rows,1)+size(false_rows,1))
    r = rand(size(missing_rows,1))
    for (ridx,row) in enumerate(eachrow(missing_rows))
        if r[ridx] <= p
            true_rows = vcat(true_rows,permutedims(row))
        else
            false_rows = vcat(false_rows,permutedims(row))
        end
    end
    return true_rows, false_rows
end

"""Counts the number of each type of example in a dataset."""
function class_counts(data)
    counts = Dict{Any,Int64}()  # a dictionary of label -> count.
    for row in eachrow(data)
        # in our dataset format, the label is always the last column
        label = row[end]
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
    counts = class_counts(rows)
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
function info_gain(left, right, current_uncertainty)
    p = size(left,1) / (size(left,1) + size(right,1))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
end

"""Find the best question to ask by iterating over every feature / value
and calculating the information gain."""
function find_best_split(data)

    best_gain           = 0  # keep track of the best information gain
    best_question       = nothing  # keep train of the feature / value that produced it
    current_uncertainty = gini(data)
    D                   = size(data,2)-1  # number of columns (the last column is the label)

    for d in 1:D  # for each feature
        values = Set(skipmissing(data[:,d]))  # unique values in the column
        for val in values  # for each value
            question = Question(d, val)
            # try splitting the dataset
            println(question)
            true_rows, false_rows = partition(data, question)
            # Skip this split if it doesn't divide the
            # dataset.
            if size(true_rows,1) == 0 || size(false_rows,1) == 0
                continue
            end
            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)
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

q1 = Question(1,"Green")
q2 = Question(2,3)

p1,p2 = partition(training_data,q1)

info_gain(p1,p2,gini(training_data))

p1,p2 = partition(training_data,q2)

a = ["a","b","a","c","d"]
c = class_counts(a)
gini([1 2 3; 2 3 3; 4 3 3])
gini([1 2 3; 2 3 1; 4 3 2; 2 3 4; 1 2 5])
gini(training_data)

find_best_split(training_data)







end
