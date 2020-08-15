using Test
using DelimitedFiles, LinearAlgebra

import Random:seed!
seed!(123)

using BetaML.Trees


println("*** Testing Decision trees/Random Forest algorithms...")

# ==================================
# NEW TEST
# ==================================
println("Testing basic classification of decision trees...")
# ---------------------
training_data = [
    "Green"  3.0 "Apple";
    "Yellow" 3.0 "Apple";
    "Red"    1.0 "Grape";
    "Red"    1.0 "Grape";
    "Yellow" 3.0 "Lemon";
]

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

my_tree = build_tree(training_data)

print_tree(tree)

classify(training_data[1,:],my_tree)

print_leaf(classify(training_data[1,:], my_tree))

# Evaluate
testing_data = [
    "Green" 3 "Apple";
    "Yellow" 4 "Apple";
    "Red" 2 "Grape";
    "Red" 1 "Grape";
    "Yellow" 3 "Lemon"
]

for row in eachrow(testing_data)
    println("Actual: $(row[end]) Predicted: $(print_leaf(classify(row, my_tree)))")
end


# ==================================
# NEW TEST
# ==================================

println("Testing classification of the sepal database using decision trees...")
iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),",",skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = map(x->Dict("setosa" => 1, "versicolor" => 2, "virginica" =>3)[x],iris[:, 5])
y_oh = oneHotEncoder(y)

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
ytrain_oh = y_oh[1:ntrain,:]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]
