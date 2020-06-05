# Bmlt.jl Documentation


Welcome to the documentation of the [_Beta Machine Learning Toolkit_](https://github.com/sylvaticus/Bmlt.jl).


## Installation

This is NOT YET a Julia registered package:
* install it with `] add https://github.com/sylvaticus/Bmlt.jl.git`

## Loading the module(s)

This package is split in several submodules. You can access its functionality either by `using` the specific submodule of interest and then directly the provided functionality (utilities are re-exported by each of the other submodules, so normally you don't need to implicitly import them) or by `using` the root module `Bmlt` and then prefix with `Bmlt.` each object/function you want to use, e.g.:

```
using Bmlt.Nn
myLayer = DenseLayer(2,3)
```

or, equivalently,:

```
using Bmlt
res = Bmlt.kernelPerceptron([1.1 2.1; 5.3 4.2; 1.8 1.7], [-1,1,-1])
```

## Usage

Documentation for most functions can be retrieved using the inline Julia help system (just press the question mark `?` and then, on the special help prompt `help?>`, type the function name).

For documentation, please look at the individual modules:

- [**`Bmlt.Perceptron`**](Perceptron.html): Perform classification tasks using the Perceptron, Kernel Perceptron or Pegasus algorithms
- [**`Bmlt.Nn`**](Nn.html): Implementation of Artificial Neural networks
- [**`Bmlt.Clustering``**](Clustering.html): Clustering algorithms (Kmeans, Mdedoids, GMM) and collaborative filtering /recommandation systems using clusters
- [**`Bmlt.Utils``**](Utils.html): Various utility functions (scale, one-hot, distances, kernels,..)

## Examples

### Using an Artificial Neural Network for multinomial categorisation

```julia
# Load Modules
using Bmlt.Nn, DelimitedFiles, Random, StatsPlots # Load the main module and ausiliary modules
Random.seed!(123); # Fix the random seed (to obtain reproducible results)

# Load the data
iris     = readdlm(joinpath(dirname(Base.find_package("Bmlt")),"..","test","data","iris.csv"),',',skipstart=1)
iris     = iris[shuffle(axes(iris, 1)), :] # Shuffle the records, as they aren't by default
x        = convert(Array{Float64,2}, iris[:,1:4])
y        = map(x->Dict("setosa" => 1, "versicolor" => 2, "virginica" =>3)[x],iris[:, 5]) # Convert the target column to numbers
y_oh     = oneHotEncoder(y) # Convert to One-hot representation (e.g. 2 => [0 1 0], 3 => [0 0 1])

# Split the data in training/testing sets
ntrain    = Int64(round(size(x,1)*0.8))
xtrain    = x[1:ntrain,:]
ytrain    = y[1:ntrain]
ytrain_oh = y_oh[1:ntrain,:]
xtest     = x[ntrain+1:end,:]
ytest     = y[ntrain+1:end]

# Define the Artificial Neural Network model
l1   = DenseLayer(4,10,f=relu) # Activation function is ReLU
l2   = DenseLayer(10,3)        # Activation function is identity by default
l3   = VectorFunctionLayer(3,3,f=softMax) # Add a (parameterless) layer whose activation function (softMax in this case) is defined to all its nodes at once
mynn = buildNetwork([l1,l2,l3],squaredCost,name="Multinomial logistic regression Model Sepal") # Build the NN and use the squared cost (aka MSE) as error function

# Training it (default to SGD)
res = train!(mynn,scale(xtrain),ytrain_oh,epochs=100,batchSize=6) # Use optAlg=SGD (Stochastic Gradient Descent) by default

# Test it
ŷtrain        = predict(mynn,scale(xtrain))   # Note the scaling function
ŷtest         = predict(mynn,scale(xtest))
trainAccuracy = accuracy(ŷtrain,ytrain,tol=1) # 0.983
testAccuracy  = accuracy(ŷtest,ytest,tol=1)   # 1.0

# Visualise results
testSize = size(ŷtest,1)
ŷtestChosen =  [argmax(ŷtest[i,:]) for i in 1:testSize]
groupedbar([ytest ŷtestChosen], label=["ytest" "ŷtest (est)"], title="True vs estimated categories") # All records correctly labelled !
plot(0:res.epochs,res.ϵ_epochs, ylabel="epochs",xlabel="error",legend=nothing,title="Avg. error per epoch on the Sepal dataset")
```

![results](assets/sepalOutput_results.png)

![results](assets/sepalOutput_errors.png)

## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).

[![BLogos](assets/logos_betaumr.png)](hhttp://www.beta-umr7522.fr/)
