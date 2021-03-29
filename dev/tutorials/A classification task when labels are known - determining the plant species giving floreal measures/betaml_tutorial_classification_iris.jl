# # A classification task: the prediction of  plant species from floreal measures (the iris tdataset)
# The task is to estimate the species of a plant given some floreal measurements. It is a very
#
# Data origin:
# - dataset description: [https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
# - data source we use here: [https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl)

# Note that even if we are estimating a time serie, we are not using here a recurrent neural network as we assume the temporal dependence to be negligible (i.e. $Y_t = f(X_t)$ alone).

# ## Library and data loading

using LinearAlgebra, Random, Statistics, DataFrames, CSV, Plots, Pipe, BetaML, BenchmarkTools, RDatasets
import Distributions: Uniform
import DecisionTree, Flux ## For comparisions
using  Test     #src


# Differently from the [regression tutorial](../A%20regression%20task%20-%20sharing%20bike%20demand%20prediction/betaml_tutorial_regression_sharingBikes.html), we load the data here from `RDatasets`, a package providing standard datasets.
iris = dataset("datasets", "iris")
describe(y)


# ## Decision Trees and Random Forests

# ### Data preparation
# The first step is to prepare the data for the analysis. This indeed depends already on the model we want to employ, as some models "accept" everything as input, no matter if the data is numerical or categorical, if it has missing values or not... while other models are instead much more exigents, and require more work to "clean up" our dataset.
# Here we start using  Decision Tree and Random Forest models that belong to the first group, so the only things we have to do is to select the variables in input (the "feature matrix", we wil lindicate it with "X") and those representing our output (the values we want to learn to predict, we call them "y"):

x = Matrix{Float64}(iris[:,1:4])
y = Vector{String}(iris[:,5])

# We can now split the dataset between the data we will use for training the algorithm (`xtrain`/`ytrain`), those for selecting the hyperparameters (`xval`/`yval`) and finally those for testing the quality of the algoritm with the optimal hyperparameters (`xtest`/`ytest`). We use the `partition` function specifying the share we want to use for these three different subsets, here 75%, 12.5% and 12.5 respectively. As the dataset is shuffled by default, to obtain replicable results we call `partition` with `rng=copy(FIXEDRNG)`, where `FIXEDRNG` is a fixed-seeded random number generator guaranteed to maintain the same stream of random numbers even between different julia versions. That's also what we use for our unit tests.

((xtrain,xval,xtest),(ytrain,yval,ytest)) = partition([x,y],[0.75,0.125,1-0.75-0.125],rng=copy(FIXEDRNG))
(ntrain, nval, ntest) = size.([ytrain,yval,ytest],1)

# We can now "tune" our model so-called hyperparameters, i.e. choose the best exogenous parameters of our algorithm, where "best" refers to some minimisation of a "loss" function between the true and the predicted value. To make the comparision we use a specific "validation" subset of data (`xval` and `yval`).

# BetaML doesn't have a dedicated function for hyperparameters optimisation, but it is easy to write some custom julia code, at least for a simple grid-based "search". Indeed one of the main reasons that a dedicated function exists in other Machine Learning libraries is that loops in other languages are slow, but this is not a problem in julia, so we can retain the flexibility to write the kind of hyperparameter tuning that best fits our needs.

# Below is an example of a possible such function. Note there are more "elegant" ways to code it, but this one does the job. We will see the various functions inside `tuneHyperParameters()` in a moment. For now let's going just to observe that `tuneHyperParameters` just loops over all the possible hyperparameters and selects the one where the error between `xval` and `yval` is minimised. For the meaning of the various hyperparameter, consult the documentation of the `buildTree` and `buildForest` functions.
# The function uses multiple threads, so we calls `generateParallelRngs()` (in the `BetaML.Utils` submodule) to generate thread-safe random number generators and locks the comparision step.
function tuneHyperParameters(model,xtrain,ytrain,xval,yval;maxDepthRange=15:15,maxFeaturesRange=size(xtrain,2):size(xtrain,2),nTreesRange=20:20,βRange=0:0,minRecordsRange=2:2,repetitions=5,rng=Random.GLOBAL_RNG)
    ## We start with an infinitely high error
    bestAcc         = +Inf
    bestMaxDepth    = 1
    bestMaxFeatures = 1
    bestMinRecords  = 2
    bestNTrees      = 1
    bestβ           = 0
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
              accVal = accuracy(ŷval,yval)
              totAttemptError += accVal
           end
           avgAttemptedError = totAttemptError / repetitions
           begin
               lock(compLock) ## This step can't be run in parallel...
               try
                   ## Select this specific combination of hyperparameters if the error is the lowest
                   if avgAttemptedError < bestAcc
                     bestAcc         = avgAttemptedError
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
    return (bestAcc,bestMaxDepth,bestMaxFeatures,bestMinRecords,bestNTrees,bestβ)
end


# We can now run the hyperparameter optimisation function with some "reasonable" ranges.
(bestAcc,bestMaxDepth,bestMaxFeatures,bestMinRecords) = tuneHyperParameters("DecisionTree",xtrain,ytrain,xval,yval,
           maxDepthRange=[size(xtrain,1),3,5,10,15],maxFeaturesRange=3:4,minRecordsRange=2:6,repetitions=8,rng=copy(FIXEDRNG))
println("DT: $bestAcc - $bestMaxDepth - $bestMaxFeatures - $bestMinRecords") #src

# Now that we have found the "optimal" hyperparameters we can build ("train") our model using them:
myTree = buildTree(xtrain,ytrain, maxDepth=bestMaxDepth, maxFeatures=bestMaxFeatures,minRecords=bestMinRecords,rng=copy(FIXEDRNG));

# Let's benchmark the time and memory usage of the training step of a decision tree:
@btime  buildTree(xtrain,ytrain, maxDepth=bestMaxDepth, maxFeatures=bestMaxFeatures,minRecords=bestMinRecords,rng=copy(FIXEDRNG));
# Individual decision trees are blazing fast, among the fastest algorithms we could use.

#-

# The above `buildTree`function produces a DecisionTree object that can be used to make predictions given some new features, i.e. given some X matrix of (number of observations x dimensions), predict the corresponding Y vector of scalers in R.
(ŷtrain,ŷval,ŷtest) = predict.([myTree], [xtrain,xval,xtest])

# Note that the above code uses the "dot syntax" to "broadcast" `predict()` over an array of label matrices. It is exactly equivalent to:
ŷtrain = predict(myTree, xtrain)
ŷval   = predict(myTree, xval)
ŷtest  = predict(myTree, xtest)

# Note the format of the output. Each of the predictions (`ŷtrain`, `ŷval` or `ŷtest`) is a vector of _dictionaries_, whose keys are the labels with a non-zero predicted probability and the value is the relatively probability.


myForest       = buildForest(xtrain,ytrain,30,rng=FIXEDRNG);
ŷtrain         = predict(myForest, xtrain)
ŷtest          = predict(myForest, xtest)
trainAccuracy  = accuracy(ŷtrain,ytrain) # 1.00
testAccuracy   = accuracy(ŷtest,ytest)   # 0.956

# ## Perceptron

# ## Neural networks
