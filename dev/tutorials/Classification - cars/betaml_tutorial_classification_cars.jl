# # [A classification task when labels are known - determining the country of origin of cars given the cars characteristics](@id classification_tutorial)

# In this exercise we are provided with several technical characteristics (mpg, horsepower,weight, model year...) for several car's models, together with the country of origin of such models, and we would like to create a machine learning model such that the country of origin can be accurately predicted given the technical characteristics.
# As the information to predict is a multi-class one, this is a _[classification]_(https://en.wikipedia.org/wiki/Statistical_classification) task.
# It is a challenging exercise due to the simultaneous presence of three factors: (1) presence of missing data; (2) unbalanced data - 254 out of 406 cars are US made; (3) small dataset.

#
# Data origin:
# - dataset description: [https://archive.ics.uci.edu/ml/datasets/auto+mpg](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
#src Also useful: https://www.rpubs.com/dksmith01/cars
# - data source we use here: [https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original)

# Field description:

# 1. mpg:           _continuous_
# 2. cylinders:     _multi-valued discrete_
# 3. displacement:  _continuous_
# 4. horsepower:    _continuous_
# 5. weight:        _continuous_
# 6. acceleration:  _continuous_
# 7. model year:    _multi-valued discrete_
# 8. origin:        _multi-valued discrete_
# 9. car name:      _string (unique for each instance)_

# The car name is not used in this tutorial, so that the country is inferred only from technical data. As this field includes also the car maker, and there are several car's models from the same car maker, a more sophisticated machine learnign model could exploit this information e.g. using a bag of word encoding.

using Dates     #src
println(now(), " ", "*** Starting car classification tutorial..." )  #src

# ## Library loading and initialisation

# Activating the local environment specific to BetaML documentation
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))

# We load a buch of packages that we'll use during this tutorial..
using Random, HTTP, Plots, CSV, DataFrames, BenchmarkTools, StableRNGs, BetaML
import DecisionTree, Flux
import Pipe: @pipe
using  Test     #src
println(now(), " - getting the data..." )  #src

# Machine Learning workflows include stochastic components in several steps: in the data sampling, in the model initialisation and often in the models's own algorithms (and sometimes also in the prediciton step).
# BetaML provides a random nuber generator  (RNG) in order to simplify reproducibility ( [`FIXEDRNG`](@ref BetaML.Utils.FIXEDRNG). This is nothing else than an istance of `StableRNG(123)` defined in the [`BetaML.Utils`](@ref utils_module) sub-module, but you can choose of course your own "fixed" RNG). See the [Dealing with stochasticity](@ref dealing_with_stochasticity) section in the [Getting started](@ref getting_started) tutorial for details.

# Here we are explicit and we use our own fixed RNG:
seed = 123 # The table at the end of this tutorial has been obtained with seeds 123, 1000 and 10000
AFIXEDRNG = StableRNG(seed)

# ## Data loading and preparation

# To load the data from the internet our workflow is
# (1) Retrieve the data --> (2) Clean it --> (3) Load it --> (4) Output it as a DataFrame.

# For step (1) we use `HTTP.get()`, for step (2) we use `replace!`, for steps (3) and (4) we uses the `CSV` package, and we use the "pip" `|>` operator to chain these operations, so that no file is ever saved on disk:

urlDataOriginal = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"
data = @pipe HTTP.get(urlDataOriginal).body                                                |>
             replace!(_, UInt8('\t') => UInt8(' '))                                        |> # the original dataset has mixed field delimiters !
             CSV.File(_, delim=' ', missingstring="NA", ignorerepeated=true, header=false) |>
             DataFrame;

println(now(), " ", "- data wrangling..." )  #src
# This results in a table where the rows are the observations (the various cars' models) and the column the fields. All BetaML models expect this layout.

# As the dataset is ordered, we randomly shuffle the data.
idx = randperm(copy(AFIXEDRNG),size(data,1))
data[idx, :]
describe(data)

# Columns 1 to 7 contain  characteristics of the car, while column 8 encodes the country or origin ("1" -> US, "2" -> EU, "3" -> Japan). That's the variable we want to be able to predict.

# Columns 9 contains the car name, but we are not going to use this information in this tutorial.
# Note also that some fields have missing data.

# Our first step is hence to divide the dataset in features (the x) and the labels (the y) we want to predict. The `x` is then a Julia standard `Matrix` of 406 rows by 7 columns and the `y` is a vector of the 406 observations:
x     = Matrix{Union{Missing,Float64}}(data[:,1:7]);
y     = Vector{Int64}(data[:,8]);
x     = fit!(Scaler(),x)

# Some algorithms that we will use today don't accept missing data, so we need to _impute_ them. BetaML provides several imputation models in the [`Imputation`](@ref) module. Note that many of these imputation models can be used for Collaborative Filtering / Recomendation Systems. Models as [`GMMImputer`](@ref) have the advantage over traditional algorithms as k-nearest neighbors (KNN) that GMM can "detect" the hidden structure of the observed data, where some observation can be similar to a certain pool of other observvations for a certain characteristic, but similar to an other pool of observations for other characteristics.
# Here we use [`RFImputer`](@ref). While the model allows for reproducible multiple imputations (with the parameter `multiple_imputation=an_integer`) and multiple passages trough the various columns (fields) containing missing data (with the option `recursive_passages=an_integer`), we use here just a single imputation and a single passage. 
# As all `BetaML` models, `RFImputer` follows the patters `m=ModelConstruction(pars); fit!(m,x,[y]); est = predict(m,x)` where `est` can be an estimation of some labels or be some characteristics of x itself (the imputed version, as in this case, a reprojected version as in [`PCA`](@ref)), depending if the model is supervised or not. See the [`API user documentation`](@ref api_usage)` for more details.
# For imputers, the output of `predict` is the matrix with the imputed values replacing the missing ones, and we write here the model in a single line using a convenience feature that when the default `cache` parameter is used in the model constructor the `fit!` function returns itself the prediciton over the trained data:  

x = fit!(RFImputer(rng=copy(AFIXEDRNG)),x) # Same as `m = RFImputer(rng=copy(AFIXEDRNG)); fit!(m,x); x= predict(m,x)`

# Further, some models don't work with categorical data as well, so we need to represent our `y` as a matrix with a separate column for each possible categorical value (the so called "one-hot" representation).
# For example, within a three classes field, the individual value `2` (or `"Europe"` for what it matters) would be represented as the vector `[0 1 0]`, while `3` (or `"Japan"`) would become the vector `[0 0 1]`.
# To encode as one-hot we use the [`OneHotEncoder`](@ref) in [`BetaML.Utils`](@ref utils_module), using the same shortcut as for the imputer we used earlier:
y_oh  = fit!(OneHotEncoder(),y)

# In supervised machine learning it is good practice to partition the available data in a _training_, _validation_, and _test_ subsets, where the first one is used to train the ML algorithm, the second one to train any eventual "hyper-parameters" of the algorithm and the _test_ subset is finally used to evaluate the quality of the algorithm.
# Here, for brevity, we use only the _train_ and the _test_ subsets, implicitly assuming we already know the best hyper-parameters. Please refer to the [regression tutorial](@ref regression_tutorial) for examples of the auto-tune feature of BetaML models to "automatically" train the hyper-parameters (hint: in most cases just add the parameter `autotune=true` in the model constructor), or the [clustering tutorial](@ref clustering_tutorial) for an example of using the [`cross_validation`](@ref) function to do it manually.

# We use then the [`partition`](@ref) function in [BetaML.Utils](@ref utils_module), where we can specify the different data to partition (each matrix or vector to partition must have the same number of observations) and the shares of observation that we want in each subset. Here we keep 80% of observations for training (`xtrain`, and `ytrain`) and we use 20% of them for testing (`xtest`, and `ytest`):

((xtrain,xtest),(ytrain,ytest),(ytrain_oh,ytest_oh)) = partition([x,y,y_oh],[0.8,1-0.8],rng=copy(AFIXEDRNG));

# We finally set up a dataframe to store the accuracies of the various models we'll use.
results = DataFrame(model=String[],train_acc=Float64[],test_acc=Float64[])

# ## Random Forests
println(now(), " ", "- random forests..." )  #src

# We are now ready to use our first model, the [`RandomForestEstimator`](@ref). Random Forests build a "forest" of decision trees models and then average their predictions in order to make an overall prediction, wheter a regression or a classification.

# While here the missing data has been imputed and the dataset is comprised of only numerical values, one attractive feature of BetaML `RandomForestEstimator` is that they can work directly with missing and categorical data without any prior processing required. 

# However as the labels are encoded using integers, we need also to specify the parameter `force_classification=true`, otherwise the model would undergo a _regression_ job instead.
rfm      = RandomForestEstimator(force_classification=true, rng=copy(AFIXEDRNG))

# Opposite to the `RFImputer` and `OneHotEncoder` models used earielr, to train a `RandomForestEstimator` model we need to provide it with both the training feature matrix and the associated "true" training labels. We use the same shortcut to get the training predictions directly from the `fit!` function. In this case the predictions correspond to the labels:

ŷtrain   = fit!(rfm,xtrain,ytrain)

# You can notice that for each record the result is reported in terms of a dictionary with the possible categories and their associated probabilities.

# !!! warning
#     Only categories with non-zero probabilities are reported for each record, and being a dictionary, the order of the categories is not undefined

# For example `ŷtrain[1]` is a `Dict(2 => 0.0333333, 3 => 0.933333, 1 => 0.0333333)`, indicating an overhelming probability that that car model originates from Japan.
# To retrieve the predictions with the highest probabilities use `mode(ŷ)`:
ŷtrain_top = mode(ŷtrain,rng=copy(AFIXEDRNG))

# Why `mode` takes (optionally) a RNG ? I let the answer for you :-) 

# To obtain the predicted labels for the test set we simply run the `predict` function over the features of the test set:

ŷtest   = predict(rfm,xtest)


# Finally we can measure the _accuracy_ of our predictions with the [`accuracy`](@ref) function. We don't need to explicitly use `mode`, as `accuracy` does it itself when it is passed with predictions expressed as a dictionary:
trainAccuracy,testAccuracy  = accuracy.([ytrain,ytest],[ŷtrain,ŷtest],rng=copy(AFIXEDRNG))
#src (0.9969230769230769, 0.8271604938271605) without autotuning, (0.8646153846153846, 0.7530864197530864) with it



@test testAccuracy > 0.70 #src

# We are now ready to store our first model accuracies in the `results` dataframe: 
push!(results,["RF",trainAccuracy,testAccuracy]);


# The predictions are quite good, for the training set the algoritm predicted almost all cars' origins correctly, while for the testing set (i.e. those records that has **not** been used to train the algorithm), the correct prediction level is still quite high, at around 80% (depends on the random seed)

# While accuracy can sometimes suffice, we may often want to better understand which categories our model has trouble to predict correctly.
# We can investigate the output of a multi-class classifier more in-deep with a [`ConfusionMatrix`](@ref) where the true values (`y`) are given in rows and the predicted ones (`ŷ`) in columns, together to some per-class metrics like the _precision_ (true class _i_ over predicted in class _i_), the _recall_ (predicted class _i_ over the true class _i_) and others.

# We fist build the [`ConfusionMatrix`](@ref) model, we train it with `ŷ` and `y` and then we print it (we do it here for the test subset):

cfm = ConfusionMatrix(categories_names=Dict(1=>"US",2=>"EU",3=>"Japan"),rng=copy(AFIXEDRNG))
fit!(cfm,ytest,ŷtest) # the output is by default the confusion matrix in relative terms
print(cfm)

# From the report we can see that Japanese cars have more trouble in being correctly classified, and in particular many Japanease cars are classified as US ones. This is likely a result of the class imbalance of the data set, and could be solved by balancing the dataset with various sampling tecniques before training the model.

# If you prefer a more graphical approach, we can also plot the confusion matrix. In order to do so, we pick up information from the `info(cfm)` function. Indeed most BetaML models can be queried with `info(model)` to retrieve additional information, in terms of a dictionary, that is not necessary to the prediciton, but could still be relevant. Other functions that you can use with BetaML models are `parameters(m)` and `hyperparamaeters(m)`.

res = info(cfm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

#src # When we benchmark the resourse used (time and memory) we find that Random Forests remain pretty fast, expecially when we compare them with neural networks.
#src # @btime buildForest(xtrain,ytrain,30, rng=copy(AFIXEDRNG),force_classification=true);
#src # 134.096 ms (781027 allocations: 196.30 MiB)

# ### Comparision with DecisionTree.jl
println(now(), " ", "- DecisionTree.jl..." )  #src

# We now compare BetaML [`RandomForestEstimator`] with the random forest estimator of the package [`DecisionTrees.jl`](https://github.com/JuliaAI/DecisionTree.jl)` random forests are similar in usage: we first "build" (train) the forest and we then make predictions out of the trained model.
#src # They are much faster than [`RandomForestEstimator`], but they don't work with missing or fully categorical (unordered) data. As we will see the accuracy is roughly the same, if not a bit lower.


## We train the model...
model = DecisionTree.build_forest(ytrain, xtrain,rng=seed)
## ..and we generate predictions and measure their error
(ŷtrain,ŷtest) = DecisionTree.apply_forest.([model],[xtrain,xtest]);
(trainAccuracy,testAccuracy) = accuracy.([ytrain,ytest],[ŷtrain,ŷtest])
#src (0.9846153846153847, 0.8518518518518519)
push!(results,["RF (DecisionTrees.jl)",trainAccuracy,testAccuracy]);

#src nothing; cm = ConfMatrix(ŷtest,ytest,classes=[1,2,3],labels=["US","EU","Japan"])
#src nothing; println(cm)
@test testAccuracy > 0.70 #src

# While the accuracy on the training set is exactly the same as for `BetaML` random forets, `DecisionTree.jl` random forests are slighly less accurate in the testing sample.
# Where however `DecisionTrees.jl` excell is in the efficiency: they are extremelly fast and memory thrifty, even if we should consider also the resources needed to impute the missing values, as they don't work with missing data.

# Also, one of the reasons DecisionTrees are such efficient is that internally the data is sorted to avoid repeated comparision, but in this way they work only with features that are sortable, while BetaML random forests accept virtually any kind of input without the needs to process it.
#src @btime  DecisionTree.build_forest(ytrain, xtrain_full,-1,30,rng=123);
#src 1.431 ms (10875 allocations: 1.52 MiB)

# ### Neural network
println(now(), " ", "- neutal networks..." )  #src

# Neural networks (NN) can be very powerfull, but have two "inconvenients" compared with random forests: first, are a bit "picky". We need to do a bit of work to provide data in specific format. Note that this is _not_ feature engineering. One of the advantages on neural network is that for the most this is not needed for neural networks. However we still need to "clean" the data. One issue is that NN don't like missing data. So we need to provide them with the feature matrix "clean" of missing data. Secondly, they work only with numerical data. So we need to use the one-hot encoding we saw earlier.
# Further, they work best if the features are scaled such that each feature has mean zero and standard deviation 1. This is why we scaled the data back at the beginning of this tutorial.

# We firt measure the dimensions of our data in input (i.e. the column of the feature matrix) and the dimensions of our output, i.e. the number of categories or columns in out one-hot encoded y.

D               = size(xtrain,2)
classes         = unique(y)
nCl             = length(classes)


# The second "inconvenient" of NN is that, while not requiring feature engineering, they still need a bit of practice on the way the structure of the network is built . It's not as simple as `fit!(Model(),x,y)` (altougth BetaML provides a "default" neural network structure that can be used, it isn't often adapted to the specific task). We need instead to specify how we want our layers, _chain_ the layers together and then decide a _loss_ overall function. Only when we done these steps, we have the model ready for training.
# Here we define 2 [`DenseLayer`](@ref) where, for each of them, we specify the number of neurons in input (the first layer being equal to the dimensions of the data), the output layer (for a classification task, the last layer output size beying equal to the number of classes) and an _activation function_ for each layer (default the `identity` function).

ls   = 50 # number of neurons in the inned layer
l1   = DenseLayer(D,ls,f=relu,rng=copy(AFIXEDRNG))
l2   = DenseLayer(ls,nCl,f=relu,rng=copy(AFIXEDRNG))

# For a classification task, the last layer is a [`VectorFunctionLayer`](@ref) that has no learnable parameters but whose activation function is applied to the ensemble of the neurons, rather than individually on each neuron. In particular, for classification we pass the [`softmax`](@ref) function whose output has the same size as the input (i.e. the number of classes to predict), but we can use the `VectorFunctionLayer` with any function, including the [`pool1d`](@ref) function to create a "pooling" layer (using maximum, mean or whatever other sub-function we pass to `pool1d`)

l3   = VectorFunctionLayer(nCl,f=softmax) ## Add a (parameterless) layer whose activation function (softmax in this case) is defined to all its nodes at once

# Finally we _chain_ the layers and assign a loss function and the number of epochs we want to train the model to the constructor of [`NeuralNetworkEstimator`](@ref):
nn = NeuralNetworkEstimator(layers=[l1,l2,l3],loss=crossentropy,rng=copy(AFIXEDRNG),epochs=500)
# Aside the layer structure and size and the number of epochs, other hyper-parameters you may want to try are the `batch_size` and the optimisation algoritm to employ (`opt_alg`).

# Now we can train our network:
ŷtrain = fit!(nn, xtrain, ytrain_oh)

# Predictions are in form of a _n_records_ by _n_classes_ matrix of the probabilities of each record being in that class. To retrieve the classes with the highest probabilities we can use again the `mode` function:

ŷtrain_top = mode(ŷtrain)

# Once trained, we can predict the test labels. As the trained was based on the scaled feature matrix, so must be for the predictions
ŷtest  = predict(nn,xtest)

# And finally we can measure the accuracies and store the accuracies in the `result` dataframe:
trainAccuracy, testAccuracy   = accuracy.([ytrain,ytest],[ŷtrain,ŷtest],rng=copy(AFIXEDRNG)) 
#src (0.8923076923076924, 0.7654320987654321
push!(results,["NN",trainAccuracy,testAccuracy]);
#-

@test testAccuracy > 0.70 #src


cfm = ConfusionMatrix(categories_names=Dict(1=>"US",2=>"EU",3=>"Japan"),rng=copy(AFIXEDRNG))
fit!(cfm,ytest,ŷtest)
print(cfm)
res = info(cfm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

# While accuracies are a bit lower, the distribution of misclassification is similar, with many Jamanease cars misclassified as US ones (here we have also some EU cars misclassified as Japanease ones).


# ### Comparisons with Flux

println(now(), " ", "- Flux.jl..." )  #src

# As we did for Random Forests, we compare BetaML neural networks with the leading package for deep learning in Julia, [`Flux.jl`](https://github.com/FluxML/Flux.jl). 

# In Flux the input must be in the form (fields, observations), so we transpose our original matrices
xtrainT, ytrain_ohT = transpose.([xtrain, ytrain_oh])
xtestT, ytest_ohT   = transpose.([xtest, ytest_oh])


# We define the Flux neural network model in a similar way than BetaML and load it with data, we train it, predict and measure the accuracies on the training and the test sets:

#src function poolForFlux(x,wsize=5)
#src     hcat([pool1d(x[:,i],wsize;f=maximum) for i in 1:size(x,2)]...)
#src end
# We fix the random seed for Flux, altough you may still get different results depending on the number of threads used.. this is a problem we solve in BetaML with [`generate_parallel_rngs`](@ref).
Random.seed!(seed)

l1         = Flux.Dense(D,ls,Flux.relu)
l2         = Flux.Dense(ls,nCl,Flux.relu)
Flux_nn    = Flux.Chain(l1,l2)
fluxloss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
ps         = Flux.params(Flux_nn)
nndata     = Flux.Data.DataLoader((xtrainT, ytrain_ohT),shuffle=true)
begin for i in 1:500  Flux.train!(fluxloss, ps, nndata, Flux.ADAM()) end end
ŷtrain     = Flux.onecold(Flux_nn(xtrainT),1:3)
ŷtest      = Flux.onecold(Flux_nn(xtestT),1:3)
trainAccuracy, testAccuracy   = accuracy.([ytrain,ytest],[ŷtrain,ŷtest])
#-

push!(results,["NN (Flux.jl)",trainAccuracy,testAccuracy]);

#src 0.9384615384615385, 0.7283950617283951
# While the train accuracy is little bit higher that BetaML, the test accuracy remains comparable

@test testAccuracy > 0.65 #src

#src # However the time is again lower than BetaML, even if here for "just" a factor 2
#src # @btime begin for i in 1:500 Flux.train!(loss, ps, nndata, Flux.ADAM()) end end;
#src # 5.665 s (8943640 allocations: 1.07 GiB)

# ## Perceptron-like classifiers.
println(now(), " ", "- perceptrons-like classifiers..." )  #src

# We finaly test 3 "perceptron-like" classifiers, the "classical" Perceptron ([`PerceptronClassifier`](@ref)), one of the first ML algorithms (a linear classifier), a "kernellised" version of it ([`KernelPerceptronClassifier`](@ref), default to using the radial kernel) and "PegasosClassifier" ([`PegasosClassifier`](@ref)) another linear algorithm that starts considering a gradient-based optimisation, altought without the regularisation term as in the Support Vector Machines (SVM).  

# As for the previous classifiers we construct the model object, we train and predict and we compute the train and test accuracies:

pm = PerceptronClassifier(rng=copy(AFIXEDRNG))
ŷtrain = fit!(pm, xtrain, ytrain)
ŷtest  = predict(pm, xtest)
(trainAccuracy,testAccuracy) = accuracy.([ytrain,ytest],[ŷtrain,ŷtest])
#src (0.7784615384615384, 0.7407407407407407) without autotune, (0.796923076923077, 0.7777777777777778) with it
push!(results,["Perceptron",trainAccuracy,testAccuracy]);

kpm = KernelPerceptronClassifier(rng=copy(AFIXEDRNG))
ŷtrain = fit!(kpm, xtrain, ytrain)
ŷtest  = predict(kpm, xtest)
(trainAccuracy,testAccuracy) = accuracy.([ytrain,ytest],[ŷtrain,ŷtest])
#src (0.9661538461538461, 0.6790123456790124) without autotune, (1.0, 0.7037037037037037) with it
push!(results,["KernelPerceptronClassifier",trainAccuracy,testAccuracy]);


pegm = PegasosClassifier(rng=copy(AFIXEDRNG))
ŷtrain = fit!(pegm, xtrain, ytrain)
ŷtest  = predict(pm, xtest)
(trainAccuracy,testAccuracy) = accuracy.([ytrain,ytest],[ŷtrain,ŷtest])
#src (0.6984615384615385, 0.7407407407407407) without autotune, (0.6615384615384615, 0.7777777777777778) with it
push!(results,["Pegasaus",trainAccuracy,testAccuracy]);

# ## Summary

# This is the summary of the results we had trying to predict the country of origin of the cars, based on their technical characteristics:

println(results)

# If you clone BetaML repository  

# Model accuracies on my machine with seedd 123, 1000 and 10000 respectivelly

# | model                 | train 1   |  test 1  | train 2  |  test 2  |  train 3 |  test 3  |  
# | --------------------- | --------- | -------- | -------- | -------- | -------- | -------- |
# | RF                    |  0.996923 | 0.765432 | 1.000000 | 0.802469 | 1.000000 | 0.888889 |
# | RF (DecisionTrees.jl) |  0.975385 | 0.765432 | 0.984615 | 0.777778 | 0.975385 | 0.864198 |
# | NN                    |  0.886154 | 0.728395 | 0.916923 | 0.827160 | 0.895385 | 0.876543 |
# │ NN (Flux.jl)          |  0.793846 | 0.654321 | 0.938462 | 0.790123 | 0.935385 | 0.851852 |
# │ Perceptron            |  0.778462 | 0.703704 | 0.720000 | 0.753086 | 0.670769 | 0.654321 |
# │ KernelPerceptronClassifier      |  0.987692 | 0.703704 | 0.978462 | 0.777778 | 0.944615 | 0.827160 |
# │ Pegasaus              |  0.732308 | 0.703704 | 0.633846 | 0.753086 | 0.575385 | 0.654321 |

# We warn that this table just provides a rought idea of the various algorithms performances. Indeed there is a large amount of stochasticity both in the sampling of the data used for training/testing and in the initial settings of the parameters of the algorithm. For a statistically significant comparision we would have to repeat the analysis with multiple sampling (e.g. by cross-validation, see the [clustering tutorial](@ref clustering_tutorial) for an example) and initial random parameters.

# Neverthless the table above shows that, when we compare BetaML with the algorithm-specific leading packages, we found similar results in terms of accuracy, but often the leading packages are better optimised and run more efficiently (but sometimes at the cost of being less verstatile).
# Also, for this dataset, Random Forests seems to remain marginally more accurate than Neural Network, altought of course this depends on the hyper-parameters and, with a single run of the models, we don't know if this difference is significant.
println(now(), " ", "- DONE classification tutorial..." )  #src