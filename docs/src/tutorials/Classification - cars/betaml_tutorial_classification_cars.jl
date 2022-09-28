# # [A classification task when labels are known - determining the country of origin of cars given the cars characteristics](@id classification_tutorial)

# In this exercise we have some car technical characteristics (mpg, horsepower,weight, model year...) and the country of origin and we would like to create a model such that the country of origin can be accurately predicted given the technical characteristics.
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
# 9. car name:      _string (unique for each instance)_ - not used here


# ## Library and data loading

# We load a buch of packages that we'll use during this tutorial..
using Random, HTTP, CSV, DataFrames, BenchmarkTools, BetaML
import DecisionTree, Flux
import Pipe: @pipe
using  Test     #src

# To load the data from the internet our workflow is
# (1) Retrieve the data --> (2) Clean it --> (3) Load it --> (4) Output it as a DataFrame.

# For step (1) we use `HTTP.get()`, for step (2) we use `replace!`, for steps (3) and (4) we uses the `CSV` package, and we use the "pip" `|>` operator to chain these operations:

urlDataOriginal = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"
data = @pipe HTTP.get(urlDataOriginal).body                                                |>
             replace!(_, UInt8('\t') => UInt8(' '))                                        |>
             CSV.File(_, delim=' ', missingstring="NA", ignorerepeated=true, header=false) |>
             DataFrame;

# This results in a table where the rows are the observations (the various cars) and the column the fields. All BetaML models expect this layout.

# As the dataset is ordered, we randomly shuffle the data. Note that we pass to shuffle `copy(FIXEDRNG)` as the random nuber generator in order to obtain reproducible output ( [`FIXEDRNG`](@ref BetaML.Utils.FIXEDRNG) is nothing else than an istance of `StableRNG(123)` defined in the [`BetaML.Utils`](@ref utils_module) sub-module, but you can choose of course your own "fixed" RNG). See the [Dealing with stochasticity](@ref dealing_with_stochasticity) section in the [Getting started](@ref getting_started) tutorial for details.
data[shuffle(copy(FIXEDRNG),axes(data, 1)), :]
describe(data)

# Columns 1 to 7 contain  characteristics of the car, while column 8 encodes the country or origin ("1" -> US, "2" -> EU, "3" -> Japan). That's the variable we want to be able to predict.

# Columns 9 contains the car name, but we are not going to use this information in this tutorial.
# Note also that some fields have missing data.

# Our first step is hence to divide the dataset in features (the x) and the labels (the y) we want to predict. The `x` is then a Julia standard `Matrix` of 406 rows by 7 columns and the `y` is a vector of the 406 observations:
x     = Matrix{Union{Missing,Float64}}(data[:,1:7]);
y     = Vector{Int64}(data[:,8]);

# Some algorithms that we will use today don't work with missing data, so we need to _impute_ them. We use the [`predictMissing`](@ref) function provided by the [`BetaML.Clustering`](@ref clustering_module) sub-module. Internally the function uses a Gaussian Mixture Model to assign to the missing walue of a given record an average of the values of the non-missing records weighted for how close they are to our specific record.
# Note that the same function (`predictMissing`) can be used for Collaborative Filtering / recomendation systems. Using GMM has the advantage over traditional algorithms as k-nearest neighbors (KNN) that GMM can "detect" the hidden structure of the observed data, where some observation can be similar to a certain pool of other observvations for a certain characteristic, but similar to an other pool of observations for other characteristics.

#xFull = predictMissing(x,rng=copy(FIXEDRNG)).X̂;
#xFull = fit!(GMMImputer(rng=copy(FIXEDRNG)),x)
xFull = fit!(RFImputer(rng=copy(FIXEDRNG)),x)


# Further, some models don't work with categorical data as such, so we need to represent our `y` as a matrix with a separate column for each possible categorical value (the so called "one-hot" representation).
# For example, within a three classes field, the individual value `2` (or `"Europe"` for what it matters) would be represented as the vector `[0 1 0]`, while `3` (or `"Japan"`) would become the vector `[0 0 1]`.
# To encode as one-hot we use the function [`onehotencoder`](@ref) in [`BetaML.Utils`](@ref utils_module)
ohm = OneHotEncoder()
y_oh  = fit!(ohm,y);

# In supervised machine learning it is good practice to partition the available data in a _training_, _validation_, and _test_ subsets, where the first one is used to train the ML algorithm, the second one to train any eventual "hyper-parameters" of the algorithm and the _test_ subset is finally used to evaluate the quality of the algorithm.
# Here, for brevity, we use only the _train_ and the _test_ subsets, implicitly assuming we already know the best hyper-parameters. Please refer to the [regression tutorial](@ref regression_tutorial) for examples of how to use the validation subset to train the hyper-parameters, or even better the [clustering tutorial](@ref clustering_tutorial) for an example of using the [`cross_validation`](@ref) function.

# We use then the [`partition`](@ref) function in [BetaML.Utils](@ref utils_module), where we can specify the different data to partition (each matrix or vector to partition must have the same number of observations) and the shares of observation that we want in each subset. Here we keep 80% of observations for training (`xtrain`, `xTrainFull` and `ytrain`) and we use 20% of them for testing (`xtest`, `xTestFull` and `ytest`):
((xtrain,xtest),(xtrainFull,xtestFull),(ytrain,ytest),(ytrain_oh,ytest_oh)) = partition([x,xFull,y,y_oh],[0.8,1-0.8],rng=copy(FIXEDRNG));

# ## Random Forests

# We are now ready to use our first model, the Random Forests (in the [`BetaML.Trees`](@ref trees_module) sub-module). Random Forests build a "forest" of decision trees models and then average their predictions in order to make an overall prediction out of a feature vector.

# To "build" the forest model (i.e. to "train" it) we need to give the model the training feature matrix and the associated "true" training labels, and we need to specify the number of trees to employ (this is an example of hyper-parameters). Here we use 30 individual decision trees.

# As the labels are encoded using integers,  we need also to specify the parameter `force_classification=true`, otherwise the model would undergo a _regression_ job instead.


#myForest       = buildForest(xtrain,ytrain,30, rng=copy(FIXEDRNG),force_classification=true);

rfm      = DecisionTreeEstimator(force_classification=true,autotune=true)
ŷtrain   = fit!(rfm,xtrain,string(ytrain))



# To obtain the predicted values, we can simply use the function [`BetaML.Trees.predict`](@ref)
#src [`predict`](@ref BetaML.Trees.predict)  [`predict`](@ref forest_prediction)
# with our `myForest` model and either the training or the testing data.
#ŷtrain,ŷtest   = predict.(Ref(myForest), [xtrain,xtest],rng=copy(FIXEDRNG))
ŷtest   = predict(rfm,xtest)


# Finally we can measure the _accuracy_ of our predictions with the [`accuracy`](@ref) function, with the sidenote that we need first to "parse" the ŷs as forcing the classification job transformed automatically them to strings (they originally were integers):

trainAccuracy,testAccuracy  = accuracy.([ytrain,ytest],[parse.(Int64,mode(ŷtrain,rng=copy(FIXEDRNG))),parse.(Int64,mode(ŷtest,rng=copy(FIXEDRNG)))])
#src (0.9969230769230769,0.8024691358024691)

@test testAccuracy > 0.77 #src


# The predictions are quite good, for the training set the algoritm predicted almost all cars' origins correctly, while for the testing set (i.e. those records that has **not** been used to train the algorithm), the correct prediction level is still quite high, at 80%

# While accuracy can sometimes suffice, we may often want to better understand which categories our model has trouble to predict correctly.
# We can investigate the output of a multi-class classifier more in-deep with a [`ConfMatrix`](@ref) where the true values (`y`) are given in rows and the predicted ones (`ŷ`) in columns, together to some per-class metrics like the _precision_ (true class _i_ over predicted in class _i_), the _recall_ (predicted class _i_ over the true class _i_) and others.

# We fist build the [`ConfMatrix`](@ref BetaML.Utils.ConfMatrix) object between `ŷ` and `y` and then we print it (we do it here for the test subset):

cm = ConfMatrix(ytest,parse.(Int64,mode(ŷtest,rng=copy(FIXEDRNG))),classes=[1,2,3],labels=["US","EU","Japan"])
print(cm,"all")

# From the report we can see that Japanese cars have more trouble in being correctly classified, and in particular many Japanease cars are classified as US ones. This is likely a result of the class imbalance of the data set, and could be solved by balancing the dataset with various sampling tecniques before training the model.

# When we benchmark the resourse used (time and memory) we find that Random Forests remain pretty fast, expecially when we compare them with neural networks (see later)
# @btime buildForest(xtrain,ytrain,30, rng=copy(FIXEDRNG),force_classification=true);
# 134.096 ms (781027 allocations: 196.30 MiB)

# ### Comparision with DecisionTree.jl

# DecisionTrees.jl random forests are similar in usage: we first "build" (train) the forest and we then make predictions out of the trained model.
# The main difference is that the model requires data with nonmissing values, so we are going to use the `xtrainFull` and `xtestFull` feature labels we created earlier:
## We train the model...
model = DecisionTree.build_forest(ytrain, xtrainFull,-1,30,rng=123)
## ..and we generate predictions and measure their error
(ŷtrain,ŷtest) = DecisionTree.apply_forest.([model],[xtrainFull,xtestFull]);
(trainAccuracy,testAccuracy) = accuracy.([ytrain,ytest],[ŷtrain,ŷtest])
#src (0.9969230769230769, 0.7530864197530864)

#src nothing; cm = ConfMatrix(ŷtest,ytest,classes=[1,2,3],labels=["US","EU","Japan"])
#src nothing; println(cm)
@test testAccuracy > 0.75 #src

# While the accuracy on the training set is exactly the same as for `BetaML` random forets, `DecisionTree.jl` random forests are slighly less accurate in the testing sample.
# Where however `DecisionTrees.jl` excell is in the efficiency: they are extremelly fast and memory thrifty, even if to this benchmark we should add the resources needed to impute the missing values.

# Also, one of the reasons DecisionTrees are such efficient is that internally they sort the data to avoid repeated comparision, but in this way they work only with features that are sortable, while BetaML random forests accept virtually any kind of input without the need of adapt it.
# @btime  DecisionTree.build_forest(ytrain, xtrainFull,-1,30,rng=123);
# 1.431 ms (10875 allocations: 1.52 MiB)

# ### Neural network

# Neural networks (NN) can be very powerfull, but have two "inconvenients" compared with random forests: first, are a bit "picky". We need to do a bit of work to provide data in specific format. Note that this is _not_ feature engineering. One of the advantages on neural network is that for the most this is not needed for neural networks. However we still need to "clean" the data. One issue is that NN don't like missing data. So we need to provide them with the feature matrix "clean" of missing data. Secondly, they work only with numerical data. So we need to use the one-hot encoding we saw earlier.
# Further, they work best if the features are scaled such that each feature has mean zero and standard deviation 1. We can achieve it with the function [`scale`](@ref) or, as in this case, [`get_scalefactors`](@ref).

xScaleFactors   = get_scalefactors(xtrainFull)
D               = size(xtrainFull,2)
classes         = unique(y)
nCl             = length(classes)

# The second "inconvenient" of NN is that, while not requiring feature engineering, they stil lneed a bit of practice on the way to build the network. It's not as simple as `train(model,x,y)`. We need here to specify how we want our layers, _chain_ the layers together and then decide a _loss_ overall function. Only when we done these steps, we have the model ready for training.
# Here we define 2 [`DenseLayer`](@ref) where, for each of them, we specify the number of neurons in input (the first layer being equal to the dimensions of the data), the output layer (for a classification task, the last layer output size beying equal to the number of classes) and an _activation function_ for each layer (default the `identity` function).
ls   = 50
l1   = DenseLayer(D,ls,f=relu,rng=copy(FIXEDRNG))
l2   = DenseLayer(ls,nCl,f=relu,rng=copy(FIXEDRNG))

# For a classification the last layer is a [`VectorFunctionLayer`](@ref) that has no learnable parameters but whose activation function is applied to the ensemble of the neurons, rather than individually on each neuron. In particular, for classification we pass the [`BetaML.Utils.softmax`](@ref) function whose output has the same size as the input (and the number of classes to predict), but we can use the `VectorFunctionLayer` with any function, including the [`pool1d`](@ref) function to create a "pooling" layer (using maximum, mean or whatever other subfunction we pass to `pool1d`)

l3   = VectorFunctionLayer(nCl,f=softmax) ## Add a (parameterless) layer whose activation function (softMax in this case) is defined to all its nodes at once

# Finally we _chain_ the layers and assign a loss function with [`buildNetwork`](@ref):
mynn = buildNetwork([l1,l2,l3],squared_cost,name="Multinomial logistic regression Model Cars") ## Build the NN and use the squared cost (aka MSE) as error function (crossentropy could also be used)

# Now we can train our network using the function [`train!`](@ref). It has many options, have a look at the documentation for all the possible arguments.
# Note that we train the network based on the scaled feature matrix.
res  = train!(mynn,scale(xtrainFull,xScaleFactors),ytrain_oh,epochs=500,batch_size=8,opt_alg=ADAM(),rng=copy(FIXEDRNG)) ## Use opt_alg=SGD() to use Stochastic Gradient Descent instead

# Once trained, we can predict the label. As the trained was based on the scaled feature matrix, so must be for the predictions
(ŷtrain,ŷtest)  = predict.(Ref(mynn),[scale(xtrainFull,xScaleFactors),scale(xtestFull,xScaleFactors)])
trainAccuracy   = accuracy(ytrain,ŷtrain,rng=copy(FIXEDRNG))

#-

testAccuracy    = accuracy(ytest,ŷtest,rng=copy(FIXEDRNG))
#src (0.8953846153846153, 0.7654320987654321

@test testAccuracy > 0.76 #src

#src accuracy(mode(ŷtest,rng=copy(FIXEDRNG)),ytest)

cm = ConfMatrix(ytest,ŷtest,classes=[1,2,3],labels=["US","EU","Japan"],rng=copy(FIXEDRNG))
# print(cm)

# 4×4 Matrix{Any}:
#  "Labels"    "US"    "EU"   "Japan"
#  "US"      44       0      5
#  "EU"       3      10      3
#  "Japan"    6       2      8
# 4×4 Matrix{Any}:
#  "Labels"   "US"      "EU"   "Japan"
#  "US"      0.897959  0.0    0.102041
#  "EU"      0.1875    0.625  0.1875
#  "Japan"   0.375     0.125  0.5



# We see a bit the limits of neural networks in this example. While NN can be extremelly performant in many domains, they also require lot of data and computational power, expecially considering the many possible hyper-parameters and hence its large space in the hyper-parameter tuning.
# In this example we arrive short to the performance of random forests, yet with a significant numberof neurons.

# @btime train!(mynn,scale(xtrainFull),ytrain_oh,epochs=300,batch_size=8,rng=copy(FIXEDRNG),verbosity=NONE);
# 11.841 s (62860672 allocations: 4.21 GiB)


# ### Comparisons with Flux

# In Flux the input must be in the form (fields, observations), so we transpose our original matrices
xtrainT, ytrain_ohT = transpose.([scale(xtrainFull,xScaleFactors), ytrain_oh])
xtestT, ytest_ohT   = transpose.([scale(xtestFull,xScaleFactors), ytest_oh])


# We define the Flux neural network model in a similar way than BetaML and load it with data, we train it, predict and measure the accuracies on the training and the test sets:

#src function poolForFlux(x,wsize=5)
#src     hcat([pool1d(x[:,i],wsize;f=maximum) for i in 1:size(x,2)]...)
#src end
Random.seed!(123)

l1         = Flux.Dense(D,ls,Flux.relu)
#l2         = Flux.Dense(ls,ls,Flux.relu)
l3         = Flux.Dense(ls,nCl,Flux.relu)
Flux_nn    = Flux.Chain(l1,l3)
fluxloss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
ps         = Flux.params(Flux_nn)
nndata     = Flux.Data.DataLoader((xtrainT, ytrain_ohT), batchsize=8,shuffle=true)
begin for i in 1:500  Flux.train!(fluxloss, ps, nndata, Flux.ADAM()) end end
ŷtrain     = Flux.onecold(Flux_nn(xtrainT),1:3)
ŷtest      = Flux.onecold(Flux_nn(xtestT),1:3)
trainAccuracy =  accuracy(ŷtrain,ytrain)

#-

testAccuracy  = accuracy(ytest,ŷtest)
#src 0.9384615384615385, 0.7407407407407407
# While the train accuracy is little bit higher that BetaML, the test accuracy remains comparable

@test testAccuracy > 0.72 #src

# However the time is again lower than BetaML, even if here for "just" a factor 2
# @btime begin for i in 1:500 Flux.train!(loss, ps, nndata, Flux.ADAM()) end end;
# 5.665 s (8943640 allocations: 1.07 GiB)


# ## Summary

# This is the summary of the results we had trying to predict the country of origin of the cars, based on their technical characteristics:

# | Model                | Train acc     | Test Acc |  Training time (ms)* | Training mem (MB) *|
# |:-------------------- |:-------------:| --------:| ------------------- | ----------------- |
# | RF                   | 0.9969        | 0.8025   | 134                 | 196               |
# | RF (DecisionTree.jl) | 0.9969        | 0.7531   | 1.43                | 1.5               |
# | NN                   | 0.895         | 0.765    | 11841               | 4311              |
# | NN (Flux.jl)         | 0.938         | 0.741    | 5665                | 1096              |


# * on a Intel Core i5-8350U laptop and using the tutorial in BetaML 0.7 (without autotuning and model wrapping)


# We warn that this table just provides a rought idea of the various algorithms performances. Indeed there is a large amount of stochasticity both in the sampling of the data used for training/testing and in the initial settings of the parameters of the algorithm. For a statistically significant comparision we would have to repeat the analysis with multiple sampling (e.g. by cross-validation, see the [clustering tutorial](@ref clustering_tutorial) for an example) and initial random parameters.

# Neverthless the table above shows that, when we compare BetaML with the algorithm-specific leading packages, we found similar results in terms of accuracy, but often the leading packages are better optimised and run more efficiently (but sometimes at the cost of being less verstatile).
# Also, for this dataset, Random Forests seems to remain marginally more accurate than Neural Network, altought of course this depends on the hyper-parameters and, with a single run of the models, we don't know if this difference is significant.
