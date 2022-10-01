# # [A regression task: the prediction of  bike  sharing demand](@id regression_tutorial)
# The task is to estimate the influence of several variables (like the weather, the season, the day of the week..) on the demand of shared bicycles, so that the authority in charge of the service can organise the service in the best way.
#
# Data origin:
# - original full dataset (by hour, not used here): [https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
# - simplified dataset (by day, with some simple scaling): [https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec](https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec)
# - description: [https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf](https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf)
# - data: [https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip](https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip)

# Note that even if we are estimating a time serie, we are not using here a recurrent neural network as we assume the temporal dependence to be negligible (i.e. $Y_t = f(X_t)$ alone).

# !!! warning
#     As the above example is automatically run by GitHub on every code update, it uses parameters (epoch numbers, parameter space of hyperparameter validation, number of trees,...) that minimise the computation. As the GitHub script automatically update all the packages, it doesn't run exactly the same code and some output may be slightly different than the one discussed.

# ## Library and data loading

#src # Activating the local environment specific to 
#src using Pkg
#src Pkg.activate(joinpath(@__DIR__,"..","..",".."))

# We first load all the packages we are going to use
using  LinearAlgebra, Random, Statistics, StableRNGs, DataFrames, CSV, Plots, Pipe, BenchmarkTools, BetaML
import Distributions: Uniform, DiscreteUniform
import DecisionTree, Flux ## For comparisions
using  Test     #src

# Here we are explicit and we use our own fixed RNG:
seed = 123 # The table at the end of this tutorial has been obtained with seeds 123, 1000 and 10000
AFIXEDRNG = StableRNG(seed)


# Here we load the data from a csv provided by the BataML package
basedir = joinpath(dirname(pathof(BetaML)),"..","docs","src","tutorials","Regression - bike sharing")
data    = CSV.File(joinpath(basedir,"data","bike_sharing_day.csv"),delim=',') |> DataFrame
describe(data)

# The variable we want to learn to predict is `cnt`, the total demand of bikes for a given day. Even if it is indeed an integer, we treat it as a continuous variable, so each single prediction will be a scalar $Y \in \mathbb{R}$.
plot(data.cnt, title="Daily bike sharing rents (2Y)", label=nothing)

# ## Decision Trees

# We start our regression task with Decision Trees.

# Decision trees training consist in choosing the set of questions (in a hierarcical way, so to form indeed a "decision tree") that "best" split the dataset given for training, in the sense that the split generate the sub-samples (always 2 subsamples in the BetaML implementation) that are, for the characteristic we want to predict, the most homogeneous possible. Decision trees are one of the few ML algorithms that has an intuitive interpretation and can be used for both regression or classification tasks.

# ### Data preparation

# The first step is to prepare the data for the analysis. This indeed depends already on the model we want to employ, as some models "accept" almost everything as input, no matter if the data is numerical or categorical, if it has missing values or not... while other models are instead much more exigents, and require more work to "clean up" our dataset.

# The tutorial starts using  Decision Tree and Random Forest models that definitly belong to the first group, so the only thing we have to do is to select the variables in input (the "feature matrix", that we will indicate with "X") and the variable representing our output (the information we want to learn to predict, we call it "y"):
x    = Matrix{Float64}(data[:,[:instant,:season,:yr,:mnth,:holiday,:weekday,:workingday,:weathersit,:temp,:atemp,:hum,:windspeed]])
y    = data[:,16];

# We finally set up a dataframe to store the relative mean errors of the various models we'll use.
results = DataFrame(model=String[],train_rme=Float64[],test_rme=Float64[])

# ### Model selection


# We can now split the dataset between the data that we will use for training the algorithm and selecting the hyperparameters (`xtrain`/`ytrain`) and those for testing the quality of the algoritm with the optimal hyperparameters (`xtest`/`ytest`). We use the `partition` function specifying the share we want to use for these two different subsets, here 80%, and 20% respectively. As our data represents indeed a time serie, we want our model to be able to predict _future_ demand of bike sharing from _past_, observed rented bikes, so we do not shuffle the datasets as it would be the default.

((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.75,1-0.75],shuffle=false)
(ntrain, ntest) = size.([ytrain,ytest],1)

# Then we define the model we want to use, [`DecisionTreeEstimator`](@ref) in this case, and we create an instance of the model: 

m = DecisionTreeEstimator(autotune=true, rng=copy(AFIXEDRNG))

# Passing a fixed Random Number Generator (RNG) to the `rng` parameter guarantees that everytime we use the model with the same data (from the model creation downward to value prediciton) we obtain the same results. In particular BetaML provide `FIXEDRNG`, an istance of `StableRNG` that guarantees reproducibility even across different Julia versions. See the section ["Dealing with stochasticity"](@ref dealing_with_stochasticity) for details. 
# Note the `autotune` parameter. BetaML has perhaps what is the easiest method for automatically tuning the model hyperparameters (thus becoming in this way _learned_ parameters). Indeed, in most cases it is enought to pass the attribute `autotune=true` on the model constructor and hyperparameters search will be automatically performed on the first `fit!` call.
# If needed we can customise hyperparameter tuning, chosing the tuning method on the parameter `tunemethod`. The single-line above is equivalent to:
tuning_method = SuccessiveHalvingSearch(
                   hpranges     = Dict("max_depth" =>[5,10,nothing], "min_gain"=>[0.0, 0.1, 0.5], "min_records"=>[2,3,5],"max_features"=>[nothing,5,10,30]),
                   loss         = l2loss_by_cv,
                   res_shares   = [0.05, 0.2, 0.3],
                   multithreads =true
                )
m_dt = DecisionTreeEstimator(autotune=true, rng=copy(AFIXEDRNG), tunemethod=tuning_method)

# Note that the defaults change according to the specific model, for example `RandomForestEstimator`](@ref) autotuning default to not being multithreaded, as the individual model is already multithreaded.

# !!! Tip 
#     Refer to [versions of this tutorial for BetaML <= 0.6](/BetaML.jl/v0.7/tutorials/Regression - bike sharing/betaml_tutorial_regression_sharingBikes.html) for a good exercise on how to perform model selection using the [`cross_validation`](@ref) function, or even by custom grid search.

# We can now fit the model, that is learn the model parameters that lead to the best predictions from the data. By default (unless we use `cache=false` in the model constructor) the model stores also the training predictions, so we can just use `fit!()` instead of `fit!()` followed by `predict(model,xtrain)`
ŷtrain = fit!(m_dt,xtrain,ytrain) 

#src # Let's benchmark the time and memory usage of the training step of a decision tree:
#src # - including auto-tuning:
#src # ```
#src # @btime let 
#src #    m = DecisionTreeEstimator(autotune=true, rng=copy(AFIXEDRNG), verbosity=NONE, cache=false)
#src #    fit!(m,$xtrain,$ytrain)
#src # end
#src # ```
#src # 323.560 ms (4514026 allocations: 741.38 MiB)
#src # - excluding autotuning:
#src # ```
#src # m = DecisionTreeEstimator(autotune=false, rng=copy(AFIXEDRNG), verbosity=NONE, cache=false)
#src # @btime let 
#src #     fit!(m,$xtrain,$ytrain)
#src #     reset!(m)
#src # end
#src # ```
#src # 53.118 ms (242924 allocations: 91.54 MiB)
#src # Individual decision trees are blazing fast, among the fastest algorithms we could use.

#-

# The above code produces a fitted `DecisionTreeEstimator` object that can be used to make predictions given some new features, i.e. given a new X matrix of (number of observations x dimensions), predict the corresponding Y vector of scalars in R.

ŷtest  = predict(m_dt, xtest)


# We now compute the mean relative error for the training and the test set. The [`relative_mean_error`](@ref) is a very flexible error function. Without additional parameter, it computes, as the name says, the _relative mean error_, between an estimated and a true vector.
# However it can also compute the _mean relative error_, also known as the "mean absolute percentage error" ([MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)), or use a p-norm higher than 1.
# The _mean relative error_ enfatises the relativeness of the error, i.e. all observations and dimensions weigth the same, wether large or small. Conversly, in the _relative mean error_ the same relative error on larger observations (or dimensions) weights more.
# In this tutorial we use the later, as our data has clearly some outlier days with very small rents, and we care more of avoiding our customers finding empty bike racks than having unrented bikes on the rack. Targeting a low mean average error would push all our predicitons down to try accomodate the low-level predicitons (to avoid a large relative error), and that's not what we want.

# We can then compute the relative mean error for the decision tree

rme_train = relative_mean_error(ytrain,ŷtrain) # 0.1367
rme_test  = relative_mean_error(ytest,ŷtest) # 0.1547

@test rme_test <= 0.3 #src

# And we save the real mean accuracies in the `results` dataframe:
push!(results,["DT",rme_train,rme_test]);


# We can plot the true labels vs the estimated one for the three subsets...
scatter(ytrain,ŷtrain,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in training period (DT)")
#-
scatter(ytest,ŷtest,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in testing period (DT)")


# Or we can visualise the true vs estimated bike shared on a temporal base.
# First on the full period (2 years) ...
ŷtrainfull = vcat(ŷtrain,fill(missing,ntest))
ŷtestfull  = vcat(fill(missing,ntrain), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷtestfull], label=["obs" "train" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period (DT)")

# ..and then focusing on the testing period
stc = ntrain
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷtestfull[stc:endc]], label=["obs" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period (DT)")

# The predictions aren't so bad in this case, however decision trees are highly instable, and the output could have depended just from the specific initial random seed.

# ## Random Forests
# Rather than trying to solve this problem using a single Decision Tree model, let's not try to use a _Random Forest_ model. Random forests average the results of many different decision trees and provide a more "stable" result.
# Being made of many decision trees, random forests are hovever more computationally expensive to train.

m_rf      = RandomForestEstimator(autotune=true, oob=true, rng=copy(AFIXEDRNG))
ŷtrain    = fit!(m_rf,xtrain,ytrain);
ŷtest     = predict(m_rf,xtest);
rme_train = relative_mean_error(ytrain,ŷtrain) # 0.056
rme_test  = relative_mean_error(ytest,ŷtest)   # 0.161
push!(results,["RF",rme_train,rme_test]);

#src # Let's now benchmark the training of the BetaML Random Forest model
#src #
#src # - including auto-tuning:
#src # ```
#src # @btime let 
#src #    m = RandomForestEstimator(autotune=true, rng=copy(AFIXEDRNG), verbosity=NONE, cache=false)
#src #    fit!(m,$xtrain,$ytrain)
#src # end
#src #  ```
#src # 69.524 s (592717390 allocations: 80.28 GiB)
#src # - excluding autotuning:
#src # ```
#src # m = RandomForestEstimator(autotune=false, rng=copy(AFIXEDRNG), verbosity=NONE, cache=false)
#src # @btime let 
#src #     fit!(m,$xtrain,$ytrain)
#src #     reset!(m)
#src # end
#src # ```
#src # 5124.769 ms (1400309 allocations: 466.66 MiB)

# While slower than individual decision trees, random forests remain relativly fast. We should also consider that they are by default efficiently parallelised, so their speed increases with the number of available cores (in building this documentation page, GitHub CI servers allow for a single core, so all the bechmark you see in this tutorial are run with a single core available).

#-

# Random forests support the so-called "out-of-bag" error, an estimation of the error that we would have when the model is applied on a testing sample.
# However in this case the oob reported is much smaller than the testing error we will actually find. This is due to the fact that the division between training/validation and testing in this exercise is not random, but has a temporal basis. It seems that in this example the data in validation/testing follows a different pattern/variance than those in training (in probabilistic terms, the daily observations are not i.i.d.).

info(m_rf)
oob_error, rme_test  = info(m_rf)["oob_errors"],relative_mean_error(ytest,ŷtest)
#+
@test rme_test <= 0.20 #src

# In this case we found an error very similar to the one employing a single decision tree. Let's print the observed data vs the estimated one using the random forest and then along the temporal axis:
scatter(ytrain,ŷtrain,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in training period (RF)")
#-
scatter(ytest,ŷtest,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in testing period (RF)")

# Full period plot (2 years):
ŷtrainfull = vcat(ŷtrain,fill(missing,ntest))
ŷtestfull  = vcat(fill(missing,ntrain), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷtestfull], label=["obs" "train" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period (RF)")

# Focus on the testing period:
stc = 620
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷtrainfull[stc:endc] ŷtestfull[stc:endc]], label=["obs" "val" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period (RF)")

# ### Comparison with DecisionTree.jl random forest

# We now compare our results with those obtained employing the same model in the [DecisionTree package](https://github.com/bensadeghi/DecisionTree.jl), using the hyperparameters of the obtimal BetaML Random forest model:

best_rf_hp = hyperparameters(m_rf)
# Hyperparameters of the DecisionTree.jl random forest model

#src # set of classification parameters and respective default values
#src # n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
#src # n_trees: number of trees to train (default: 10)
#src # partial_sampling: fraction of samples to train each tree on (default: 0.7)
#src # max_depth: maximum depth of the decision trees (default: no maximum)
#src # min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
#src # min_samples_split: the minimum number of samples in needed for a split (default: 2)
#src # min_purity_increase: minimum purity needed for a split (default: 0.0)
#src # keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
#src #              multi-threaded forests must be seeded with an `Int`
n_subfeatures=isnothing(best_rf_hp.max_features) ? -1 : best_rf_hp.max_features; n_trees=best_rf_hp.n_trees; partial_sampling=0.7; max_depth=isnothing(best_rf_hp.max_depth) ? typemax(Int64) : best_rf_hp.max_depth;
min_samples_leaf=best_rf_hp.min_records; min_samples_split=best_rf_hp.min_records; min_purity_increase=best_rf_hp.min_gain;

# We train the model..
model = DecisionTree.build_forest(ytrain, convert(Matrix,xtrain),
                     n_subfeatures,
                     n_trees,
                     partial_sampling,
                     max_depth,
                     min_samples_leaf,
                     min_samples_split,
                     min_purity_increase;
                     rng = seed)

# And we generate predictions and measure their error
(ŷtrain,ŷtest) = DecisionTree.apply_forest.([model],[xtrain,xtest]);

#src # Let's benchmark the DecisionTrees.jl Random Forest training
#src # ```
#src # @btime  DecisionTree.build_forest(ytrain, convert(Matrix,xtrain),
#src #                      n_subfeatures,
#src #                      n_trees,
#src #                      partial_sampling,
#src #                      max_depth,
#src #                      min_samples_leaf,
#src #                      min_samples_split,
#src #                      min_purity_increase;
#src #                      rng = seed);
#src # 36.924 ms (70622 allocations: 10.09 MiB)
#src # ```
#src # DecisionTrees.jl makes a good job in optimising the Random Forest algorithm, as it is over 3 times faster that BetaML.

(rme_train, rme_test) = relative_mean_error.([ytrain,ytest],[ŷtrain,ŷtest]) # 0.022 and 0.304
push!(results,["RF (DecisionTree.jl)",rme_train,rme_test]);

# While the train error is very small, the error on the test set remains relativly high. The very low error level on the training set is a sign that it overspecialised on the training set, and we should have better ran a dedicated hyper-parameter tuning function for the DecisionTree.jl model (we did try using the default `DecisionTrees.jl` parameters, but we obtained roughtly the same results).

@test rme_test <= 0.32 #src

# Finally we plot the DecisionTree.jl predictions alongside the observed value:
ŷtrainfull = vcat(ŷtrain,fill(missing,ntest))
ŷtestfull  = vcat(fill(missing,ntrain), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷtestfull], label=["obs" "train" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period (DT.jl RF)")

# Again, focusing on the testing data:
stc  = ntrain
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷtestfull[stc:endc]], label=["obs" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period (DT.jl RF)")

# ### Conclusions of Decision Trees / Random Forests methods
# The error obtained employing DecisionTree.jl is significantly larger than those obtained using a BetaML random forest model, altought to be fair with DecisionTrees.jl we didn't tuned its hyper-parameters. Also, the DecisionTree.jl random forest model is much faster.
# This is partially due by the fact that, internally, DecisionTree.jl models optimise the algorithm by sorting the observations. BetaML trees/forests don't employ this optimisation and hence they can work with true categorical data for which ordering is not defined. An other explanation of this difference in speed is that BetaML Random Forest models accept `missing` values within the feature matrix.
# To sum up, BetaML random forests are ideal algorithms when we want to obtain good predictions in the most simpler way, even without manually tuning the hyper-parameters, and without spending time in cleaning ("munging") the feature matrix, as they accept almost "any kind" of data as it is.

# ## Neural Networks

# BetaML provides only _deep forward neural networks_, artificial neural network units where the individual "nodes" are arranged in _layers_, from the _input layer_, where each unit holds the input coordinate, through various _hidden layer_ transformations, until the actual _output_ of the model:

# ![Neural Networks](imgs/nn_scheme.png)

# In this layerwise computation, each unit in a particular layer takes input from _all_ the preceding layer units and it has its own parameters that are adjusted to perform the overall computation. The _training_ of the network consists in retrieving the coefficients that minimise a _loss_ function between the output of the model and the known data.
# In particular, a _deep_ (feedforward) neural network refers to a neural network that contains not only the input and output layers, but also (a variable number of) hidden layers in between.

# Neural networks accept only numerical inputs. We hence need to convert all categorical data in numerical units. A common approach is to use the so-called "one-hot-encoding" where the catagorical values are converted into indicator variables (0/1), one for each possible value. This can be done in BetaML using the [`OneHotEncoder`](@ref) function:
seasonDummies  = fit!(OneHotEncoder(),data.season)
weatherDummies = fit!(OneHotEncoder(),data.weathersit)
wdayDummies    = fit!(OneHotEncoder(),data.weekday .+ 1)


## We compose the feature matrix with the new dimensions obtained from the onehotencoder functions
x = hcat(Matrix{Float64}(data[:,[:instant,:yr,:mnth,:holiday,:workingday,:temp,:atemp,:hum,:windspeed]]),
         seasonDummies,
         weatherDummies,
         wdayDummies)
y = data[:,16];


# As we did for decision trees/ random forests, we split the data in training, validation and testing sets
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.75,1-0.75],shuffle=false)
(ntrain, ntest) = size.([ytrain,ytest],1)

# An other common operation with neural networks is to scale the feature vectors (X) and the labels (Y). The BetaML [`scale`](@ref) function, by default, scales the data such that each dimension has mean 0 and variance 1.

# Note that we can provide the function with different scale factors or specify the columns that shoudn't be scaled (e.g. those resulting from the one-hot encoding). Finally we can reverse the scaling (this is useful to retrieve the unscaled features from a model trained with scaled ones).

cols_nottoscale = [2;4;5;10:23]
xsm             = Scaler(skip=cols_nottoscale)
xtrain_scaled   = fit!(xsm,xtrain)
xtest_scaled    = predict(xsm,xtest)
ytrain_scaled   = ytrain ./ 1000 # We just divide Y by 1000, as using full scaling of Y we may get negative demand.
ytest_scaled    = ytest ./ 1000
D               = size(xtrain,2)

#-

# We can now build our feed-forward neaural network. We create three layers, the first layers will always have a input size equal to the dimensions of our data (the number of columns), and the output layer, for a simple regression where the predictions are scalars, it will always be one. We will tune the size of the middle layer size.

# There are already several kind of layers available (and you can build your own kind by defining a new `struct` and implementing a few functions. See the [`Nn`](@ref nn_module) module documentation for details). Here we use only _dense_ layers, those found in typycal feed-fordward neural networks.

# For each layer, on top of its size (in "neurons") we can specify an _activation function_. Here we use the [`relu`](@ref) for the terminal layer (this will guarantee that our predictions are always positive) and `identity` for the hidden layer. Again, consult the `Nn` module documentation for other activation layers already defined, or use any function of your choice.

# Initial weight parameters can also be specified if needed. By default [`DenseLayer`](@ref) use the so-called _Xavier initialisation_.

# Let's hence build our candidate neural network structures, choosing between 5 and 10 nodes in the hidden layers:

candidate_structures = [
        [DenseLayer(D,k,f=relu,df=drelu,rng=copy(AFIXEDRNG)),     # Activation function is ReLU, it's derivative is drelu
         DenseLayer(k,k,f=identity,df=identity,rng=copy(AFIXEDRNG)), # This is the hidden layer we vant to test various sizes
         DenseLayer(k,1,f=relu,df=didentity,rng=copy(AFIXEDRNG))] for k in 5:2:10]

# Note that specify the derivatives of the activation functions (and of the loss function that we'll see in a moment) it totally optional, as without them BetaML will use [`Zygote.jl`](https://github.com/FluxML/Zygote.jl for automatic differentiation.

# We do also set a few other parameters as "turnable": the number of "epochs" to train the model (the number of iterations trough the whole dataset), the sample size at each batch and the optimisation algorithm to use.
# Several optimisation algorithms are indeed available, and each accepts different parameters, like the _learning rate_ for the Stochastic Gradient Descent algorithm ([`SGD`](@ref), used by default) or the exponential decay rates for the  moments estimates for the [`ADAM`](@ref) algorithm (that we use here, with the default parameters).

# The hyperparameter ranges will then look as follow:
hpranges = Dict("layers"     => candidate_structures, 
                "epochs"     => rand(copy(AFIXEDRNG),DiscreteUniform(50,100),3), # 3 values sampled at random between 50 and 100
                "batch_size" => [4,8,16],
                "opt_alg"    => [SGD(λ=2),SGD(λ=1),SGD(λ=3),ADAM(λ=0.5),ADAM(λ=1),ADAM(λ=0.25)])

# Finally we can build "neural network" [`NeuralNetworkEstimator`](@ref) model where we "chain" the layers together and we assign a final loss function (again, you can provide your own loss function, if those available in BetaML don't suit your needs): 

nnm = NeuralNetworkEstimator(loss=squared_cost, descr="Bike sharing regression model", tunemethod=SuccessiveHalvingSearch(hpranges = hpranges), autotune=true,rng=copy(AFIXEDRNG)) # Build the NN model and use the squared cost (aka MSE) as error function by default

#src NN without any parameters:
#src nnm2                  = NeuralNetworkEstimator(autotune=true)
#src ŷtrain_scaled         = fit!(nnm2,xtrain_scaled,ytrain_scaled)
#src ŷtrain                = ŷtrain_scaled .* 1000
#src ŷtest                 = @pipe predict(nnm2,xtest_scaled) .* 1000 |> dropdims(_,dims=2)
#src (rme_train, rme_test) = relative_mean_error.([ŷtrain,ŷtest],[ytrain,ytest]) #0.041, 0.236

#-

# We can now fit and autotune the model: 
ŷtrain_scaled = fit!(nnm,xtrain_scaled,ytrain_scaled)

# The model training is one order of magnitude slower than random forests, altought the memory requirement is approximatly the same.


#-

# To obtain the neural network predictions we apply the function `predict` to the feature matrix X for which we want to generate previsions, and then we rescale y.
# Normally we would apply here the `inverse_predict` function, but as we simple divided by 1000, we multiply ŷ by the same amount:

ŷtrain = ŷtrain_scaled .* 1000 
ŷtest  = predict(nnm,xtest_scaled) .* 1000
#-
(rme_train, rme_test) = relative_mean_error.([ŷtrain,ŷtest],[ytrain,ytest])
push!(results,["NN",rme_train,rme_test]);

#src 0.134, 0.149

# The error is much lower. Let's plot our predictions:
@test rme_test < 0.25 #src

# Again, we can start by plotting the estimated vs the observed value:
scatter(ytrain,ŷtrain,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in training period (NN)")
#-
scatter(ytest,ŷtest,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in testing period (NN)")
#-

# We now plot across the time dimension, first plotting the whole period (2 years):
ŷtrainfull = vcat(ŷtrain,fill(missing,ntest))
ŷtestfull  = vcat(fill(missing,ntrain), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷtestfull], label=["obs" "train" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period  (NN)")

# ...and then focusing on the testing data
stc  = 620
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷtestfull[stc:endc]], label=["obs" "val" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period (NN)")


# ### Comparison with Flux

# We now apply the same Neural Network model using the [Flux](https://fluxml.ai/) framework, a dedicated neural network library, reusing the optimal parameters that we did learn from tuning `NeuralNetworkEstimator`:

hp_opt         = hyperparameters(nnm)
opt_size       = size(hp_opt.layers[1])[2]
opt_batch_size = hp_opt.batch_size
opt_epochs     = hp_opt.epochs

# We fix the default random number generator so that the Flux example gives a reproducible output
Random.seed!(seed)

# We define the Flux neural network model and load it with data...
l1         = Flux.Dense(D,opt_size,Flux.relu)
l2         = Flux.Dense(opt_size,opt_size,identity)
l3         = Flux.Dense(opt_size,1,Flux.relu)
Flux_nn    = Flux.Chain(l1,l2,l3)
fluxloss(x, y) = Flux.mse(Flux_nn(x), y)
ps         = Flux.params(Flux_nn)
nndata     = Flux.Data.DataLoader((xtrain_scaled', ytrain_scaled'), batchsize=opt_batch_size,shuffle=true)

#src Flux_nn2   = deepcopy(Flux_nn)      ## A copy for the time benchmarking
#src ps2        = Flux.params(Flux_nn2)  ## A copy for the time benchmarking

# We do the training of the Flux model...
[Flux.train!(fluxloss, ps, nndata, Flux.ADAM(0.001, (0.9, 0.8))) for i in 1:opt_epochs]

#src # ..and we benchmark it..
#src # ```
#src # @btime begin for i in 1:bestEpoch Flux.train!(loss, ps2, nndata, Flux.ADAM(0.001, (0.9, 0.8))) end end
#src # 690.231 ms (3349901 allocations: 266.76 MiB)
#src # ```
#src #src # Quite surprisling, Flux training seems a bit slow. The actual results seems to depend from the actual hardware and by default Flux seems not to use multi-threading. While I suspect Flux scales better with larger networks and/or data, for these small examples on my laptop it is still a bit slower than BetaML even on a single thread.
#src # On this small example the speed of Flux is on the same order than BetaML (the actual difference seems to depend on the specific RNG seed and hardware), however I suspect that Flux scales much better with larger networks and/or data.

# We obtain the predicitons...
ŷtrainf = @pipe Flux_nn(xtrain_scaled')' .* 1000;
ŷtestf  = @pipe Flux_nn(xtest_scaled')'  .* 1000;

# ..and we compute the mean relative errors..
(rme_train, rme_test) = relative_mean_error.([ŷtrainf,ŷtestf],[ytrain,ytest])
push!(results,["NN (Flux.jl)",rme_train,rme_test]);
#src 0.102, 0.171
# .. finding an error not significantly different than the one obtained from BetaML.Nn.

#-
@test rme_test < 0.3 #src

# Plots:
scatter(ytrain,ŷtrainf,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in training period (Flux.NN)")
#-
scatter(ytest,ŷtestf,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in testing period (Flux.NN)")
#-
ŷtrainfullf = vcat(ŷtrainf,fill(missing,ntest))
ŷtestfullf  = vcat(fill(missing,ntrain), ŷtestf)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfullf ŷtestfullf], label=["obs" "train" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period (Flux.NN)")
#-
stc = 620
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷtestfullf[stc:endc]], label=["obs" "val" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period (Flux.NN)")


# ### Conclusions of Neural Network models

# If we strive for the most accurate predictions, deep neural networks are usually the best choice. However they are computationally expensive, so with limited resourses we may get better results by fine tuning and running many repetitions of "simpler" decision trees or even random forest models than a large naural network with insufficient hyper-parameter tuning.
# Also, we shoudl consider that decision trees/random forests are much simpler to work with.

# That said, specialised neural network libraries, like Flux, allow to use GPU and specialised hardware letting neural networks to scale with very large datasets.

# Still, for small and medium datasets, BetaML provides simpler yet customisable solutions that are accurate and fast.

# ## GMM-based regressors
# BetaML 0.8 introduces new regression algorithms based on Gaussian Mixture Model.
# Specifically, there are two variants available, `GMMRegressor1` and `GMMRegressor2`, and this example uses  `GMMRegressor2`
# As for neural networks, they work on numerical data only, so we reuse the datasets we prepared for the neural networks.

# As usual we first define the model with the autotune option:
m = GMMRegressor2(rng=copy(AFIXEDRNG), autotune=true,verbosity=NONE)
#src # @btime begin fit!(m,xtrainScaled,ytrainScaled); reset!(m) end
#src # 13.584 ms (103690 allocations: 25.08 MiB)

# We then fit the model to the training data..
ŷtrainGMM_unscaled = fit!(m,xtrain_scaled,ytrain_scaled)
# And we predict...
ŷtrainGMM = ŷtrainGMM_unscaled .* 1000;
ŷtestGMM  = predict(m,xtest_scaled)  .* 1000;

(rme_train, rme_test) = relative_mean_error.([ŷtrainGMM,ŷtestGMM],[ytrain,ytest])
push!(results,["GMM",rme_train,rme_test]);

# ## Summary

# This is the summary of the results (train and test relative mean error) we had trying to predict the daily bike sharing demand, given weather and calendar information:

println(results)

# You may ask how stable are these results? How much do they depend from the specific RNG seed ? We re-evaluated a couple of times the whole script but changing random seeds (to `1000` and `10000`):

# | Model                | Train rme1 | Test rme1 | Train rme2 | Test rme2 | Train rme3 | Test rme3 | 
# |:-------------------- |:----------:|:---------:|:----------:|:---------:|:----------:|:---------:|
# | DT                   | 0.1366960  | 0.154720  | 0.0233044  | 0.249329  | 0.0621571  | 0.161657  |
# | RF                   | 0.0421267  | 0.180186  | 0.0535776  | 0.136920  | 0.0386144  | 0.141606  |
# | RF (DecisionTree.jl) | 0.0230439  | 0.235823  | 0.0801040  | 0.243822  | 0.0168764  | 0.219011  |
# | NN                   | 0.1604000  | 0.169952  | 0.1091330  | 0.121496  | 0.1481440  | 0.150458  | 
# | NN (Flux.jl)         | 0.0931161  | 0.166228  | 0.0920796  | 0.167047  | 0.0907810  | 0.122469  | 
# | GMMRegressor2*       | 0.1432800  | 0.293891  | 0.1380340  | 0.295470  | 0.1477570  | 0.284567  |

# * GMM is a deterministic model, the variations are due to the different random sampling in choosing the best hyperparameters

# Neural networks can be more precise than random forests models, but are more computationally expensive (and tricky to set up). When we compare BetaML with the algorithm-specific leading packages, we found similar results in terms of accuracy, but often the leading packages are better optimised and run more efficiently (but sometimes at the cost of being less versatile).
# GMM_based regressors are very computationally cheap and a good compromise if accuracy can be traded off for performances.
