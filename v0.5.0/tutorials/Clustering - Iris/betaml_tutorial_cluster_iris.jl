# # [A classification task: the prediction of  plant species from floreal measures (the iris dataset)](@id clustering_tutorial)
# The task is to estimate the species of a plant given some floreal measurements. It use the classical "Iris" dataset.
# Note that in this example we are using clustering approaches, so we try to understand the "structure" of our data, without relying to actually knowing the true labels ("classes" or "factors"). However we have chosen a dataset for which the true labels are actually known, so to compare the accuracy of the algorithms we use, but these labels will not be used during the algorithms training.

#
# Data origin:
# - dataset description: [https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
# - data source we use here: [https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl)


# ## Library and data loading
# We load the Beta Machine Learning Toolkit as well as some other packages that we use in this tutorial
using BetaML
using Random, Statistics, Logging, BenchmarkTools, RDatasets, Plots, DataFrames

# We are also going to compare our results with two other leading packages in Julia for clustering analysis, [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl) that provides (inter alia) kmeans and kmedoids algorithms and [`GaussianMixtures.jl`](https://github.com/davidavdav/GaussianMixtures.jl) that provides, as the name says, Gaussian Mixture Models. So we import them (we "import" them, rather than "use", not to bound their full names into namespace as some would collide with BetaML).
import Clustering, GaussianMixtures
using  Test     #src

# We do a few tweeks for the Clustering and GaussianMixtures packages. Note that in BetaML we can also control both the random seed and the verbosity in the algorithm call, not only globally
Random.seed!(123)
#logger  = Logging.SimpleLogger(stdout, Logging.Error); global_logger(logger); ## For suppressing GaussianMixtures output

# Differently from the [regression tutorial](@ref regression_tutorial), we load the data here from [`RDatasets`](https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl), a package providing standard datasets.
iris = dataset("datasets", "iris")
describe(iris)

# The iris dataset  provides floreal measures in columns 1 to 4 and the assigned species name in column 5. There are no missing values

# ## Data preparation
# The first step is to prepare the data for the analysis. We collect the first 4 columns as our _feature_ `x` matrix and the last one as our `y` label vector.
# As we are using clustering algorithms, we are not actually using the labels to train the algorithms, we'll behave like we do not know them, we'll just let the algorithm "learn" fro mthe structure of the data itself. We'll however use it to judge the accuracy that they did reach.
x       = Matrix{Float64}(iris[:,1:4]);
yLabels = unique(iris[:,5]);
# As the labels are expressed as strings, the first thing we do is encode them as integers for our analysis using the function [`integerEncoder`](@ref).
y       = integerEncoder(iris[:,5],factors=yLabels);

# The dataset from RDatasets is ordered by species, so we need to shuffle it to avoid biases.
# Shuffling happens by default in crossValidation, but we are keeping here a copy of the shuffled version for later.
# Note that the version of [`shuffle`](@ref) that is included in BetaML accepts several n-dimensional arrays and shuffle them (by default on rows, by we can specify the dimension) keeping the association  between the various arrays in the shuffled output.
(xs,ys) = shuffle([x,y]);

# ## Main analysis

# We will try 3 BetaML models ([`kmeans`](@ref), [`kmedoids`](@ref) and [`gmm`](@ref)) and we compare them with `kmeans` from Clusterings.jl and `GMM` from GaussianMixtures.jl
# `Kmeans` and `kmedoids` works by first initialising the centers of the k-clusters (the "representative" (step a ) . For `kmeans` they must be selected within one of the data, for kmeans they are the geometrical center) n a nutshell. Then ( b ) iterate for each point to assign the point to the cluster of the closest representative (according with a user defined distance metric, default to Euclidean), and ( c ) move each representative at the center of its newly acquired cluster (where "center" depends again from the metric). Steps ( b ) and ( c ) are reiterated until the algorithm converge, i.e. the tentative k representative points (and their relative clusters) don't move any more. The result (output of the algorithm) is that each point is assigned to one of the clusters (classes).
# The `gmm` algorithm is similar in that it employs an iterative approach (the Expectation_Minimisation algorithm, "em") but here we make the hipothesis that the data points are the observed outcomes of some _mixture_ probabilistic models where we have first a k-categorical variables whose outcomes are the (unobservble) parameters of a probabilistic distribution from which the data is finally drawn. Because the parameters of each of the k-possible distributions is unobservable this is also called a model with latent variables.
# Most `gmm` models use the Gaussain distribution as the family of the mixture components, so we can tought the `gmm` acronym to indicate _Gaussian Mixture Model_. In BetaML we do implemented only Gaussain components, but any distribution could be used by just subclassing `AbstractMixture` and implementing a couple of methids (you are invited to contribute or just ask for a distribution family you are interested), so I prefer to think "gmm" as an acronym for _Generative Mixture Model_.
# The algorithm try to find the mixture that maximises the likelihood that the data has been generated indeed from such mixture, where the "E" step refers to computing the probability that each point belongs to each of the k-composants (somehow similar to the step _b_ in the kmeans/kmedoids algorithm), and the "M" step estimates, giving the association probabilities in step "M", the parameters of the mixture and of the individual components (similar to step _c_).
# The result here is that each point has a categorical distribution (PMF) representing the probabilities that it belongs to any of the k-components (our classes or clusters). This is interesting, as `gmm` can be used for many other things that clustering. It forms the backbone of the [`predictMissing`](@ref) function to impute missing values (on some or all dimensions) based to how close the record seems to its pears. For the same reasons, `predictMissing` can also be used to predict user's behaviours (or users' appreciation) according to the behaviour/ranking made by pears ("collaborative filtering").
# While the result of `gmm` is a vector of PMFs (one for each record), error measures and reports with the true values (if known) can be directly applied, as in BetaML they internally call `mode()` to retrieve the class with the highest probability for each record.


# As we are here, we also try different versions of the BetaML models, even if the default "versions" should be fine. For `kmeans` and `kmedoids` we will try different initialisation strategies ("gird", the default one, "random" and "shuffle"), while for the `gmm` model we'll choose different distributions of the Gaussain family (`SphericalGaussian` - where the variance is a scalar, `DiagonalGaussian` - with a vector variance, and `FullGaussian`, where the covariance is a matrix).

# As the result would depend on stochasticity both in the data selected and in the random initialisation, we use a cross-validation approach to run our models several times (with different data) and then we average their results.
# Cross-Validation in BetaML is very flexible and it is done using the [`crossValidation`](@ref) function.
# crossValidation works by calling the function `f`, defined by the user, passing to it the tuple `trainData`, `valData` and `rng` and collecting the result of the function f. The specific method for which `trainData`, and `valData` are selected at each iteration depends on the specific `sampler`.
# We start by selectign a k-fold sampler that split our data in 5 different parts, it uses 4 for training and 1 part (not used here) for validation. We run the simulations twice and, to be sure to have replicable results, we fix the random seed (at the whole crossValidaiton level, not on each iteration).
sampler = KFold(nSplits=5,nRepeats=3,shuffle=true, rng=copy(FIXEDRNG))

# We can now run the cross-validation with our models. Note that instead of defining the function `f` and then calling `crossValidation[f(trainData,testData,rng),[x,y],...)` we use the Julia `do` block syntax and we write directly the content of the `f` function in the `do` block.
# Also, by default crossValidation already returns the mean and the standard deviation of the output of the user-provided `f` function (or the `do` block). However this requires that the `f` function return a single scalar. Here we are returning a vector of the accuracies of the different models (so we can run the cross-validation only once), and hence we indicate with `returnStatistics=false` to crossValidation not to attempt to generate statistics but rather report the whole output.
# We'll compute the statistics ex-post.

# Inside the `do` block we do 4 things:
# - we recover from `trainData` (a tuple, as we passed a tuple to `crossValidation` too) the `xtrain` features and `ytrain` labels;
# - we run the various clustering algorithms
# - we use the real labels to compute the model accuracy. Note that the clustering algorithm know nothing about the specific label name or even their order. This is why [`accuracy`](@ref) has the parameter `ignoreLabels` to compute the accuracy oven any possible permutation of the classes found.
# - we return the various models' accuracies
cOut = crossValidation([x,y],sampler,returnStatistics=false) do trainData,testData,rng
          ## For unsupervised learning we use only the train data.
          ## Also, we use the associated labels only to measure the performances
         (xtrain,ytrain)  = trainData;
         ## We run the clustering algorithm...
         clusteringOut     = kmeans(xtrain,3,rng=rng) ## init is grid by default
         ## ... and we compute the accuracy using the real labels
         kMeansAccuracy    = accuracy(clusteringOut[1],ytrain,ignoreLabels=true)
         clusteringOut     = kmeans(xtrain,3,rng=rng,initStrategy="random")
         kMeansRAccuracy   = accuracy(clusteringOut[1],ytrain,ignoreLabels=true)
         clusteringOut     = kmeans(xtrain,3,rng=rng,initStrategy="shuffle")
         kMeansSAccuracy   = accuracy(clusteringOut[1],ytrain,ignoreLabels=true)
         clusteringOut     = kmedoids(xtrain,3,rng=rng)   ## init is grid by default
         kMedoidsAccuracy  = accuracy(clusteringOut[1],ytrain,ignoreLabels=true)
         clusteringOut     = kmedoids(xtrain,3,rng=rng,initStrategy="random")
         kMedoidsRAccuracy = accuracy(clusteringOut[1],ytrain,ignoreLabels=true)
         clusteringOut     = kmedoids(xtrain,3,rng=rng,initStrategy="shuffle")
         kMedoidsSAccuracy = accuracy(clusteringOut[1],ytrain,ignoreLabels=true)
         clusteringOut     = gmm(xtrain,3,mixtures=[SphericalGaussian() for i in 1:3], verbosity=NONE, rng=rng)
         gmmSpherAccuracy  = accuracy(clusteringOut.pₙₖ,ytrain,ignoreLabels=true, rng=rng)
         clusteringOut     = gmm(xtrain,3,mixtures=[DiagonalGaussian() for i in 1:3], verbosity=NONE, rng=rng)
         gmmDiagAccuracy   = accuracy(clusteringOut.pₙₖ,ytrain,ignoreLabels=true, rng=rng)
         clusteringOut     = gmm(xtrain,3,mixtures=[FullGaussian() for i in 1:3], verbosity=NONE, rng=rng)
         gmmFullAccuracy   = accuracy(clusteringOut.pₙₖ,ytrain,ignoreLabels=true, rng=rng)
         ## For comparision with Clustering.jl
         clusteringOut     = Clustering.kmeans(xtrain', 3)
         kMeans2Accuracy   = accuracy(clusteringOut.assignments,ytrain,ignoreLabels=true)
         ## For comparision with GaussianMistures.jl - sometimes GaussianMistures.jl em! fails with a PosDefException
         dGMM              = GaussianMixtures.GMM(3, xtrain; method=:kmeans, kind=:diag)
         GaussianMixtures.em!(dGMM, xtrain)
         gmmDiag2Accuracy  = accuracy(GaussianMixtures.gmmposterior(dGMM, xtrain)[1],ytrain,ignoreLabels=true)
         fGMM              = GaussianMixtures.GMM(3, xtrain; method=:kmeans, kind=:full)
         GaussianMixtures.em!(fGMM, xtrain)
         gmmFull2Accuracy  = accuracy(GaussianMixtures.gmmposterior(fGMM, xtrain)[1],ytrain,ignoreLabels=true)
         ## Returning the accuracies
         return kMeansAccuracy,kMeansRAccuracy,kMeansSAccuracy,kMedoidsAccuracy,kMedoidsRAccuracy,kMedoidsSAccuracy,gmmSpherAccuracy,gmmDiagAccuracy,gmmFullAccuracy,kMeans2Accuracy,gmmDiag2Accuracy,gmmFull2Accuracy
 end

## We transform the output in matrix for easier analysis
accuracies = fill(0.0,(length(cOut),length(cOut[1])))
[accuracies[r,c] = cOut[r][c] for r in 1:length(cOut),c in 1:length(cOut[1])]
μs = mean(accuracies,dims=1)
σs = std(accuracies,dims=1)

@test all(μs .> 0.7) #src

@test μs[1] > 0.89 &&  μs[4] > 0.89 &&  μs[9] > 0.96 #src
modelLabels=["kMeansG","kMeansR","kMeansS","kMedoidsG","kMedoidsR","kMedoidsS","gmmSpher","gmmDiag","gmmFull","kMeans (Clustering.jl)","gmmDiag (GaussianMixtures.jl)","gmmFull (GaussianMixtures.jl)"]
report = DataFrame(mName = modelLabels, avgAccuracy = dropdims(round.(μs',digits=3),dims=2), stdAccuracy = dropdims(round.(σs',digits=3),dims=2))

#src plot(modelLabels,μs',seriestype=:scatter)
#src yerror=collect(zip(rand(12), rand(12))

# ### BetaML model accuracies

# From the output We see that the gmm models perform for this dataset generally better than kmeans or kmedoids algorithms, also with very low variances.
# In detail, it is the (default) `grid` initialisation that leads to the better results for `kmeans` and `kmedoids`, while for the `gmm` models it is the `FullGaussian` to perform better.

# ### Comparisions with `Clustering.jl` and `GaussianMixtures.jl`
# For this specific case, both `Clustering.jl` and `GaussianMixtures.jl` report substantially worst accuracies, and with very high variances. But we maintain the ranking that Full Gaussian gmm > Diagonal Gaussian > Kmeans accuracy.
# I suspect the reason that BetaML gmm works so weel is in relation to the usage of kmeans algorithm with itself the grid initialisation.
# The grid initialisation "guarantee" indeed that the initial means of the mixture components are well spread across the multidimensional space defined by the data, and it helps avoiding the EM algoritm to converge to a bad local optimus.

# ## Working without the labels

# Up to now we used the real labels to compare the model accuracies. But in real clustering examples we don't have the true classes, or we wouln't need to do clustering in the first instance, so we don't know the number of classes to use.
# There are several methods to judge clusters algorithms goodness, perhaps the simplest one, at least for the expectation-maximisation algorithm employed in `gmm` to fit the data to the unknown mixture, is to use a information criteria that trade the goodness of the lickelyhood with the parameters used to do the fit.
# BetaML provide by default in the gmm clustering outputs both the _Bayesian information criterion_  ([`BIC`](@ref bic)) and the _Akaike information criterion_  ([`AIC`](@ref aic)), where for both a lower value is better.

# We can then run the model with different number of classes and see which one leads to the lower BIC or AIC.
# We run hence `crossValidation` again with the `FullGaussian` gmm model
# Note that we use the BIC/AIC criteria here for establishing the "best" number of classes but we could have used it also to select the kind of Gaussain distribution to use. This is one example of hyper-parameter tuning that we developed more in detail (but without using cross-validation) in the [regression tutorial](@ref regression_tutorial).

# Let's try up to 8 possible classes:

K = 8
sampler = KFold(nSplits=5,nRepeats=2,shuffle=true, rng=copy(FIXEDRNG))
cOut = crossValidation([x,y],sampler,returnStatistics=false) do trainData,testData,rng
    (xtrain,ytrain)  = trainData;
    clusteringOut  = [gmm(xtrain,k,mixtures=[FullGaussian() for i in 1:k], verbosity=NONE, rng=rng) for k in 1:K]
    BICS           = [clusteringOut[i].BIC for i in 1:K]
    AICS           = [clusteringOut[i].AIC for i in 1:K]
    return (BICS,AICS)
end

## Transforming the output in matrices for easier analysis
Nit = length(cOut)

BICS = fill(0.0,(Nit,K))
AICS = fill(0.0,(Nit,K))
[BICS[r,c] = cOut[r][1][c] for r in 1:Nit,c in 1:K]
[AICS[r,c] = cOut[r][2][c] for r in 1:Nit,c in 1:K]

μsBICS = mean(BICS,dims=1)
#-
σsBICS = std(BICS,dims=1)
#-
μsAICS = mean(AICS,dims=1)
#-
σsAICS = std(AICS,dims=1)
#-
plot(1:K,[μsBICS' μsAICS'], labels=["BIC" "AIC"], title="Information criteria by number of classes", xlabel="number of classes", ylabel="lower is better")

# We see that following the "lowest AIC" rule we would indeed choose three classes, while following the "best AIC" criteria we would have choosen only two classes. This means that there is two classes that, concerning the floreal measures used in the database, are very similar, and opur models are unsure about them. Perhaps the biologists will end up one day with the conclusion that it is indeed only one specie :-).

# We could study this issue more in detail by analysing the [`ConfusionMatrix`](@ref), but the one used in BetaML does not account for the ignoreLabels option (yet).

# ## Benchmarking computational efficiency

# We now benchmark the time and memory required by the various models by using the `@btime` macro of the `BenchmarkTools` package:

# ```
# @btime kmeans($xs,3);
# # 261.540 μs (3777 allocations: 442.53 KiB)
# @btime kmedoids($xs,3);
# 4.576 ms (97356 allocations: 10.42 MiB)
# @btime gmm($xs,3,mixtures=[SphericalGaussian() for i in 1:3], verbosity=NONE);
# # 5.498 ms (133365 allocations: 8.42 MiB)
# @btime gmm($xs,3,mixtures=[DiagonalGaussian() for i in 1:3], verbosity=NONE);
# # 18.901 ms (404333 allocations: 25.65 MiB)
# @btime gmm($xs,3,mixtures=[FullGaussian() for i in 1:3], verbosity=NONE);
# # 49.257 ms (351500 allocations: 61.95 MiB)
# @btime Clustering.kmeans($xs', 3);
# # 17.071 μs (23 allocations: 14.31 KiB)
# @btime begin dGMM = GaussianMixtures.GMM(3, $xs; method=:kmeans, kind=:diag); GaussianMixtures.em!(dGMM, $xs) end;
# # 530.528 μs (2088 allocations: 488.05 KiB)
# @btime begin fGMM = GaussianMixtures.GMM(3, $xs; method=:kmeans, kind=:full); GaussianMixtures.em!(fGMM, $xs) end;
# # 4.166 ms (58910 allocations: 3.59 MiB)
# ```
# (_note: the values reported here are of a local pc, not of the GitHub CI server, as sometimes - depending on data and random initialisation - `GaussainMixtures.em!`` fails with a `PosDefException`. This in turln would lead the whole documentation to fail to compile_)

# Like for supervised models, dedicated models are much better optimized than BetaML models, and are order of magnitude more efficient. However even the slowest BetaML clusering model (gmm using full gaussians) is realtively fast and can handle mid-size datasets (tens to hundreds of thousand records) without significant slow downs.

# ## Conclusions

# We have shown in this tutorial how we can easily run clustering almgorithms in BetaML with just one line of code `choosenModel(x,k)`, but also how can we use cross-validation in order to help the model or parameter selection, with or whithout knowing the real classes.
# We retrieve here what we observed with supervised models. Globally the accuracy of BetaML models are comparable to those of leading specialised packages (in this case they are even better), but there is a significant gap in computational efficiency that restricts the pratical usage of BetaML to mid-size datasets. However we trade this relative inefficiency with very flexible model definition and utility functions (for example the BetaML gmm works with missing data, allowing it to be used as the backbone of the [`predictMissing`](@ref) missing imputation function, or for collaborative reccomendation systems).
