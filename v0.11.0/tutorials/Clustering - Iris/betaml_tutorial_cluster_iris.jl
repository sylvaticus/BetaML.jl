# # [A clustering task: the prediction of  plant species from floreal measures (the iris dataset)](@id clustering_tutorial)
# The task is to estimate the species of a plant given some floreal measurements. It use the classical "Iris" dataset.
# Note that in this example we are using clustering approaches, so we try to understand the "structure" of our data, without relying to actually knowing the true labels ("classes" or "factors"). However we have chosen a dataset for which the true labels are actually known, so we can compare the accuracy of the algorithms we use, but these labels will not be used during the algorithms training.

#
# Data origin:
# - dataset description: [https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
# - data source we use here: [https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl)


# ## Library and data loading
using Dates #src
println(now(), " ", "*** Start iris clustering tutorial..." )  #src

# Activating the local environment specific to BetaML documentation
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))

# We load the Beta Machine Learning Toolkit as well as some other packages that we use in this tutorial
using BetaML
using Random, Statistics, Logging, BenchmarkTools, StableRNGs, RDatasets, Plots, DataFrames

# We are also going to compare our results with two other leading packages in Julia for clustering analysis, [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl) that provides (inter alia) kmeans and kmedoids algorithms and [`GaussianMixtures.jl`](https://github.com/davidavdav/GaussianMixtures.jl) that provides, as the name says, Gaussian Mixture Models. So we import them (we "import" them, rather than "use", not to bound their full names into namespace as some would collide with BetaML).
import Clustering, GaussianMixtures
using  Test     #src

# Here we are explicit and we use our own fixed RNG:
seed = 123 # The table at the end of this tutorial has been obtained with seeds 123, 1000 and 10000
AFIXEDRNG = StableRNG(seed)

# We do a few tweeks for the Clustering and GaussianMixtures packages. Note that in BetaML we can also control both the random seed and the verbosity in the algorithm call, not only globally
Random.seed!(seed)
#logger  = Logging.SimpleLogger(stdout, Logging.Error); global_logger(logger); ## For suppressing GaussianMixtures output
println(now(), " ", "- data wrangling..." )  #src

# Differently from the [regression tutorial](@ref regression_tutorial), we load the data here from [`RDatasets`](https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl), a package providing standard datasets.
iris = dataset("datasets", "iris")
describe(iris)

# The iris dataset  provides floreal measures in columns 1 to 4 and the assigned species name in column 5. There are no missing values

# ## Data preparation
# The first step is to prepare the data for the analysis. We collect the first 4 columns as our _feature_ `x` matrix and the last one as our `y` label vector.
# As we are using clustering algorithms, we are not actually using the labels to train the algorithms, we'll behave like we do not know them, we'll just let the algorithm "learn" from the structure of the data itself. We'll however use it to judge the accuracy that the various algorithms reach.
x       = Matrix{Float64}(iris[:,1:4]);
yLabels = unique(iris[:,5])
# As the labels are expressed as strings, the first thing we do is encode them as integers for our analysis using the [`OrdinalEncoder`](@ref) model (data isn't really needed to be actually ordered):
y  = fit!(OrdinalEncoder(categories=yLabels),iris[:,5])


# The dataset from RDatasets is ordered by species, so we need to shuffle it to avoid biases.
# Shuffling happens by default in cross_validation, but we are keeping here a copy of the shuffled version for later.
# Note that the version of [`consistent_shuffle`](@ref) that is included in BetaML accepts several n-dimensional arrays and shuffle them (by default on rows, by we can specify the dimension) keeping the association  between the various arrays in the shuffled output.
(xs,ys) = consistent_shuffle([x,y], rng=copy(AFIXEDRNG));




# ## Main analysis
println(now(), " ", "- main analysis..." )  #src

# We will try 3 BetaML models ([`KMeansClusterer`](@ref), [`KMedoidsClusterer`](@ref) and [`GaussianMixtureClusterer`](@ref)) and we compare them with `kmeans` from Clusterings.jl and `GMM` from GaussianMixtures.jl
#
# `KMeansClusterer` and `KMedoidsClusterer` works by first initialising the centers of the k-clusters (step a ). These centers, also known as the "representatives", must be selected within the data for kmedoids, while for kmeans they are the geometrical centers.
#
# Then ( step b ) the algorithms iterates toward each point to assign the point to the cluster of the closest representative (according with a user defined distance metric, default to Euclidean), and ( step c ) moves each representative at the center of its newly acquired cluster (where "center" depends again from the metric).
#
# Steps _b_ and _c_ are reiterated until the algorithm converge, i.e. the tentative k representative points (and their relative clusters) don't move any more. The result (output of the algorithm) is that each point is assigned to one of the clusters (classes).
#
# The algorithm in `GaussianMixtureClusterer` is similar in that it employs an iterative approach (the Expectation_Minimisation algorithm, "em") but here we make the hipothesis that the data points are the observed outcomes of some _mixture_ probabilistic models where we have first a k-categorical variables whose outcomes are the (unobservble) parameters of a probabilistic distribution from which the data is finally drawn. Because the parameters of each of the k-possible distributions is unobservable this is also called a model with latent variables.
#
# Most `gmm` models use the Gaussain distribution as the family of the mixture components, so we can tought the `gmm` acronym to indicate _Gaussian Mixture Model_. In BetaML we have currently implemented only Gaussain components, but any distribution could be used by just subclassing `AbstractMixture` and implementing a couple of methids (you are invited to contribute or just ask for a distribution family you are interested), so I prefer to think "gmm" as an acronym for _Generative Mixture Model_.
#
# The algorithm tries to find the mixture that maximises the likelihood that the data has been generated indeed from such mixture, where the "E" step refers to computing the probability that each point belongs to each of the k-composants (somehow similar to the step _b_ in the kmeans/kmedoids algorithms), and the "M" step estimates, giving the association probabilities in step "E", the parameters of the mixture and of the individual components (similar to step _c_).
#
# The result here is that each point has a categorical distribution (PMF) representing the probabilities that it belongs to any of the k-components (our classes or clusters). This is interesting, as `gmm` can be used for many other things that clustering. It forms the backbone of the [`GaussianMixtureImputer`](@ref) model to impute missing values (on some or all dimensions) based to how close the record seems to its pears. For the same reasons, `GaussianMixtureImputer` can also be used to predict user's behaviours (or users' appreciation) according to the behaviour/ranking made by pears ("collaborative filtering").
#
# While the result of `GaussianMixtureClusterer` is a vector of PMFs (one for each record), error measures and reports with the true values (if known) can be directly applied, as in BetaML they internally call `mode()` to retrieve the class with the highest probability for each record.
#
#
# As we are here, we also try different versions of the BetaML models, even if the default "versions" should be fine. For `KMeansClusterer` and `KMedoidsClusterer` we will try different initialisation strategies ("gird", the default one, "random" and "shuffle"), while for the `GaussianMixtureClusterer` model we'll choose different distributions of the Gaussain family (`SphericalGaussian` - where the variance is a scalar, `DiagonalGaussian` - with a vector variance, and `FullGaussian`, where the covariance is a matrix).
#
# As the result would depend on stochasticity both in the data selected and in the random initialisation, we use a cross-validation approach to run our models several times (with different data) and then we average their results.
# Cross-Validation in BetaML is very flexible and it is done using the [`cross_validation`](@ref) function. It is used by default for hyperparameters autotuning of the BetaML supervised models.
# `cross_validation` works by calling the function `f`, defined by the user, passing to it the tuple `trainData`, `valData` and `rng` and collecting the result of the function f. The specific method for which `trainData`, and `valData` are selected at each iteration depends on the specific `sampler`.
#
# We start by selectign a k-fold sampler that split our data in 5 different parts, it uses 4 for training and 1 part (not used here) for validation. We run the simulations twice and, to be sure to have replicable results, we fix the random seed (at the whole crossValidaiton level, not on each iteration).
sampler = KFold(nsplits=5,nrepeats=3,shuffle=true, rng=copy(AFIXEDRNG))

# We can now run the cross-validation with our models. Note that instead of defining the function `f` and then calling `cross_validation[f(trainData,testData,rng),[x,y],...)` we use the Julia `do` block syntax and we write directly the content of the `f` function in the `do` block.
# Also, by default cross_validation already returns the mean and the standard deviation of the output of the user-provided `f` function (or the `do` block). However this requires that the `f` function returns a single scalar. Here we are returning a vector of the accuracies of the different models (so we can run the cross-validation only once), and hence we indicate with `return_statistics=false` to cross_validation not to attempt to generate statistics but rather report the whole output.
# We'll compute the statistics ex-post.

# Inside the `do` block we do 4 things:
# - we recover from `trainData` (a tuple, as we passed a tuple to `cross_validation` too) the `xtrain` features and `ytrain` labels;
# - we run the various clustering algorithms
# - we use the real labels to compute the model accuracy. Note that the clustering algorithm know nothing about the specific label name or even their order. This is why [`accuracy`](@ref) has the parameter `ignorelabels` to compute the accuracy oven any possible permutation of the classes found.
# - we return the various models' accuracies


cOut = cross_validation([x,y],sampler,return_statistics=false) do trainData,testData,rng
          ## For unsupervised learning we use only the train data.
          ## Also, we use the associated labels only to measure the performances
         (xtrain,ytrain)  = trainData;
         ## We run the clustering algorithm and then and we compute the accuracy using the real labels:
         estcl = fit!(KMeansClusterer(n_classes=3,initialisation_strategy="grid",rng=rng),xtrain)
         kMeansGAccuracy    = accuracy(ytrain,estcl,ignorelabels=true)
         estcl = fit!(KMeansClusterer(n_classes=3,initialisation_strategy="random",rng=rng),xtrain)
         kMeansRAccuracy   = accuracy(ytrain,estcl,ignorelabels=true)
         estcl = fit!(KMeansClusterer(n_classes=3,initialisation_strategy="shuffle",rng=rng),xtrain)
         kMeansSAccuracy   = accuracy(ytrain,estcl,ignorelabels=true)
         estcl = fit!(KMedoidsClusterer(n_classes=3,initialisation_strategy="grid",rng=rng),xtrain) 
         kMedoidsGAccuracy  = accuracy(ytrain,estcl,ignorelabels=true)
         estcl = fit!(KMedoidsClusterer(n_classes=3,initialisation_strategy="random",rng=rng),xtrain)
         kMedoidsRAccuracy = accuracy(ytrain,estcl,ignorelabels=true)
         estcl = fit!(KMedoidsClusterer(n_classes=3,initialisation_strategy="shuffle",rng=rng),xtrain)
         kMedoidsSAccuracy = accuracy(ytrain,estcl,ignorelabels=true)
         estcl = fit!(GaussianMixtureClusterer(n_classes=3,mixtures=SphericalGaussian,rng=rng,verbosity=NONE),xtrain)
         gmmSpherAccuracy  = accuracy(ytrain,estcl,ignorelabels=true, rng=rng)
         estcl = fit!(GaussianMixtureClusterer(n_classes=3,mixtures=DiagonalGaussian,rng=rng,verbosity=NONE),xtrain)
         gmmDiagAccuracy   = accuracy(ytrain,estcl,ignorelabels=true, rng=rng)
         estcl = fit!(GaussianMixtureClusterer(n_classes=3,mixtures=FullGaussian,rng=rng,verbosity=NONE),xtrain)
         gmmFullAccuracy   = accuracy(ytrain,estcl,ignorelabels=true, rng=rng)
         ## For comparision with Clustering.jl
         clusteringOut     = Clustering.kmeans(xtrain', 3)
         kMeans2Accuracy   = accuracy(ytrain,clusteringOut.assignments,ignorelabels=true)
         ## For comparision with GaussianMistures.jl - sometimes GaussianMistures.jl em! fails with a PosDefException
         dGMM              = GaussianMixtures.GMM(3, xtrain; method=:kmeans, kind=:diag)
         GaussianMixtures.em!(dGMM, xtrain)
         gmmDiag2Accuracy  = accuracy(ytrain,GaussianMixtures.gmmposterior(dGMM, xtrain)[1],ignorelabels=true)
         fGMM              = GaussianMixtures.GMM(3, xtrain; method=:kmeans, kind=:full)
         GaussianMixtures.em!(fGMM, xtrain)
         gmmFull2Accuracy  = accuracy(ytrain,GaussianMixtures.gmmposterior(fGMM, xtrain)[1],ignorelabels=true)
         ## Returning the accuracies
         return kMeansGAccuracy,kMeansRAccuracy,kMeansSAccuracy,kMedoidsGAccuracy,kMedoidsRAccuracy,kMedoidsSAccuracy,gmmSpherAccuracy,gmmDiagAccuracy,gmmFullAccuracy,kMeans2Accuracy,gmmDiag2Accuracy,gmmFull2Accuracy
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


# Accuracies (mean and its standard dev.) running this scripts with different random seeds (`123`, `1000` and `10000`):
#
# | model                         | μ 1   |  σ² 1 |  μ 2  |  σ² 2 |  μ 3  |  σ² 3 |  
# | ------------------------------| ----- | ----- | ----- | ----- | ----- | ----- |
# │ kMeansG                       | 0.891 | 0.017 | 0.892 | 0.012 | 0.893 | 0.017 |
# │ kMeansR                       | 0.866 | 0.083 | 0.831 | 0.127 | 0.836 | 0.114 |
# │ kMeansS                       | 0.764 | 0.174 | 0.822 | 0.145 | 0.779 | 0.170 |
# │ kMedoidsG                     | 0.894 | 0.015 | 0.896 | 0.012 | 0.894 | 0.017 |
# │ kMedoidsR                     | 0.804 | 0.144 | 0.841 | 0.123 | 0.825 | 0.134 |
# │ kMedoidsS                     | 0.893 | 0.018 | 0.834 | 0.130 | 0.877 | 0.085 |
# │ gmmSpher                      | 0.893 | 0.016 | 0.891 | 0.016 | 0.895 | 0.017 |
# │ gmmDiag                       | 0.917 | 0.022 | 0.912 | 0.016 | 0.916 | 0.014 |
# │ gmmFull                       | 0.970 | 0.035 | 0.982 | 0.013 | 0.981 | 0.009 |
# │ kMeans (Clustering.jl)        | 0.856 | 0.112 | 0.873 | 0.083 | 0.873 | 0.089 |
# │ gmmDiag (GaussianMixtures.jl) | 0.865 | 0.127 | 0.872 | 0.090 | 0.833 | 0.152 |
# │ gmmFull (GaussianMixtures.jl) | 0.907 | 0.133 | 0.914 | 0.160 | 0.917 | 0.141 |
#
# We can see that running the script multiple times with different random seed confirm the estimated standard deviations collected with the cross_validation, with the BetaML GMM-based models and grid based ones being the most stable ones. 

#src plot(modelLabels,μs',seriestype=:scatter)
#src yerror=collect(zip(rand(12), rand(12))

# ### BetaML model accuracies

# From the output We see that the gmm models perform for this dataset generally better than kmeans or kmedoids algorithms, and they further have very low variances.
# In detail, it is the (default) `grid` initialisation that leads to the better results for `kmeans` and `kmedoids`, while for the `gmm` models it is the `FullGaussian` to perform better.

# ### Comparisions with `Clustering.jl` and `GaussianMixtures.jl`
# For this specific case, both `Clustering.jl` and `GaussianMixtures.jl` report substantially worst accuracies, and with very high variances. But we maintain the ranking that Full Gaussian gmm > Diagonal Gaussian > Kmeans accuracy.
# I suspect the reason that BetaML gmm works so well is in relation to the usage of kmeans algorithm for initialisation of the mixtures, itself initialized with a "grid" arpproach.
# The grid initialisation "guarantee" indeed that the initial means of the mixture components are well spread across the multidimensional space defined by the data, and it helps avoiding the EM algoritm to converge to a bad local optimus.

# ## Working without the labels
println(now(), " ", "- BIC based tuning of K..." )  #src

# Up to now we used the real labels to compare the model accuracies. But in real clustering examples we don't have the true classes, or we wouln't need to do clustering in the first instance, so we don't know the number of classes to use.
# There are several methods to judge clusters algorithms goodness. For likelyhood based algorithms as `GaussianMixtureClusterer` we can use a information criteria that trade the goodness of the lickelyhood with the number of parameters used to do the fit.
# BetaML provides by default in the gmm clustering outputs both the _Bayesian information criterion_  ([`BIC`](@ref bic)) and the _Akaike information criterion_  ([`AIC`](@ref aic)), where for both a lower value is better.

# We can then run the model with different number of classes and see which one leads to the lower BIC or AIC.
# We run hence `cross_validation` again with the `FullGaussian` gmm model.
# Note that we use the BIC/AIC criteria here for establishing the "best" number of classes but we could have used it also to select the kind of Gaussain distribution to use. This is one example of hyper-parameter tuning that we developed more in detail using autotuning in the [regression tutorial](@ref regression_tutorial).

# Let's try up to 4 possible classes:

K = 4
sampler = KFold(nsplits=5,nrepeats=2,shuffle=true, rng=copy(AFIXEDRNG))
cOut = cross_validation([x,y],sampler,return_statistics=false) do trainData,testData,rng
    (xtrain,ytrain)  = trainData;
    BICS = []
    AICS = []
    for k in 1:K
        m = GaussianMixtureClusterer(n_classes=k,mixtures=FullGaussian,rng=rng,verbosity=NONE)
        fit!(m,xtrain)
        push!(BICS,info(m)["BIC"])
        push!(AICS,info(m)["AIC"])
    end
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

# We see that following the "lowest AIC" rule we would indeed choose three classes, while following the "lowest BIC" criteria we would have choosen only two classes. This means that there is two classes that, concerning the floreal measures used in the database, are very similar, and our models are unsure about them. Perhaps the biologists will end up one day with the conclusion that it is indeed only one specie :-).

# We could study this issue more in detail by analysing the [`ConfusionMatrix`](@ref), but the one used in BetaML does not account for the ignorelabels option (yet).
#
# ### Analysing the silhouette of the cluster
#
# A further metric to analyse cluster output is the so-called [Sinhouette method](https://en.wikipedia.org/wiki/Silhouette_(clustering))
#
# Silhouette is a distance-based metric and require as first argument a matrix of pairwise distances. This can be computed with the [`pairwise`](@ref) function, that default to using `l2_distance` (i.e. Euclidean). Many other distance functions are available in the [`Clustering`](@ref) sub-module or one can use the efficiently implemented distances from the [`Distances`](https://github.com/JuliaStats/Distances.jl) package, as in this example.

#
# We'll use here the [`silhouette`](@ref) function over a simple loop:

x,y = consistent_shuffle([x,y],dims=1)
import Distances
pd = pairwise(x,distance=Distances.euclidean) # we compute the pairwise distances
nclasses = 2:6
models = [KMeansClusterer, KMedoidsClusterer, GaussianMixtureClusterer]
println("Silhouette score by model type and class number:")
for ncl in nclasses, mtype in models
    m = mtype(n_classes=ncl, verbosity=NONE)
    ŷ = fit!(m,x)
    if mtype == GaussianMixtureClusterer
        ŷ = mode(ŷ)
    end
    s = mean(silhouette(pd,ŷ))
    println("$mtype \t ($ncl classes): $s")
end

# Highest levels are better. We see again that 2 classes have better scores !

#src # ## Benchmarking computational efficiency
#src 
#src # We now benchmark the time and memory required by the various models by using the `@btime` macro of the `BenchmarkTools` package:
#src 
#src # ```
#src # @btime kmeans($xs,3);
#src # # 261.540 μs (3777 allocations: 442.53 KiB)
#src # @btime kmedoids($xs,3);
#src # 4.576 ms (97356 allocations: 10.42 MiB)
#src # @btime gmm($xs,3,mixtures=[SphericalGaussian() for i in 1:3], verbosity=NONE);
#src # # 5.498 ms (133365 allocations: 8.42 MiB)
#src # @btime gmm($xs,3,mixtures=[DiagonalGaussian() for i in 1:3], verbosity=NONE);
#src # # 18.901 ms (404333 allocations: 25.65 MiB)
#src # @btime gmm($xs,3,mixtures=[FullGaussian() for i in 1:3], verbosity=NONE);
#src # # 49.257 ms (351500 allocations: 61.95 MiB)
#src # @btime Clustering.kmeans($xs', 3);
#src # # 17.071 μs (23 allocations: 14.31 KiB)
#src # @btime begin dGMM = GaussianMixtures.GMM(3, $xs; method=:kmeans, kind=:diag); GaussianMixtures.em!(dGMM, $xs) end;
#src # # 530.528 μs (2088 allocations: 488.05 KiB)
#src # @btime begin fGMM = GaussianMixtures.GMM(3, $xs; method=:kmeans, kind=:full); GaussianMixtures.em!(fGMM, $xs) end;
#src # # 4.166 ms (58910 allocations: 3.59 MiB)
#src # ```
#src # (_note: the values reported here are of a local pc, not of the GitHub CI server, as sometimes - depending on data and random #src initialisation - `GaussainMixtures.em!`` fails with a `PosDefException`. This in turn would lead the whole documentation to fail to #src compile_)
#src 
#src # Like for supervised models, dedicated models are much better optimized than BetaML models, and are order of magnitude more #src efficient. However even the slowest BetaML clusering model (gmm using full gaussians) is realtively fast and can handle mid-size #src datasets (tens to hundreds of thousand records) without significant slow downs.

# ## Conclusions

# We have shown in this tutorial how we can easily run clustering algorithms in BetaML with just one line of code `fit!(ChoosenClusterer(),x)`, but also how can we use cross-validation in order to help the model or parameter selection, with or whithout knowing the real classes.
# We retrieve here what we observed with supervised models. Globally the accuracy of BetaML models are comparable to those of leading specialised packages (in this case they are even better), but there is a significant gap in computational efficiency that restricts the pratical usage of BetaML to datasets that fits in the pc memory. However we trade this relative inefficiency with very flexible model definition and utility functions (for example `GaussianMixtureClusterer` works with missing data, allowing it to be used as the backbone of the [`GaussianMixtureImputer`](@ref) missing imputation function, or for collaborative reccomendation systems).
println(now(), " ", "- DONE clustering tutorial..." )  #src