# Clustering of the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (150 records, 4 features) in 3 clusters. The algorithms use only the features, while the species are used only to compute the accuracy of the clusters (`acc`, with the best matching between clusters and species). We also report the silhouette score (`sil`) of the clusters. The scores are computed on the training subsets of the cross-validation.
iris     = readdlm(joinpath(DATADIR,"iris_shuffled.csv"),',',skipstart=1)
x        = convert(Array{Float64,2},iris[:,1:4])
ystrings = convert(Array{String,1},iris[:,5])
y        = fit!(OrdinalEncoder(categories=unique(ystrings)),ystrings)
results  = DataFrame(model=String[],time_s=Float64[],memory_MiB=Float64[],allocs=Int[],
                     acc=Float64[],acc_std=Float64[],sil=Float64[],sil_std=Float64[])

## The functions returning the cluster of each record, for BetaML and the other packages
cl_functions = OrderedDict(
    "KMeans"                    => x -> fit!(KMeansClusterer(n_classes=3,rng=copy(TESTRNG),verbosity=NONE),x),
    "KMedoids"                  => x -> fit!(KMedoidsClusterer(n_classes=3,rng=copy(TESTRNG),verbosity=NONE),x),
    "GMM"                       => x -> mode(fit!(GaussianMixtureClusterer(n_classes=3,rng=copy(TESTRNG),verbosity=NONE),x)),
    "KMeans (Clustering.jl)"    => x -> Clustering.kmeans(permutedims(x),3).assignments,
    "KMedoids (Clustering.jl)"  => x -> Clustering.kmedoids(pairwise(x),3).assignments,
    "HC (Clustering.jl)"        => x -> Clustering.cutree(Clustering.hclust(pairwise(x)),k=3),
    "DBSCAN (Clustering.jl)"    => x -> Clustering.dbscan(permutedims(x),0.8).assignments,
    "GMM (GaussianMixtures.jl)" => x -> mode(GaussianMixtures.gmmposterior(GaussianMixtures.GMM(3,x),x)[1]),
)
for (name,f) in cl_functions
    Random.seed!(123)
    with_logger(NullLogger()) do # GaussianMixtures.jl is quite verbose
        perf   = bmark(() -> f(x))
        scores = cv_scores([x,y]) do (xtrain,ytrain),(xtest,ytest),rng
            ŷtrain = f(xtrain)
            (accuracy(ytrain,ŷtrain,ignorelabels=true), mean(silhouette(pairwise(xtrain),ŷtrain)))
        end
        push!(results,[name,perf...,scores...])
    end
end

results
