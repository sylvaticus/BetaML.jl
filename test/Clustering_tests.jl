using Test, DelimitedFiles


using BetaML
import BetaML.Clustering: init_representatives, kmeans, kmedoids

TESTRNG = FIXEDRNG # This could change...

println("*** Testing Clustering...")

# ==================================
# New test
# ==================================
println("Testing initRepreserntative...")

initial_representatives = init_representatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initialisation_strategy="given",initial_representatives=[1.7 15; 3.6 40])
@test isapprox(initial_representatives,[1.7  15.0; 3.6  40.0])

# ==================================
# New test
# ==================================
println("Testing kmeans...")

X = [1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4]

(clIdxKMeans,Z) = kmeans(X,3,initialisation_strategy="grid",rng=copy(TESTRNG))
@test clIdxKMeans == [2, 2, 2, 2, 3, 3, 3, 1, 1]
#@test (clIdx,Z) .== ([2, 2, 2, 2, 3, 3, 3, 1, 1], [5.15 -2.3499999999999996; 1.5 11.075; 3.366666666666667 36.666666666666664])
m = KMeansClusterer(n_classes=3,verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG), descr="First test k-means model")
fit_ex(m,X)
classes = predict(m)
@test clIdxKMeans == classes
X2 = [1.5 11; 3 40; 3 40; 5 -2]
classes2 = predict(m,X2)
@test classes2 == [2,3,3,1]
fit!(m,X2)
classes3 = predict(m)
@test classes3 == [2,3,3,1]
reset_ex(m)
fit!(m,X)
classes = predict(m)
@test clIdxKMeans == classes
@test info(m)["fitted_records"] == 9
@test sprint(print, m) == "First test k-means model\nKMeansClusterer - A 2-dimensions 3-classes K-Means Model (fitted on 9 records)\nDict{String, Any}(\"fitted_records\" => 9, \"av_distance_last_fit\" => 1.9492325925652934, \"xndims\" => 2)\nRepresentatives:\n[5.15 -2.3499999999999996; 1.5 11.075; 3.366666666666667 36.666666666666664]\n"

# ==================================
# New test
# ==================================
println("Testing kmedoids...")
(clIdxKMedoids,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,initialisation_strategy="shuffle",rng=copy(TESTRNG))
@test clIdxKMedoids == [2, 2, 2, 1, 3, 3, 3, 2, 2]
m = KMedoidsClusterer(n_classes=3,verbosity=NONE, initialisation_strategy="shuffle",rng=copy(TESTRNG))
fit!(m,X)
classes = predict(m)
@test clIdxKMedoids == classes
X2 = [1.5 11; 3 40; 3 40; 5 -2]
classes2 = predict(m,X2)
#@test classes2 == [1,2,2,3]
fit!(m,X2)
classes3 = predict(m)
#@test classes3 == [1,2,2,3]
@test info(m)["fitted_records"] == 13
reset!(m)
@test sprint(print, m) == "KMedoidsClusterer - A 3-classes K-Medoids Model (unfitted)"

# Testing on iris
println("Testing hard clustering on the sepal database...")
iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = convert(Array{String,1}, iris[:,5])
pd = pairwise(x) 
yi = fit!(OrdinalEncoder(),y)
m = KMeansClusterer(n_classes=3, rng=copy(TESTRNG))
ŷ = fit!(m,x)
acc = accuracy(yi,ŷ,ignorelabels=true)
s = mean(silhouette(pd,ŷ))
@test s > 0.55
m = KMedoidsClusterer(n_classes=3, rng=copy(TESTRNG))
ŷ = fit!(m,x)
acc = accuracy(yi,ŷ,ignorelabels=true)
@test acc > 0.8
s = mean(silhouette(pd,ŷ))
@test s > 0.52

# ==================================
# NEW TEST
println("Testing MLJ interface for Clustering models....")
import MLJBase, MLJTestInterface
const Mlj = MLJBase
const mljti = MLJTestInterface

X, y                           = Mlj.@load_iris

model                          = BetaML.Bmlj.KMeansClusterer(rng=copy(TESTRNG))
modelMachine                   = Mlj.machine(model, X)
(fitResults, cache, report)    = Mlj.fit(model, 0, X)
distances                      = Mlj.transform(model,fitResults,X)
yhat                           = Mlj.predict(model, fitResults, X)
acc = BetaML.accuracy(Mlj.levelcode.(yhat),Mlj.levelcode.(y),ignorelabels=true)
@test acc > 0.8

model                          = BetaML.Bmlj.KMedoidsClusterer(rng=copy(TESTRNG))
modelMachine                   = Mlj.machine(model, X)
(fitResults, cache, report)    = Mlj.fit(model, 0, X)
distances                      = Mlj.transform(model,fitResults,X)
yhat                           = Mlj.predict(model, fitResults, X)
acc = BetaML.accuracy(Mlj.levelcode.(yhat),Mlj.levelcode.(y),ignorelabels=true)
@test acc > 0.8

@testset "generic mlj interface test" begin
    f, s = mljti.test(
        [BetaML.Bmlj.KMeansClusterer,],
        mljti.make_regression()[1];
        mod=@__MODULE__,
        verbosity=0, # bump to debug
        throw=true,  # set to true to debug (`false` in CI)
    )
@test isempty(f)
end
