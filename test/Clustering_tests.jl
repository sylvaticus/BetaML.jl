using Test
#using DelimitedFiles, LinearAlgebra

#using StableRNGs
#rng = StableRNG(123)

import MLJBase
const Mlj = MLJBase
using BetaML

TESTRNG = FIXEDRNG # This could change...

println("*** Testing Clustering...")

# ==================================
# New test
# ==================================
println("Testing initRepreserntative...")

Z₀ = initRepresentatives([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,initStrategy="given",Z₀=[1.7 15; 3.6 40])

@test isapprox(Z₀,[1.7  15.0; 3.6  40.0])

# ==================================
# New test
# ==================================
println("Testing kmeans...")

(clIdx,Z) = kmeans([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,rng=copy(TESTRNG))

@test clIdx == [2, 2, 2, 2, 3, 3, 3, 1, 1]
#@test (clIdx,Z) .== ([2, 2, 2, 2, 3, 3, 3, 1, 1], [5.15 -2.3499999999999996; 1.5 11.075; 3.366666666666667 36.666666666666664])

# ==================================
# New test
# ==================================
println("Testing kmedoids...")
(clIdx,Z) = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.3 38; 5.1 -2.3; 5.2 -2.4],3,initStrategy="grid")
@test clIdx == [2, 2, 2, 2, 3, 3, 3, 1, 1]


# ==================================
# New test
# ==================================
println("Testing mixture initialisation and log-pdf...")

m1 = SphericalGaussian()
m2 = SphericalGaussian([1.1,2,3])
m3 = SphericalGaussian(nothing,10.2)
mixtures = [m1,m2,m3]
X = [1 10 20; 1.2 12 missing; 3.1 21 41; 2.9 18 39; 1.5 15 25]
initMixtures!(mixtures,X,minVariance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.μ) for m in mixtures]) ≈ 102.2
@test sum([sum(m.σ²) for m in mixtures]) ≈ 19.651086419753085
mask = [true, true, false]
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.818323669882357

m1 = DiagonalGaussian()
m2 = DiagonalGaussian([1.1,2,3])
m3 = DiagonalGaussian(nothing,[0.1,11,25.0])
mixtures = [m1,m2,m3]
initMixtures!(mixtures,X,minVariance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.σ²) for m in mixtures]) ≈ 291.27933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.4365786131066063

m1 = FullGaussian()
m2 = FullGaussian([1.1,2,3])
m3 = FullGaussian(nothing,[0.1 0.2 0.5; 0 2 0.8; 1 0 5])
mixtures = [m1,m2,m3]
initMixtures!(mixtures,X,minVariance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.σ²) for m in mixtures]) ≈ 264.77933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.4365786131066063

# ==================================
# New test
# ==================================
println("Testing gmm...")
clusters = gmm([1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4],3,verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test isapprox(clusters.BIC,119.04816608007282)

# ==================================
# New test
# ==================================
println("Testing predictMissing...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out = predictMissing(X,3,mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test isapprox(out.X̂[2,2],14.187187936786232)

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out2 = predictMissing(X,3,mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test out2.X̂[2,2] ≈ 11.438358350316872

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out3 = predictMissing(X,3,mixtures=[FullGaussian() for i in 1:3],verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test out3.X̂[2,2] ≈ 11.166652292936876

# ==================================
# NEW TEST
println("Testing MLJ interface for Clustering models....")
X, y                           = Mlj.@load_iris
model                          = KMeans(rng=copy(TESTRNG))
modelMachine                   = Mlj.machine(model, X)
(fitResults, cache, report)    = Mlj.fit(model, 0, X)
distances                      = Mlj.transform(model,fitResults,X)
yhat                           = Mlj.predict(model, fitResults, X)
acc = accuracy(Mlj.levelcode.(yhat),Mlj.levelcode.(y),ignoreLabels=true)
@test acc > 0.8

model                          = KMedoids(rng=copy(TESTRNG))
modelMachine                   = Mlj.machine(model, X)
(fitResults, cache, report)    = Mlj.fit(model, 0, X)
distances                      = Mlj.transform(model,fitResults,X)
yhat                           = Mlj.predict(model, fitResults, X)
acc = accuracy(Mlj.levelcode.(yhat),Mlj.levelcode.(y),ignoreLabels=true)
@test acc > 0.8

model                       =  GMM(rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model, nothing, X) # DimensionMismatch
(fitResults, cache, report) =  Mlj.fit(model, 0, nothing, X)
yhat_prob                   =  Mlj.transform(model,fitResults,X)
yhat_prob                   =  Mlj.predict(model, fitResults, X)
@test length(yhat_prob)     == size(Mlj.matrix(X),1)

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
X = Mlj.table(X)
model                       =  MissingImputator(rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model,X)
(fitResults, cache, report) = Mlj.fit(model, 0, X)
XD                          =  Mlj.transform(model,fitResults,X)
XDM                         =  Mlj.matrix(XD)
@test isapprox(XDM[2,2],11.166666666667362)
# Use the previously learned structure to imput missings..
Xnew_withMissing            = Mlj.table([1.5 missing; missing 38; missing -2.3; 5.1 -2.3])
XDNew                       = Mlj.transform(model,fitResults,Xnew_withMissing)
XDMNew                      =  Mlj.matrix(XDNew)
@test isapprox(XDMNew[1,2],11.166666666667362)

#=
# Marginally different
XD  == XDNew
Mlj.matrix(XD)  ≈ Mlj.matrix(XDNew)
XDM  == XDMNew

for r in 1:size(XDM,1)
    for c in 1:size(XDM,2)
        if XDM[r,c] != XDMNew[r,c]
            println("($r,$c): $(XDM[r,c]) - $(XDMNew[r,c])")
        end
    end
end
=#


#=
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat_prob, y)) < 0.0002
Mlj.predict_mode(yhat_prob)
N = size(Mlj.matrix(X),1)
nCl = size(fitResults[2],1)
yhat_matrix = Array{Float64,2}(undef,N,nCl)
[yhat_matrix[n,c]= yhat_prob[n][c] for n in 1:N for c in 1:nCl]
yhat_prob[2]
 Mlj.matrix(yhat_prob)
acc = accuracy(Mlj.levelcode.(yhat),Mlj.levelcode.(y),ignoreLabels=true)
ynorm = Mlj.levelcode.(y)
accuracy(yhat_prob,ynorm,ignoreLabels=true)
@test acc > 0.8
=#

#=
using MLJBase, BetaML
y, _                        =  make_regression(1000, 3, rng=123);
ym                          =  MLJBase.matrix(y)
model                       =  GMM(rng=copy(BetaML.FIXEDRNG))
(fitResults, cache, report) =  MLJBase.fit(model, 0, nothing, ym)
yhat_prob                   =  MLJBase.transform(model,fitResults,ym)
yhat_prob                   =  MLJBase.predict(model, fitResults, ym)
modelMachine                =  MLJBase.machine(model, nothing, y)
mach                        =  MLJBase.fit!(modelMachine)
yhat_prob                   =  MLJBase.predict(mach, nothing)
=#
