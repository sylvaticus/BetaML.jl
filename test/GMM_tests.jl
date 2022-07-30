using Test

import MLJBase
const Mlj = MLJBase
import Distributions
using BetaML

TESTRNG = FIXEDRNG # This could change...

println("*** Testing GMM...")
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
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.461552516784797

m1 = DiagonalGaussian()
m2 = DiagonalGaussian([1.1,2,3])
m3 = DiagonalGaussian(nothing,[0.1,11,25.0])
mixtures = [m1,m2,m3]
initMixtures!(mixtures,X,minVariance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.σ²) for m in mixtures]) ≈ 291.27933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.383055441795939

m1 = FullGaussian()
m2 = FullGaussian([1.1,2,3])
m3 = FullGaussian(nothing,[0.1 0.2 0.5; 0 2 0.8; 1 0 5])
mixtures = [m1,m2,m3]
initMixtures!(mixtures,X,minVariance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.σ²) for m in mixtures]) ≈ 264.77933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.383055441795939

# ==================================
# New test
# ==================================
println("Testing gmm...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
clusters = gmm(X,3,verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test isapprox(clusters.BIC,114.1492467835965)

# ==================================
# New test
# ==================================
println("Testing predictMissing...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out = predictMissing(X,3,mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test isapprox(out.X̂[2,2],14.155186593170251)

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out2 = predictMissing(X,3,mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test out2.X̂[2,2] ≈ 14.588514438886131

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
out3 = predictMissing(X,3,mixtures=[FullGaussian() for i in 1:3],verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test out3.X̂[2,2] ≈ 11.166652292936876


println("Testing GMMClusterModel...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]

m = GMMClusterModel(nClasses=3,verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
fit!(m,X)
probs = predict(m)
gmmOut = gmm(X,3,verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
@test gmmOut.pₙₖ == probs

μ_x1alone = hcat([m.par.mixtures[i].μ for i in 1:3]...)
pk_x1alone = m.par.probMixtures

X2 = [2.0 12; 3 20; 4 15; 1.5 11]

m2 = GMMClusterModel(nClasses=3,verbosity=NONE, initStrategy="grid",rng=copy(TESTRNG))
fit!(m2,X2)
#μ_x2alone = hcat([m.par.mixtures[i].μ for i in 1:3]...)
probsx2alone = predict(m2)
@test probsx2alone[1,1] < 0.999

probX2onX1model = predict(m,X2)
@test probX2onX1model[1,1] ≈ 0.5214795038476924 

fit!(m,X2) # this greately reduces mixture variance
#μ_x1x2 = hcat([m.par.mixtures[i].μ for i in 1:3]...)
probsx2 = predict(m)
@test probsx2[1,1] > 0.999 # it feels more certain as it uses the info of he first training
reset!(m)
@test sprint(print,m) == "GMMClusterModel - A 3-classes Generative Mixture Model (untrained)"

# Testing GMM Regressor 2

ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

m = GMMRegressor2(nClasses=2,rng=copy(TESTRNG))
fit!(m,xtrain,ytrain)
ŷtrain = predict(m, xtrain)
ŷtest = predict(m, xtest)
mreTrain = meanRelError(ŷtrain,ytrain)
@test mreTrain <= 0.08
mreTest  = meanRelError(ŷtest,ytest)
@test mreTest <= 0.35

# testing it with multidimensional Y
ytrain2d = hcat(ytrain,ytrain .+ 0.1)
reset!(m)
fit!(m,xtrain,ytrain2d)
ŷtrain2d = predict(m, xtrain)
mreTrain2d = meanRelError(ŷtrain2d,ytrain2d)
@test mreTrain2d <= 0.08

# ==================================
# NEW TEST
println("Testing MLJ interface for GMM models....")
X, y                           = Mlj.@load_iris

model                       =  GMMClusterer(mixtures=:diag_gaussian,rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model, X) # DimensionMismatch
(fitResults, cache, report) =  Mlj.fit(model, 0, X)
yhat_prob                   =  Mlj.predict(model, fitResults, X)  # Mlj.transform(model,fitResults,X)
# how to get this ??? Mlj.predict_mode(yhat_prob)
@test Distributions.pdf(yhat_prob[end],2) ≈ 0.5937443601647852

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
X = Mlj.table(X)
model                       =  MissingImputator(rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model,X)
(fitResults, cache, report) = Mlj.fit(model, 0, X)
XD                          =  Mlj.transform(model,fitResults,X)
XDM                         =  Mlj.matrix(XD)
@test isapprox(XDM[2,2],15.441553354222702)
# Use the previously learned structure to imput missings..
Xnew_withMissing            = Mlj.table([1.5 missing; missing 38; missing -2.3; 5.1 -2.3])
XDNew                       = Mlj.transform(model,fitResults,Xnew_withMissing)
XDMNew                      =  Mlj.matrix(XDNew)
@test isapprox(XDMNew[1,2],13.818691793037452)
