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
init_mixtures!(mixtures,X,minimum_variance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.μ) for m in mixtures]) ≈ 102.2
@test sum([sum(m.σ²) for m in mixtures]) ≈ 19.651086419753085
mask = [true, true, false]
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.461552516784797

m1 = DiagonalGaussian()
m2 = DiagonalGaussian([1.1,2,3])
m3 = DiagonalGaussian(nothing,[0.1,11,25.0])
mixtures = [m1,m2,m3]
init_mixtures!(mixtures,X,minimum_variance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.σ²) for m in mixtures]) ≈ 291.27933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.383055441795939

m1 = FullGaussian()
m2 = FullGaussian([1.1,2,3])
m3 = FullGaussian(nothing,[0.1 0.2 0.5; 0 2 0.8; 1 0 5])
mixtures = [m1,m2,m3]
init_mixtures!(mixtures,X,minimum_variance=0.25,rng=copy(TESTRNG))
@test sum([sum(m.σ²) for m in mixtures]) ≈ 264.77933333333334
@test lpdf(m1,X[2,:][mask],mask) ≈ -3.383055441795939

# ==================================
# New test
# ==================================
println("Testing gmm...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
clusters = gmm(X,3,verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
@test isapprox(clusters.BIC,114.1492467835965)


println("Testing GMMClusterer...")
X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]

m = GMMClusterer(n_classes=3,verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
fit!(m,X)
probs = predict(m)
gmmOut = gmm(X,3,verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
@test gmmOut.pₙₖ == probs

μ_x1alone = hcat([m.par.mixtures[i].μ for i in 1:3]...)
pk_x1alone = m.par.initial_probmixtures

X2 = [2.0 12; 3 20; 4 15; 1.5 11]

m2 = GMMClusterer(n_classes=3,verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
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
@test sprint(print,m) == "GMMClusterer - A 3-classes Generative Mixture Model (unfitted)"

# Testing GMM Regressor 1
ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
ytrain2d = hcat(ytrain,ytrain .+ 0.1)
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

m = GMMRegressor1(n_classes=2,rng=copy(TESTRNG), verbosity=NONE)
fit!(m,xtrain,ytrain)
ŷtrain  = predict(m, xtrain)
ŷtrain2 = predict(m)
@test isapprox(ŷtrain,ŷtrain2,atol=0.00001) # not the same as the predict(m,xtrain) goes trough a further estep
ŷtest = predict(m, xtest)
mreTrain = relative_mean_error(ytrain,ŷtrain,normrec=true)
@test mreTrain <= 0.08
mreTest  = relative_mean_error(ytest,ŷtest,normrec=true)
@test mreTest <= 0.35

# testing it with multidimensional Y
reset!(m)
fit!(m,xtrain,ytrain2d)
ŷtrain2d = predict(m, xtrain)
ŷtrain2db = predict(m)
@test isapprox(ŷtrain2d,ŷtrain2db,atol=0.00001) # not the same as the predict(m,xtrain) goes trough a further estep
mreTrain2d = relative_mean_error(ytrain2d,ŷtrain2d,normrec=true)
@test mreTrain2d <= 0.08

# Testing GMM Regressor 2

m = GMMRegressor2(n_classes=2,rng=copy(TESTRNG), verbosity=NONE)
fit!(m,xtrain,ytrain)
ŷtrain = predict(m, xtrain)
ŷtrain2 = predict(m)
@test isapprox(ŷtrain,ŷtrain2,atol=0.01) # not the same as the predict(m,xtrain) goes trough a further estep
ŷtest = predict(m, xtest)
mreTrain = relative_mean_error(ytrain,ŷtrain,normrec=true)
@test mreTrain <= 0.08
mreTest  = relative_mean_error(ytest,ŷtest,normrec=true)
@test mreTest <= 0.35

# testing it with multidimensional Y
reset!(m)
fit!(m,xtrain,ytrain2d)
ŷtrain2d = predict(m, xtrain)
ŷtrain2db = predict(m)
@test isapprox(ŷtrain2d,ŷtrain2db,atol=0.01) # not the same as the predict(m,xtrain) goes trough a further estep
mreTrain2d = relative_mean_error(ytrain2d,ŷtrain2d,normrec=true)
@test mreTrain2d <= 0.08
fit!(m,xtrain,ytrain2d) # re-fit
ŷtrain2d = predict(m, xtrain)
mreTrain2d = relative_mean_error(ytrain2d,ŷtrain2d,normrec=true)
@test mreTrain2d <= 0.08



# ==================================
# NEW TEST
println("Testing MLJ interface for GMM models....")
X, y                           = Mlj.@load_iris

model                       =  GaussianMixtureClusterer(mixtures=[DiagonalGaussian() for i in 1:3],rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model, X) # DimensionMismatch
(fitResults, cache, report) =  Mlj.fit(model, 0, X)
yhat_prob                   =  Mlj.predict(model, fitResults, X)  # Mlj.transform(model,fitResults,X)
# how to get this ??? Mlj.predict_mode(yhat_prob)
@test Distributions.pdf(yhat_prob[end],2) ≈ 0.5937443601647852


println("Testing MLJ interface for GMMRegressor models....")
X, y                           = Mlj.@load_boston

model_gmmr                      = GaussianMixtureRegressor(n_classes=20,rng=copy(TESTRNG))
regressor_gmmr                  = Mlj.machine(model_gmmr, X, y)
(fitresult_gmmr, cache, report) = Mlj.fit(model_gmmr, 0, X, y)
yhat_gmmr                       = Mlj.predict(model_gmmr, fitresult_gmmr, X)
@test relative_mean_error(y,yhat_gmmr,normrec=true) < 0.3

ydouble = hcat(y,y)
model_gmmr2                      = MultitargetGaussianMixtureRegressor(n_classes=20,rng=copy(TESTRNG))
regressor_gmmr2                  = Mlj.machine(model_gmmr2, X, ydouble)
(fitresult_gmmr2, cache, report) = Mlj.fit(model_gmmr2, 0, X, ydouble)
yhat_gmmr2                       = Mlj.predict(model_gmmr2, fitresult_gmmr2, X)
@test relative_mean_error(ydouble,yhat_gmmr2,normrec=true) < 0.3

