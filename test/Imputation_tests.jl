using Test
#using Pipe
using Statistics
using BetaML

import MLJBase
const Mlj = MLJBase

TESTRNG = FIXEDRNG # This could change...


println("*** Testing Imputations...")

# ------------------------------------------------------------------------------

println("Testing MeanImputer...")

X = [2 missing 10; 20 40 100]
mod = MeanImputer()
fit!(mod,X)
x̂ = predict(mod)
@test x̂[1,2] == 40
@test typeof(x̂) == Matrix{Real}
@test info(mod) == (fitted=1,nImputedValues =1,)

X = [2.0 missing 10; 20 40 100]
mod = MeanImputer(normaliseRecords=true)
fit!(mod,X)
x̂ = predict(mod)
@test isapprox(x̂[1,2],4.044943820224719)
@test typeof(x̂) == Matrix{Float64}


# ------------------------------------------------------------------------------

println("Testing GMMImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]

mod = GMMImputer(mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE,initStrategy="grid",rng=copy(TESTRNG))
x̂ = predict(mod)
@test x̂ == nothing
fit!(mod,X)
x̂ = predict(mod)
@test isapprox(x̂[2,2],14.155186593170251)

mod = GMMImputer(mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE,initStrategy="grid",rng=copy(TESTRNG))
fit!(mod,X)
x̂ = predict(mod)
@test isapprox(x̂[2,2],14.588514438886131)

mod = GMMImputer(mixtures=[FullGaussian() for i in 1:3],verbosity=NONE,initStrategy="grid",rng=copy(TESTRNG))
fit!(mod,X)
x̂ = predict(mod)
@test x̂[2,2] ≈ 11.166652292936876

X = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ]
mod = GMMImputer(K=2,multipleImputations=3,rng=copy(TESTRNG),verbosity=NONE, initStrategy="kmeans")
fit!(mod,X)
x̂ = predict(mod)
@test x̂[1][1,2] = x̂[2][1,2] = x̂[3][1,2] ≈ 6.281803477634331
infos = info(mod)
@test infos.fitted == true && infos.nImputedValues == 1 && infos.lL[1] ≈ -58.47338193522323  && infos.BIC[1] ≈ 134.96859056500503 && infos.AIC[1] ≈ 142.94676387044646

# ------------------------------------------------------------------------------

println("Testing RFFImputer...")

X = [2 missing 10 "aaa" missing; 20 40 100 "gggg" missing; 200 400 1000 "zzzz" 1000]
mod = RFImputer(forcedCategoricalCols=[5],recursivePassages=3,multipleImputations=10, rng=copy(TESTRNG))
fit!(mod,X)
@test predict(mod)[1][1,2] == predict(mod)[3][1,2] == 400
@test predict(mod)[2][1,2] == 40

X = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ; 4 8 20; 1 2 5]
mod = RFImputer(multipleImputations=10, rng=copy(TESTRNG),oob=true)
fit!(mod,X)
vals = predict(mod)
nR,nC = size(vals[1])
medianValues = [median([v[r,c] for v in vals]) for r in 1:nR, c in 1:nC]
@test medianValues[1,2] == 4.0
infos = info(mod)
@test infos.nImputedValues == 1
@test infos.oob[1] ≈ [0.6482801664254283, 0.5447602979262367, 1.4813804498107928]


# ------------------------------------------------------------------------------

println("Testing MLJ Interfaces...")

# ------------------------------------------------------------------------------

println("Testing MLJ Interface for BetaMLGMMImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
X = Mlj.table(X)
model                       =  BetaMLGMMImputer(initStrategy="grid",rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model,X)
(fitResults, cache, report) =  Mlj.fit(model, 0, X)
x̂                           =  Mlj.matrix(fitResults)
@test isapprox(x̂[2,2],14.588514438886131)