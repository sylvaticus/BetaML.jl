using Test
#using Pipe
using Statistics, Random
using BetaML
import DecisionTree

TESTRNG = FIXEDRNG # This could change...


println("*** Testing Imputations...")

# ------------------------------------------------------------------------------
# Old API predictMissing

# ==================================
# New test
# ==================================
#println("Testing predictMissing...")
#X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
#out = predictMissing(X,3,mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
#@test isapprox(out.X̂[2,2],14.155186593170251)

#X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
#out2 = predictMissing(X,3,mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
#@test out2.X̂[2,2] ≈ 14.588514438886131

#X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
#out3 = predictMissing(X,3,mixtures=[FullGaussian() for i in 1:3],verbosity=NONE, initialisation_strategy="grid",rng=copy(TESTRNG))
#@test out3.X̂[2,2] ≈ 11.166652292936876

# ------------------------------------------------------------------------------

println("Testing FeatureBasedImputer...")

X = [2 missing 10; 20 40 100]
mod = FeatureBasedImputer()
fit!(mod,X)
x̂ = predict(mod)
@test x̂[1,2] == 40
@test typeof(x̂) == Matrix{Float64}
@test info(mod) == Dict{String,Any}("n_imputed_values" => 1)

X2 = [2 4 missing; 20 40 100]
x̂2 = predict(mod,X2)
@test x̂2[1,3] == 55.0
reset!(mod)

X = [2.0 missing 10; 20 40 100]
mod = FeatureBasedImputer(norm=1)
fit!(mod,X)
x̂ = predict(mod)
@test isapprox(x̂[1,2],4.044943820224719)
@test typeof(x̂) == Matrix{Float64}


# ------------------------------------------------------------------------------

println("Testing GMMImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]

mod = GMMImputer(mixtures=[SphericalGaussian() for i in 1:3],verbosity=NONE,initialisation_strategy="grid",rng=copy(TESTRNG))
x̂ = predict(mod)
@test x̂ == nothing
fit!(mod,X)
x̂ = predict(mod)
@test isapprox(x̂[2,2],14.155186593170251)

mod = GMMImputer(mixtures=[DiagonalGaussian() for i in 1:3],verbosity=NONE,initialisation_strategy="grid",rng=copy(TESTRNG))
fit!(mod,X)
x̂ = predict(mod)
@test isapprox(x̂[2,2],14.588514438886131)

mod = GMMImputer(mixtures=[FullGaussian() for i in 1:3],verbosity=NONE,initialisation_strategy="grid",rng=copy(TESTRNG))
fit!(mod,X)
x̂ = predict(mod)
@test x̂[2,2] ≈ 11.166652292936876

X = [2 missing 10; 2000 4000 10000; 2000 4000 10000; 3 5 12; 4 8 20; 2000 4000 8000; 1 5 8 ]
mod = GMMImputer(n_classes=2,rng=copy(TESTRNG),verbosity=NONE, initialisation_strategy="kmeans")
fit!(mod,X)
x̂ = predict(mod)
@test x̂[1,2] ≈ 6.0
infos = info(mod)
@test infos["n_imputed_values"] == 1 && infos["lL"] ≈ -163.12896063447343  && infos["BIC"] ≈ 351.5547532066659 && infos["AIC"] ≈ 352.25792126894686

X2 = [3 6 9; 2000 missing 10000; 1 2 5; 1500 3000 9000; 1.5 3 6]

fit!(mod,X2)
X̂2 =  predict(mod)
@test X̂2[1,1] == 3
@test X̂2[2,2] == 4000

X3 = [1 2 missing; 2 4 6]
X̂3 = predict(mod,X3)
@test X̂3[1,3] ≈ 6.666666666717062
reset!(mod)
#predict(mod,X3)

mod = GMMImputer(mixtures=DiagonalGaussian)
X2 = [3 6 9; 2000 missing 10000; 1 2 5; 1500 3000 9000; 1.5 3 6]
fit!(mod,X2)
X̂2 =  predict(mod)
@test typeof(X̂2) == Matrix{Float64}

# ------------------------------------------------------------------------------

println("Testing RFFImputer...")

X = [2 missing 10 "aaa" missing; 20 40 100 "gggg" missing; 200 400 1000 "zzzz" 1000]
mod = RFImputer(n_trees=30,forced_categorical_cols=[5],recursive_passages=3,multiple_imputations=10, rng=copy(TESTRNG),verbosity=NONE)
Xs_full = fit!(mod,X)

@test Xs_full[2][1,2] == 220
@test length(Xs_full) == 10

X = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ; 4 8 20; 1 2 5]
mod = RFImputer(multiple_imputations=10, rng=copy(TESTRNG),oob=true, verbosity=NONE)
fit!(mod,X)
vals = predict(mod)
nR,nC = size(vals[1])
medianValues = [median([v[r,c] for v in vals]) for r in 1:nR, c in 1:nC]
@test medianValues[1,2] == 4.0
infos = info(mod)
@test infos["n_imputed_values"] == 1
@test all(isequal.(infos["oob_errors"][1],[missing, 0.47355452303986306, missing]))

X = [2 4 10 "aaa" 10; 20 40 100 "gggg" missing; 200 400 1000 "zzzz" 1000]
mod = RFImputer(rng=copy(TESTRNG),verbosity=NONE)
fit!(mod,X)
X̂1 = predict(mod)
X̂1b =  predict(mod,X)
@test X̂1 == X̂1b

mod = RFImputer(rng=copy(TESTRNG),verbosity=NONE,cols_to_impute="all")
fit!(mod,X)
X2 = [2 4 10 missing 10; 20 40 100 "gggg" 100; 200 400 1000 "zzzz" 1000]
X̂2 =  predict(mod,X2)
@test X̂2[1,4] == "aaa"

# ------------------------------------------------------------------------------
println("Testing UniversalImputer...")

X = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ; 4 8 20; 1 2 5]
trng = copy(TESTRNG)
mod = UniversalImputer(estimator=[GMMRegressor1(rng=trng,verbosity=NONE),RandomForestEstimator(rng=trng,verbosity=NONE),RandomForestEstimator(rng=trng,verbosity=NONE)], multiple_imputations=10, recursive_passages=3, rng=copy(TESTRNG),verbosity=NONE,cols_to_impute="all")
fit!(mod,X)
vals = predict(mod)
nR,nC = size(vals[1])
meanValues = [mean([v[r,c] for v in vals]) for r in 1:nR, c in 1:nC]
@test meanValues[1,2] == 3.0

@test vals[1] == vals[10]

model_save("test.jld2"; mod)
modj  = model_load("test.jld2","mod")
valsj = predict(modj)
@test isequal(vals,valsj)

X = [2 missing 10; 2000 4000 1000; 2000 4000 10000; 3 5 12 ; 4 8 20; 1 2 5]
mod = UniversalImputer(multiple_imputations=10, recursive_passages=3, rng=copy(TESTRNG), verbosity=NONE)
fit!(mod,X)
vals = predict(mod)
nR,nC = size(vals[1])
meanValues = [mean([v[r,c] for v in vals]) for r in 1:nR, c in 1:nC]
@test meanValues[1,2] == 70.3

X = [2 4 10 "aaa" 10; 20 40 100 "gggg" missing; 200 400 1000 "zzzz" 1000]
trng = copy(TESTRNG)
#Random.seed!(trng,123)
mod = UniversalImputer(estimator=[DecisionTreeEstimator(rng=trng,verbosity=NONE),RandomForestEstimator(n_trees=1,rng=trng,verbosity=NONE),RandomForestEstimator(n_trees=1,rng=trng,verbosity=NONE),RandomForestEstimator(n_trees=1,rng=trng,verbosity=NONE),DecisionTreeEstimator(rng=trng,verbosity=NONE)],rng=trng,verbosity=NONE,cols_to_impute="all")

fit!(mod,X)
Random.seed!(trng,123)
X̂1  = predict(mod)
@test X̂1 == Any[2 4 10 "aaa" 10; 20 40 100 "gggg" 505; 200 400 1000 "zzzz" 1000]

Random.seed!(trng,123)
X̂1b =  predict(mod,X)
@test X̂1b == Any[2 4 10 "aaa" 10; 20 40 100 "gggg" 505; 200 400 1000 "zzzz" 1000]
@test X̂1 == X̂1b
X2 = [2 4 10 missing 10; 20 40 100 "gggg" 100; 200 400 1000 "zzzz" 1000]
X̂2 =  predict(mod,X2)
@test X̂2[1,4] == "aaa"

# ------------------------------
X = [1.0 2 missing 100; 3 missing missing 200; 4 5 6 300; missing 7 8 400; 9 10 11 missing; 12 13 missing missing; 14 15 missing 700; 16 missing missing 800;]

mod = UniversalImputer(estimator=DecisionTree.DecisionTreeRegressor(),rng=copy(TESTRNG),fit_function=DecisionTree.fit!,predict_function=DecisionTree.predict,recursive_passages=10)
Xfull = BetaML.fit!(mod,X)
@test size(Xfull) == (8,4) && typeof(Xfull) == Matrix{Float64}

mod = UniversalImputer(estimator=BetaML.DecisionTreeEstimator(),rng=copy(TESTRNG),recursive_passages=10)
Xfull2 = BetaML.fit!(mod,X)
@test size(Xfull) == (8,4) && typeof(Xfull) == Matrix{Float64}

mod = UniversalImputer(estimator=BetaML.DecisionTreeEstimator(),rng=copy(TESTRNG),missing_supported=true,recursive_passages=10)
Xfull3 = BetaML.fit!(mod,X)
@test size(Xfull) == (8,4) && typeof(Xfull) == Matrix{Float64}

X =  [     12      0.3       5      11;
           21      0.1       1     18;
            8  missing       9       9;
      missing      0.6       5       4;
      missing      0.4 missing       6;
           18  missing       1 missing;
            5      0.8 missing       15;
           10      0.7       8      11;]

mod = UniversalImputer(estimator=DecisionTree.DecisionTreeRegressor(),rng=copy(TESTRNG),fit_function=DecisionTree.fit!,predict_function=DecisionTree.predict,recursive_passages=10)
mod2 = UniversalImputer(rng=copy(TESTRNG),recursive_passages=10)

Xfull = BetaML.fit!(mod,X)          
Xfull2 = BetaML.fit!(mod2,X) 
@test Xfull[4,1] > 1     
@test Xfull[3,2] < 1   
@test Xfull[5,3] > 1
@test Xfull[6,4] > 10
@test Xfull2[4,1] < Xfull2[5,1]   
@test Xfull2[3,2] > Xfull2[6,2]  
@test Xfull2[5,3] < Xfull2[7,3]
@test Xfull2[6,4] > 10

# this would error, as multiple passsages
# predict(mod2,X)



# ------------------------------------------------------------------------------

println("Testing MLJ Interfaces...")

import MLJBase
const Mlj = MLJBase

println("Testing MLJ Interface for SimpleImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
Xt = Mlj.table(X)
model                       =  BetaML.Bmlj.SimpleImputer(norm=1)
modelMachine                =  Mlj.machine(model,Xt)
(fitResults, cache, report) =  Mlj.fit(model, 0, Xt)
XM                          =  Mlj.transform(model,fitResults,Xt)
x̂                           =  Mlj.matrix(XM)
@test isapprox(x̂[2,2],0.29546633468202105)
# Use the previously learned structure to imput missings..
Xnew_withMissing            = Mlj.table([1.5 missing; missing missing; missing -2.3; 5.1 -2.3; 1 2; 1 2; 1 2; 1 2; 1 2])
XDNew                       = Mlj.transform(model,fitResults,Xnew_withMissing)
XDMNew                      = Mlj.matrix(XDNew)
@test isapprox(XDMNew[2,2],x̂[2,2]) # position only matters

println("Testing MLJ Interface for GaussianMixtureImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
Xt = Mlj.table(X)
model                       =  BetaML.Bmlj.GaussianMixtureImputer(initialisation_strategy="grid",rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model,Xt)
(fitResults, cache, report) =  Mlj.fit(model, 0, Xt)
XM                          =  Mlj.transform(model,fitResults,Xt)
x̂                           =  Mlj.matrix(XM)
@test isapprox(x̂[2,2],14.736620020139028)
# Use the previously learned structure to imput missings..
Xnew_withMissing            = Mlj.table([1.5 missing; missing 38; missing -2.3; 5.1 -2.3])
XDNew                       = Mlj.transform(model,fitResults,Xnew_withMissing)
XDMNew                      = Mlj.matrix(XDNew)
@test isapprox(XDMNew[1,2],x̂[2,2])
model                       =  BetaML.Bmlj.GaussianMixtureImputer(initialisation_strategy="grid",rng=copy(TESTRNG), mixtures=BetaML.SphericalGaussian)
modelMachine                =  Mlj.machine(model,Xt)
(fitResults, cache, report) =  Mlj.fit(model, 0, Xt)
@test report["AIC"] < 100000

println("Testing MLJ Interface for RandomForestImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
Xt = Mlj.table(X)
model                       =  BetaML.Bmlj.RandomForestImputer(n_trees=40,rng=copy(TESTRNG))
modelMachine                =  Mlj.machine(model,Xt)
(fitResults, cache, report) =  Mlj.fit(model, 0, Xt)
XM                          =  Mlj.transform(model,fitResults,Xt)
x̂                           =  Mlj.matrix(XM)
@test isapprox(x̂[2,2],10.144666666666666)
# Use the previously learned structure to imput missings..
Xnew_withMissing            = Mlj.table([1.5 missing; missing 38; missing -2.3; 5.1 -2.3])
XDNew                       = Mlj.transform(model,fitResults,Xnew_withMissing)
XDMNew                      = Mlj.matrix(XDNew)
@test isapprox(XDMNew[1,2],x̂[2,2])

println("Testing MLJ Interface for GeneralImputer...")

X = [1 10.5;1.5 missing; 1.8 8; 1.7 15; 3.2 40; missing missing; 3.3 38; missing -2.3; 5.2 -2.4]
Xt = Mlj.table(X)
trng = copy(TESTRNG)
model                       =  BetaML.Bmlj.GeneralImputer(estimator=[GMMRegressor1(rng=copy(TESTRNG),verbosity=NONE),RandomForestEstimator(n_trees=40,rng=copy(TESTRNG),verbosity=NONE)],recursive_passages=2, missing_supported=true, rng = copy(TESTRNG))
modelMachine                =  Mlj.machine(model,Xt)
(fitResults, cache, report) =  Mlj.fit(model, 0, Xt)
XM                          =  Mlj.transform(model,fitResults,Xt)
x̂                           =  Mlj.matrix(XM)
@test isapprox(x̂[2,2],11.8) # not the same as RF because the oth columns are imputed too
# Use the previously learned structure to imput missings..
Xnew_withMissing            = Mlj.table([1.5 missing; missing 38; missing -2.3; 5.1 -2.3])
XDNew                       = Mlj.transform(model,fitResults,Xnew_withMissing)
XDMNew                      = Mlj.matrix(XDNew)
@test isapprox(XDMNew[1,2],x̂[2,2])


X =  [     12      0.3       5      11;
           21      0.1       1     18;
            8  missing       9       9;
      missing      0.6       5       4;
      missing      0.4 missing       6;
           18  missing       1 missing;
            5      0.8 missing       15;
           10      0.7       8      11;]


Xt = Mlj.table(X)
trng = copy(TESTRNG)
model                       =  BetaML.Bmlj.GeneralImputer(estimator=DecisionTree.DecisionTreeRegressor(), fit_function=DecisionTree.fit!,predict_function=DecisionTree.predict,recursive_passages=10, rng = copy(TESTRNG))
modelMachine                =  Mlj.machine(model,Xt)
(fitResults, cache, report) =  Mlj.fit(model, 0, Xt)
XM                          =  Mlj.transform(model,fitResults,Xt)
x̂                           =  Mlj.matrix(XM)
@test x̂[4,1] > 1     
@test x̂[3,2] < 1   
@test x̂[5,3] > 1
@test x̂[6,4] > 10



