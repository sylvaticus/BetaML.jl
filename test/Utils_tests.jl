using Test, Statistics, CategoricalArrays, Random, StableRNGs, DelimitedFiles
#using StableRNGs
#rng = StableRNG(123)
using BetaML
#import BetaML.Utils

TESTRNG = FIXEDRNG # This could change...
#TESTRNG = StableRNG(123)

println("*** Testing individual utility functions (module `Utils`)...")

# ==================================
# TEST 1: onehotencoder


@test BetaML.Utils.singleunique([[1,2,3],[7,8],[5]]) == [1,2,3,7,8,5]
@test BetaML.Utils.singleunique([1,4,5,4]) == [1,4,5]
@test BetaML.Utils.singleunique("aaa") == ["aaa"]


m  = OneHotEncoder()
x  = [3,6,3,4]
ŷ  = fit!(m,x)
@test ŷ == [ 1  0  0  0
             0  0  0  1
             1  0  0  0
             0  1  0  0
            ]
x2 = inverse_predict(m,ŷ)
@test x2 == x
x  = [3,6,missing,3,4]
m  = OneHotEncoder(categories=[3,4,7],handle_unknown="infrequent",other_categories_name=99)
ŷ  = fit!(m,x)
@test isequal(ŷ, [  true         false         false         false
                   false         false         false          true
                 missing       missing       missing       missing
                    true         false         false         false
                   false          true         false         false])
ŷ2  = predict(m)
ŷ3  = predict(m,x)
@test isequal(ŷ,ŷ2)
@test isequal(ŷ,ŷ3)
x2 = inverse_predict(m,ŷ)
@test isequal(x2,[3,99,missing,3,4])

# Testing OneHotEncoder with vector of dictionaries
y = ["e","g","b"]
ŷ = [Dict("a"=>0.2,"e"=>0.8),Dict("h"=>0.1,"e"=>0.2,"g"=>0.7),Dict("b"=>0.4,"e"=>0.6)]
m = OneHotEncoder(handle_unknown="infrequent")
yoh = fit!(m,y)
ŷoh = predict(m,ŷ)

@test ŷoh == [0.8  0.0  0.0  0.2
              0.2  0.7  0.0  0.1
              0.6  0.0  0.4  0.0]

# Testing ordinal encoder...
x  = ["3","6",missing,"3","4"]
m  = OrdinalEncoder(categories=["3","7","4"],handle_unknown="infrequent",other_categories_name="99")
ŷ  = fit!(m,x)
@test isequal(ŷ, [1,4,missing,1,3])
ŷ2  = predict(m)
ŷ3  = predict(m,x)
@test isequal(ŷ,ŷ2)
@test isequal(ŷ,ŷ3)
x2 = inverse_predict(m,ŷ)
@test isequal(x2,["3","99",missing,"3","4"]) 

x  = ["1","2","3"]
m  = OrdinalEncoder(handle_unknown="missing")
x̂  = fit!(m,x)
x̂m = collect(predict(m,["1","4","3"]))
@test x̂  == [1,2,3] && typeof(x̂) == Vector{Int64}
@test isequal(x̂m,[1,missing,3])

x  = [2,1,3]
m  = OrdinalEncoder()
x̂  = fit!(m,x)
@test x̂  == x

# ==================================
# NEW TEST
println("Testing findFirst/ findall / integerencoder / integerdecoder...")

a = ["aa","cc","cc","bb","cc","aa"]
A = [3 2 1; 2 3 1; 1 2 3]
@test findfirst("cc",a)  == 2
@test findall("cc",a)    == [2,3,5]
@test findfirst(2,A)     == (2,1)
@test findall(2,A)       == [(2,1),(1,2),(3,2)]
@test findfirst(2,A;returnTuple=false) == CartesianIndex(2,1)

# ==================================
# TEST 2: softmax
println("** Going through Test2 (softmax and other activation functions)...")
@test isapprox(softmax([2,3,4],β=0.1),[0.3006096053557272,0.3322249935333472,0.36716540111092544])

@test didentity(-1) == 1 && relu(-1) == 0 && drelu(-1) == 0 && celu(-1,α=0.1) == -0.09999546000702375 && dcelu(-1) == 0.36787944117144233 &&
      elu(-1,α=0.1) == -0.06321205588285576 && delu(-1) == 0.36787944117144233 && plu(-1) == -1 && dplu(-1) == 1

@test didentity(1) == 1 && relu(1) == 1 && drelu(1) == 1 && celu(1) == 1 && dcelu(1) == 1 &&
    elu(1) == 1 && delu(1) == 1 && plu(1) == 1 && dplu(1) == 1

@test dtanh(1) == 0.41997434161402614 && sigmoid(1) == 0.7310585786300049 == dsoftplus(1) && dsigmoid(1) == 0.19661193324148188 &&
      softplus(1) == 1.3132616875182228 && mish(1) == 0.8650983882673103 && dmish(1) == 1.0490362200997922

# ==================================
# TEST 3: autojacobian
println("** Going through Test3 (autojacobian)...")
@test isapprox(softmax([2,3,4],β=0.1),[0.3006096053557272,0.3322249935333472,0.36716540111092544])

#import BetaML.Utils: autojacobian
@test autojacobian(x -> (x[1]*2,x[2]*x[3]),[1,2,3]) == [2.0 0.0 0.0; 0.0 3.0 2.0]

b = softmax([2,3,4],β=1/2)
c = softmax([2,3.0000001,4],β=1/2)
# Skipping this test as gives problems in CI for Julia < 1.6
#autoGrad = autojacobian(x->softmax(x,β=1/2),[2,3,4])
realG2 = [(c[1]-b[1])*10000000,(c[2]-b[2])*10000000,(c[3]-b[3])*10000000]
#@test isapprox(autoGrad[:,2],realG2,atol=0.000001)
manualGrad = dsoftmax([2,3,4],β=1/2)
@test isapprox(manualGrad[:,2],realG2,atol=0.000001)

# Manual way is hundred of times faster
#@benchmark autojacobian(softmax2,[2,3,4])
#@benchmark dSoftMax([2,3,4],β=1/2)

# ==================================
# New test
println("** Testing cross-entropy...")
or = crossentropy([1.0,0,0],[0.8,0.001,0.001],weight = [2,1,1])
@test or ≈ 0.4462871026284194
d = dcrossentropy([1.0,0,0],[0.8,0.001,0.001],weight = [2,1,1])
δ = 0.001
dest = crossentropy([1.0,0,0],[0.8+δ,0.101,0.001],weight = [2,1,1])
@test isapprox(dest-or, d[1]*δ,atol=0.0001)

# ==================================
# New test
println("** Testing permutations...")
y = ["a","a","a","b","b","c","c","c"]
yp = getpermutations(y,keepStructure=true)
ypExpected = [
  ["a", "a", "a", "b", "b", "c", "c", "c"],
  ["a", "a", "a", "c", "c", "b", "b", "b"],
  ["b", "b", "b", "a", "a", "c", "c", "c"],
  ["b", "b", "b", "c", "c", "a", "a", "a"],
  ["c", "c", "c", "a", "a", "b", "b", "b"],
  ["c", "c", "c", "b", "b", "a", "a", "a"],
]
@test yp == ypExpected

# ==================================
# New test
println("** Going through testing accuracy...")


y = ["a","a","a","b","b","c","c","c"]
ŷ = ["b","b","a","c","c","a","a","c"]

accuracyConsideringClassLabels = accuracy(y,ŷ) # 2 out of 8
accuracyConsideringAnyClassLabel = accuracy(y,ŷ,ignorelabels=true) # 6 out of 8

@test accuracyConsideringClassLabels == 2/8
@test accuracyConsideringAnyClassLabel == 6/8

# with categorical arrays..
y = CategoricalArray(y)
ŷ = CategoricalArray(ŷ)

accuracyConsideringClassLabels = accuracy(y,ŷ) # 2 out of 8
accuracyConsideringAnyClassLabel = accuracy(y,ŷ,ignorelabels=true) # 6 out of 8

@test accuracyConsideringClassLabels == 2/8
@test accuracyConsideringAnyClassLabel == 6/8

x = [0.01 0.02 0.1 0.05 0.2 0.1  0.05 0.27  0.2;
     0.05 0.01 0.2 0.02 0.1 0.27 0.1  0.05  0.2]
y = [3,3]
@test [accuracy(y,x,tol=i) for i in 1:10] == [0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
x = [0.3 0.2 0.5; 0.5 0.25 0.25; 0.1 0.1 0.9]
y = [1, 3, 1]
@test accuracy(y,x) == 0.0
@test accuracy(y,x, ignorelabels=true) == 1.0


yest = [0 0.12; 0 0.9]
y    = [1,2]
accuracy(y,yest)
yest = [0.0  0.0  0.0;0.0  0.0  0.0;0.0  0.0  0.706138]
y    = [2,2,1]
accuracy(y,yest)

# ==================================
# New test
println("** Going through testing scaling...")


x = [1.1 4.1 8.1 missing 8; 2 4 9 7 2; 7 2 9 3 1]
m = Scaler(method=MinMaxScaler(),skip=[1,5])
ŷfit = fit!(m,x)
ŷ  = predict(m)
@test all(isequal.(ŷ, [ 1.1  1.0       0.0   missing  8.0
            2.0  0.9523809523809526  1.0  1.0       2.0
            7.0  0.0       1.0  0.0       1.0]))
ŷ1 = predict(m,x)
@test collect(skipmissing(ŷfit)) == collect(skipmissing(ŷ)) == collect(skipmissing(ŷ1))
x1 = inverse_predict(m,ŷ)
@test collect(skipmissing(x)) == collect(skipmissing(x1))
m2 = Scaler(MinMaxScaler(),skip=[1,5])
fit!(m2,x)
ŷ2  = predict(m2)
@test all(isequal.(ŷ, ŷ2))
m3 = Scaler(skip=[1,5])
fit!(m3,x)
ŷ3 = predict(m3)
@test ŷ3[2,2] == 0.6547832409557988
means = [mean(skipmissing(v)) for v in eachcol(ŷ3)]
vars = [var(skipmissing(v),corrected=false) for v in eachcol(ŷ3)]
@test all(isapprox.(means[2:4],0.0, atol=0.00000000001))
@test all(isapprox.(vars[2:4],1.0, atol=0.00000000001))
m4 = Scaler(StandardScaler(center=false,scale=false))
fit!(m4,x)
ŷ4  =predict(m4,x)
@test all(isequal.(ŷ4,x))

# Check it works with integer matrices..
X = [1 10 100; 2 20 200; 3 30 300]
m = Scaler()
Xs1 = fit!(m,X)
Xs2 = predict(m,X)
@test  Xs1 == Xs2 ≈ [-1.224744871391589 -1.224744871391589 -1.224744871391589; 0.0 0.0 0.0; 1.224744871391589 1.224744871391589 1.224744871391589]
@test inverse_predict(m,Xs1) == float.(X)
m2 = Scaler(MinMaxScaler())
Xs1b = fit!(m2,X)
Xs2b = predict(m2,X)
@test  Xs1b == Xs2b ≈ [0.0 0.0 0.0; 0.5 0.5 0.5; 1.0 1.0 1.0]
@test inverse_predict(m2,Xs1b) == float.(X)

# Check it works with arrays and can get back to matrix or vector
X = [1,2,3]
m = Scaler()
Xs1c = fit!(m,X)
Xs2c = predict(m,X)
@test Xs1c == Xs2c == Xs1[:,1]
inverse_predict(m,Xs1c) == X
m2 = Scaler(MinMaxScaler())
Xs1d = fit!(m2,X)
Xs2e = predict(m2,X)
@test  Xs1d == Xs2e ==  Xs1b[:,1]
@test inverse_predict(m2,Xs1d) == float.(X)

X2 = makematrix([1,2,3])
m = Scaler()
X2s1c = fit!(m,X2)
inverse_predict(m,X2s1c) == X2

# ==================================
# New test
println("** Testing batch()...")

@test size.(batch(10,3,rng=copy(TESTRNG)),1) == [3,3,3]
@test size.(batch(10,12,rng=copy(TESTRNG)),1) == [10]

# ==================================
# New test
println("** Testing relative_mean_error()...")

ŷ = [22 142 328; 3 9 31; 5 10 32; 3 10 36]
y = [20 140 330; 1 11 33; 3 8 30; 5 12 38]
p=2
(n,d) = size(y)

# case 1 - average of the relative error (records and dimensions normalised)
avgϵRel = sum(abs.((ŷ-y)./ y).^p)^(1/p)/(n*d)
#avgϵRel = (norm((ŷ-y)./ y,p)/(n*d))
relative_mean_error(y,ŷ,normdim=true,normrec=true,p=p) == avgϵRel
# case 2 - normalised by dimensions (i.e.  all dimensions play the same)
avgϵRel_byDim = (sum(abs.(ŷ-y) .^ (1/p),dims=1).^(1/p) ./ n) ./   (sum(abs.(y) .^ (1/p) ,dims=1) ./n)
avgϵRel = mean(avgϵRel_byDim)
@test relative_mean_error(y,ŷ,normdim=true,normrec=false,p=p) == avgϵRel
# case 3
avgϵRel_byRec = (sum(abs.(ŷ-y) .^ (1/p),dims=2).^(1/p) ./ d) ./   (sum(abs.(y) .^ (1/p) ,dims=2) ./d)
avgϵRel = mean(avgϵRel_byRec)
@test relative_mean_error(y,ŷ,normdim=false,normrec=true,p=p) == avgϵRel
# case 4 - average error relativized
avgϵRel = (sum(abs.(ŷ-y).^p)^(1/p) / (n*d)) / (sum( abs.(y) .^p)^(1/p) / (n*d))
#avgϵRel = (norm((ŷ-y),p)/(n*d)) / (norm(y,p) / (n*d))
@test relative_mean_error(y,ŷ,normdim=false,normrec=false,p=p) == avgϵRel


# ==================================
# New test
println("** Testing pca()...")

X = [1 8; 4.5 5.5; 9.5 0.5]
expectedX = [-4.58465   6.63182;-0.308999  7.09961; 6.75092   6.70262]
m = PCA(outdims=2)
fit!(m,X)
ŷ = predict(m)
@test isapprox(ŷ,expectedX,atol=0.00001) || isapprox(ŷ, (.- expectedX),atol=0.00001)

X = [1 10 100; 1.1 15 120; 0.95 23 90; 0.99 17 120; 1.05 8 90; 1.1 12 95]
m = PCA(max_unexplained_var=0.05)
fit!(m,X)
ŷ = predict(m)
@test 1-m.info["prop_explained_var"] ≈ 1.0556269747774571e-5
@test sum(ŷ) ≈ 662.3492034128955
ŷ2 = predict(m,X)
@test ŷ ≈ ŷ2

# ==================================
# New test
println("** Testing AutoEncoder...")

iris = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x    = convert(Array{Float64,2}, iris[:,1:4])
y    = convert(Array{String,1}, iris[:,5])

m    = AutoEncoder(outdims=2,rng=copy(TESTRNG))
x2   = fit!(m,x)
x2b  = predict(m)
x2c  = predict(m,x)
x2d  = fit!(m,x)
@test x2 == x2b == x2c
x̂    = inverse_predict(m,x2d)
@test relative_mean_error(x,x̂) < 0.04 

((xtrain,xtest),(x2train,x2test),(ytrain,ytest)) = partition([x,x2d,y],[0.8,0.2],rng=copy(TESTRNG))

mc1 = RandomForestEstimator(rng=copy(TESTRNG))
mc2 = RandomForestEstimator(rng=copy(TESTRNG))
fit!(mc1,xtrain,ytrain)
fit!(mc2,x2train,ytrain)
ŷ1 = mode(predict(mc1,xtest))
ŷ2 = mode(predict(mc2,x2test))

em1 = accuracy(ytest,ŷ1)
em2 = accuracy(ytest,ŷ2)
@test em2 > 0.85

x = [0.12 0.31 0.29 3.21 0.21;
     0.22 0.61 0.58 6.43 0.42;
     0.51 1.47 1.46 16.12 0.99;
     0.35 0.93 0.91 10.04 0.71;
     0.44 1.21 1.18 13.54 0.85];
m    = AutoEncoder(outdims=1,epochs=400)
x_reduced = fit!(m,x)
x̂ = inverse_predict(m,x_reduced)
info(m)["rme"]
hcat(x,x̂)

# ==================================
# New test
println("** Testing accuracy() on probs given in a dictionary...")

ŷ = Dict("Lemon" => 0.33, "Apple" => 0.2, "Grape" => 0.47)
y = "Lemon"
@test accuracy(y,ŷ) == 0
@test accuracy(y,ŷ,tol=2) == 1
y = "Grape"
@test accuracy(y,ŷ) == 1
y = "Something else"
@test accuracy(y,ŷ) == 0

ŷ1 = Dict("Lemon" => 0.6, "Apple" => 0.4)
ŷ2 = Dict("Lemon" => 0.33, "Apple" => 0.2, "Grape" => 0.47)
ŷ3 = Dict("Lemon" => 0.2, "Apple" => 0.5, "Grape" => 0.3)
ŷ4 = Dict("Apple" => 0.2, "Grape" => 0.8)
ŷ = [ŷ1,ŷ2,ŷ3,ŷ4]
y = ["Lemon","Lemon","Apple","Lemon"]
@test accuracy(y,ŷ) == 0.5
@test accuracy(y,ŷ,tol=2) == 0.75

# ==================================
# New test
println("** Testing mode(dicts)...")
ŷ1 = Dict("Lemon" => 0.6, "Apple" => 0.4)
ŷ2 = Dict("Lemon" => 0.33, "Apple" => 0.2, "Grape" => 0.47)
ŷ3 = Dict("Lemon" => 0.2, "Apple" => 0.5, "Grape" => 0.3)
ŷ4 = Dict("Apple" => 0.2, "Grape" => 0.8)
ŷ5 = Dict("Lemon" => 0.4, "Grape" => 0.4, "Apple" => 0.2)
ŷ = [ŷ1,ŷ2,ŷ3,ŷ4,ŷ5]
@test mode(ŷ,rng=copy(TESTRNG)) == ["Lemon","Grape","Apple","Grape","Lemon"]

y1 = [1,4,2,5]
y2 = [2,6,6,4]
y = [y1,y2]
@test mode(y,rng=copy(TESTRNG)) == [4,3]
y = vcat(y1',y2')
mode(y,rng=copy(TESTRNG)) == [4,3]

# ==================================
# New test

println("** Testing ConfusionMatrix()...")

y = ["Lemon","Lemon","Apple","Grape","Lemon"]
ŷ = ["Lemon","Grape","Apple","Grape","Lemon"]
cm = ConfusionMatrix()
scores1 = fit!(cm,y,ŷ)
scores2 = predict(cm)
res = info(cm)
@test res["scores"]  == [2 0 1; 0 1 0; 0 0 1]
@test res["normalised_scores"] == scores1 == scores2 ≈ [ 0.6666666666666666 0.0 0.3333333333333333; 0.0 1.0 0.0; 0.0 0.0 1.0]
@test res["tp"] == [2,1,1] && res["tn"] == [2,4,3] && res["fp"] == [0,0,1] && res["fn"] == [1, 0, 0]
parameters(cm)

# Checking multiple training equal to just training on full data
scores2 = fit!(cm,y,ŷ)
res2 = info(cm)

y3         =  vcat(y,y)
ŷ3         =  vcat(ŷ,ŷ)
cm3        =  ConfusionMatrix()
scores3    =  fit!(cm3,y3,ŷ3)
res3       =  info(cm3)
@test res2 == res3

# Checking infrequent setting
cm = ConfusionMatrix(categories=["Lemon","Grape"],handle_unknown="infrequent")
scores = fit!(cm,y,ŷ)
res = info(cm)


# Example from https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
y = [0,1,2,2,0]
ŷ = [0,0,2,1,0]

cm = ConfusionMatrix()
fit!(cm,y,ŷ)
res = info(cm)
@test res["precision"] ≈ [0.6666666666666666, 0.0, 1.0]
@test res["recall"] ≈ [1.0, 0.0, 0.5]
@test res["specificity"] ≈ [0.6666666666666666, 0.75, 1.0]
@test res["f1score"] ≈ [0.8, 0.0, 0.6666666666666666]
@test res["mean_precision"] == (0.5555555555555555, 0.6666666666666666)
@test res["mean_recall"] == (0.5, 0.6)
@test res["mean_specificity"] == (0.8055555555555555, 0.8166666666666667)
@test res["mean_f1score"] == (0.48888888888888893, 0.5866666666666667)
@test res["accuracy"] == 0.6
@test res["misclassification"] == 0.4

#original_stdout = stdout
#(rd, wr) = redirect_stdout()

#redirect_stdout(original_stdout)
#close(wr)

cm = ConfusionMatrix(categories_names = Dict(0=>"0",1=>"1",2=>"2"))
fit!(cm,y,ŷ)
res = info(cm)
@test res["categories"] == ["0","1","2"]

# ==================================
# New test
println("** Testing class_counts()...")

@test class_counts_with_labels(["a","b","a","c","d"]) == Dict("a"=>2,"b"=>1,"c"=>1,"d"=>1)
@test class_counts_with_labels(['a' 'b'; 'a' 'c';'a' 'b']) == Dict(['a', 'b'] => 2,['a', 'c'] => 1)
@test class_counts(["a","b","a","c","d"],classes=["a","b","c","d","e"]) == [2,1,1,1,0]
@test collect(class_counts(['a' 'b'; 'a' 'c';'a' 'b'])) in [[2,1],[1,2]] # Order doesn't matter


# ==================================
# New test
println("** Testing gini()...")
@test gini(['a','b','c','c']) == 0.625 # (1/4) * (3/4) + (2/4) * (2/4) + (1/4)*(3/4)
@test gini([1 10; 2 20; 3 30; 2 20]) == 0.625


#a = -0.01*log2(0.01)
#b = -0.99*log2(0.99)
#c = a+b

#A = -0.49*log2(0.49)
#B = -0.51*log2(0.51)
#C = A+B

# ==================================
# New test
println("** Testing entropy()...")
@test isapprox(entropy([1,2,3]), 1.584962500721156) #-(1/3)*log2(1/3)-(1/3)*log2(1/3)-(1/3)*log2(1/3)
@test isapprox(entropy([1 10; 2 20; 3 30]), 1.584962500721156)
#par = entropy([1,1,1,1,1,0,0,0,0,0,0,0,0,0])
#k1 = entropy([1,1,1,0,0,0])
#k2 = entropy([1,1,0,0,0,0,0,0])
#kidsEntropy = k1 *(6/14) + k2*(8/14)
#gain = par - kidsEntropy
#entropy([0,1,2,3,4,5,6,7])

# ==================================
# New test
println("** Testing mean_dicts()...")

a = Dict('a'=> 0.2,'b'=>0.3,'c'=>0.5)
b = Dict('a'=> 0.3,'b'=>0.1,'d'=>0.6)
c = Dict('b'=>0.6,'e'=>0.4)
d = Dict('a'=>1)
dicts = [a,b,c,d]
@test mean_dicts(dicts) == Dict('a' => 0.375,'c' => 0.125,'d' => 0.15,'e' => 0.1,'b' => 0.25)
@test mean_dicts(dicts,weights=[1,1,97,1]) == Dict('a' => 0.015, 'b' => 0.586, 'c' => 0.005, 'd' =>0.006, 'e' =>0.388)

a = Dict(1=> 0.1,2=>0.4,3=>0.5)
b = Dict(4=> 0.3,1=>0.1,2=>0.6)
c = Dict(5=>0.6,4=>0.4)
d = Dict(2=>1)
dicts = [a,b,c,d]
@test mean_dicts(dicts)  == Dict(4 => 0.175, 2 => 0.5, 3 => 0.125, 5 => 0.15, 1 => 0.05)

# ==================================
# New test
println("** Testing partition()...")

m1 = [1:10 11:20 31:40]
m2 = convert(Array{Float64,2},[41:50 51:60])
m3 = makematrix(collect(61:70))
m4 = collect(71:80)
parts = [0.33,0.27,0.4]
out = partition([m1,m2,m3,m4],parts,shuffle=true,rng=copy(TESTRNG))
@test size(out,1) == 4 && size(out[1][3]) == (4,3)
x = [1:10 11:20]
y = collect(31:40)
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3], shuffle=false,rng=copy(TESTRNG))
@test xtest[2,2] == 19 && ytest[2] == 39

m1 = [1:2 11:12 31:32 41:42 51:52 61:62]
out = partition(m1,[0.7,0.3],dims=2,rng=copy(TESTRNG))
@test out == [[31 1 51 61; 32 2 52 62],[11 41; 12 42]]

# Testing not numeric matrices
ms = [[11:16 string.([21:26;])],[31:36;]]
out = partition(ms,[0.7,0.3],dims=1,rng=copy(TESTRNG))
@test out[1][2] == [12 "22"; 14 "24"] && out[2][2]== [32,34]

# ==================================
# New test
println("** Testing KFold sampler with a single X matrix...")
data           = [11:13 21:23 31:33 41:43 51:53 61:63]
sampleIterator = SamplerWithData(KFold(nsplits=3,nrepeats=1,shuffle=true,rng=copy(TESTRNG)),data,2)
for (i,d) in enumerate(sampleIterator)
    if i == 1
        @test d[1][1] == [51 61 21 41; 52 62 22 42; 53 63 23 43] && d[2][1] == [31 11; 32 12; 33 13]
    elseif i == 2
        @test d[1][1] == [31 11 21 41; 32 12 22 42; 33 13 23 43] && d[2][1] == [51 61; 52 62; 53 63]
    elseif i ==3
        @test d[1][1] == [31 11 51 61; 32 12 52 62; 33 13 53 63] && d[2][1] == [21 41; 22 42; 23 43]
    else
        @error "There shoudn't be more than 3 iterations for this iterator!"
    end
end

println("** Testing KFold sampler with multiple matrices...")
data           = [[11:16 string.([21:26;])],[31:36;]]
sampleIterator = SamplerWithData(KFold(nsplits=3,nrepeats=2,shuffle=false,rng=copy(TESTRNG)),data,1)
for (i,d) in enumerate(sampleIterator)
    local xtrain, ytrain, xval, yval
    (xtrain,ytrain),(xval,yval) = d
    if i == 1
        @test xtrain == [13 "23"; 14 "24"; 15 "25"; 16 "26"] && ytrain == [33, 34, 35, 36] && xval == [11 "21"; 12 "22"] && yval == [31, 32]
    elseif i == 5
        @test xtrain == [11 "21"; 12 "22"; 15 "25"; 16 "26"] && ytrain == [31, 32, 35, 36] && xval == [13 "23"; 14 "24"]  && yval == [33, 34]
    elseif i > 6
        @error "There shoudn't be more than 6 iterations for this iterator!"
    end
end


println("** Testing cross_validation...")

X = [11:19 21:29 31:39 41:49 51:59 61:69]
Y = [1:9;]
sampler = KFold(nsplits=3,nrepeats=1,shuffle=true,rng=copy(TESTRNG))
(μ,σ) = cross_validation([X,Y],sampler) do trainData,valData,rng
        (xtrain,ytrain) = trainData; (xval,yval) = valData
        rfmod = RandomForestEstimator(n_trees=30,rng=rng)
        fit!(rfmod,xtrain,ytrain)
        predictions = predict(rfmod,xval)
        ϵ = relative_mean_error(yval,predictions,normrec=false)
        return ϵ
    end

@test (μ,σ) == (0.3202242202242202, 0.04307662219315022)

println("** Testing autotuning...")

X = [11:99 99:-1:11]
y = collect(111:199)
tunemethod = GridSearch(hpranges=Dict("max_depth" =>[5,10,nothing], "min_gain"=>[0.0, 0.1, 0.5], "min_records"=>[2,3,5],"max_features"=>[nothing,5,10,30]),multithreads=true)
m = DecisionTreeEstimator(verbosity=NONE,rng=copy(TESTRNG),autotune=true,tunemethod=tunemethod)
hyperparameters(m).tunemethod.res_share=0.8
fit!(m,X,y)
opthp = hyperparameters(m)
#println("Test...")
#println(opthp)
#dump(opthp)
#@test ((opthp.max_depth == 10) && (opthp.min_gain==0.0) && (opthp.min_records==2) && (opthp.max_features==5))
ŷ = predict(m,X)
relative_mean_error(y,ŷ,normrec=false)
@test relative_mean_error(y,ŷ,normrec=false) <= 0.005 # ≈ 0.0023196810438564698

X = [11:99 99:-1:11]
y = collect(111:199)
tunemethod = SuccessiveHalvingSearch(hpranges=Dict("n_trees"=>[5,10,20,30],"max_depth" =>[5,10,nothing], "min_gain"=>[0.0, 0.1, 0.5], "min_records"=>[2,3,5],"max_features"=>[nothing,5,10,30]),multithreads=false)
m = RandomForestEstimator(verbosity=NONE,rng=copy(TESTRNG),autotune=false,tunemethod=tunemethod)
hyperparameters(m).tunemethod.res_shares=[0.3,0.6,0.8]
fit!(m,X,y)
opthp = hyperparameters(m)
#println("Test...")
#println(opthp)
#dump(opthp)
#@test ((opthp.max_depth == 10) && (opthp.min_gain==0.5) && (opthp.min_records==2) && (opthp.max_features==nothing))
ŷ = predict(m,X) 
@test relative_mean_error(y,ŷ,normrec=false) <= 0.002 # ≈ 0.0023196810438564698



# ==================================
# New test
println("** Testing consistent_shuffle()...")

a = [1 2 3; 10 20 30; 100 200 300; 1000 2000 3000; 10000 20000 30000]; b = [4,40,400,4000,40000]
out = consistent_shuffle([a,b],rng=copy(FIXEDRNG))
@test out[1] ==  [1000 2000 3000; 10000 20000 30000; 10 20 30; 1 2 3; 100 200 300] && out[2] == [4000, 40000, 40, 4, 400]
out2 = consistent_shuffle(copy(FIXEDRNG),[a,b])
@test out2 == out


a = [1 2 3 4 5; 10 20 30 40 50]; b = [100 200 300 400 500]
out = consistent_shuffle([a,b],rng=copy(FIXEDRNG),dims=2)
@test out[1] == [4 5 2 1 3; 40 50 20 10 30] && out[2] == [400 500 200 100 300]

# ==================================
# New test
println("** Testing generate_parallel_rngs()...")
x = rand(copy(TESTRNG),100)

function innerFunction(bootstrappedx; rng=Random.GLOBAL_RNG)
     sum(bootstrappedx .* rand(rng) ./ 0.5)
end
function outerFunction(x;rng = Random.GLOBAL_RNG)
    masterSeed = rand(rng,100:typemax(Int64)) # important: with some RNG it is important to do this before the generate_parallel_rngs to guarantee independance from number of threads
    rngs       = generate_parallel_rngs(rng,Threads.nthreads()) # make new copy instances
    results    = Array{Float64,1}(undef,30)
    Threads.@threads for i in 1:30
        tsrng = rngs[Threads.threadid()]    # Thread safe random number generator: one RNG per thread
        Random.seed!(tsrng,masterSeed+i*10) # But the seeding depends on the i of the loop not the thread: we get same results indipendently of the number of threads
        toSample = rand(tsrng, 1:100,100)
        bootstrappedx = x[toSample]
        innerResult = innerFunction(bootstrappedx, rng=tsrng)
        results[i] = innerResult
    end
    overallResult = mean(results)
    return overallResult
end


# Different sequences..
@test outerFunction(x) != outerFunction(x)

# Different values, but same sequence
mainRng = copy(TESTRNG)
a = outerFunction(x, rng=mainRng)
b = outerFunction(x, rng=mainRng)

mainRng = copy(TESTRNG)
A = outerFunction(x, rng=mainRng)
B = outerFunction(x, rng=mainRng)

@test a != b && a == A && b == B


# Same value at each call
a = outerFunction(x,rng=copy(TESTRNG))
b = outerFunction(x,rng=copy(TESTRNG))
@test a == b



# ==================================
# New test
println("** Testing pool1d()...")
x = [1,2,3,4,5,6]
poolsize = 3

out = pool1d(x,poolsize)
@test out == [2.0,3.0,4.0,5.0]
out = pool1d(x,poolsize;f=maximum)
@test out == [3,4,5,6]



# ==================================
# New test
println("** Testing get_parametric_types() (this uses Julia internals!)...")

o = [1,2,3]
T = get_parametric_types(o)[1]
@test T == Int64

println("Testing pairwise and silhouette...")

x  = [1 2 3 3; 1.2 3 3.1 3.2; 2 4 6 6.2; 2.1 3.5 5.9 6.3]
pd = pairwise(x)
s1 = silhouette(pd,[1,2,2,2])
@test s1 == [0.0, -0.7590778795827623, 0.5030093571833065, 0.4936350560759424]
s2  = silhouette(pd,[1,1,2,2])
@test s2 ==  [0.7846062151896173, 0.7590778795827623, 0.8860577617518799, 0.8833580446365146]

#=
using Random, StableRNGs
rDiff(rngFunction,seedBase,seedDiff,repetitions) = norm(rand(rngFunction(seedBase),repetitions) .- rand(rngFunction(seedBase+seedDiff),repetitions))/repetitions

# Seed base 1000: ok
rDiff(StableRNG,1000,1,100000)          # 0.00129
rDiff(StableRNG,1000,10,100000)         # 0.00129
rDiff(StableRNG,1000,1000,100000)       # 0.00129

rDiff(MersenneTwister,1000,1,100000)    # 0.00129
rDiff(MersenneTwister,1000,10,100000)   # 0.00129
rDiff(MersenneTwister,1000,1000,100000) # 0.00129

# Seed base 10: Still ok
rDiff(StableRNG,10,1,100000)            # 0.00129
rDiff(StableRNG,10,10,100000)           # 0.00129
rDiff(StableRNG,10,1000,100000)         # 0.00129

rDiff(MersenneTwister,10,1,100000)      # 0.00129
rDiff(MersenneTwister,10,10,100000)     # 0.00129
rDiff(MersenneTwister,10,1000,100000)   # 0.00129

# Seed base 1: We start seeing problems for StableRNG..
rDiff(StableRNG,1,1,100000)             # 0.00125  <--
rDiff(StableRNG,1,10,100000)            # 0.00129
rDiff(StableRNG,1,1000,100000)          # 0.00129

rDiff(MersenneTwister,1,1,100000)       # 0.00129
rDiff(MersenneTwister,1,10,100000)      # 0.00129
rDiff(MersenneTwister,1,1000,100000)    # 0.00129

# Seed base 0: Unexpected results for for StableRNG..
rDiff(StableRNG,0,1,100000)             # 0.00105 <----------
rDiff(StableRNG,0,2,100000)             # 0.00116 <-----
rDiff(StableRNG,0,5,100000)             # 0.00123 <---
rDiff(StableRNG,0,10,100000)            # 0.00126 <--
rDiff(StableRNG,0,1000,100000)          # 0.00129

rDiff(MersenneTwister,0,1,100000)       # 0.00130 <-
rDiff(MersenneTwister,0,2,100000)       # 0.00129
rDiff(MersenneTwister,0,5,100000)       # 0.00129
rDiff(MersenneTwister,0,10,100000)      # 0.00129
rDiff(MersenneTwister,0,1000,100000)    # 0.00129
=#
