using Test, Statistics, CategoricalArrays, Random, StableRNGs
#using StableRNGs
#rng = StableRNG(123)
using BetaML
#import BetaML.Utils

TESTRNG = FIXEDRNG # This could change...
#TESTRNG = StableRNG(123)

println("*** Testing individual utility functions (module `Utils`)...")

# ==================================
# TEST 1: oneHotEncoder
println("Going through Test1 (oneHotEncoder)...")

a = [[1,3,4],[1,4,2,2,3],[2,3]]
b = [2,1,5,2,1]
c = 2
ae = oneHotEncoder(a;d=5,count=true)
be = oneHotEncoder(b;d=6,count=true)
ce = oneHotEncoder(c;d=6)
@test sum(ae*be*ce') == 4

a = [["aa","ee","dd","aa"],["aa","dd","bb","bb","dd"],["bb","ee"]]
b = ["a","d","b","b","d"]
c = "dd"
ae = oneHotEncoder(a,factors=["aa","bb","cc","dd","ee","ff"],count=true)
be = oneHotEncoder(b,factors=["a","b","c","d","e","f"],count=true)
ce = oneHotEncoder(c,factors=["aa","bb","cc","dd","ee","ff"],count=false)
@test sum(ae * be') + ce[1,2] == 15

ae = oneHotEncoder(a,count=true)
be = oneHotEncoder(b,count=false)
ce = oneHotEncoder(c,count=true)
@test sum(ae)+sum(size(be))+sum(ce) == 20

@test BetaML.Utils.singleUnique([[1,2,3],[7,8],[5]]) == [1,2,3,7,8,5]
@test BetaML.Utils.singleUnique([1,4,5,4]) == [1,4,5]
@test BetaML.Utils.singleUnique("aaa") == ["aaa"]

@test BetaML.Utils.oneHotEncoderRow(4,d=5) == [0,0,0,1,0]
@test BetaML.Utils.oneHotEncoderRow("b",factors=["a","b","c"]) == [0,1,0]
@test BetaML.Utils.oneHotEncoderRow(["b","c","c"],factors=["a","b","c"],count=true) == [0,1,2]

# ==================================
# NEW TEST
println("Testing findFirst/ findall / integerEncoder / integerDecoder...")

a = ["aa","cc","cc","bb","cc","aa"]
A = [3 2 1; 2 3 1; 1 2 3]
@test findfirst("cc",a)  == 2
@test findall("cc",a)    == [2,3,5]
@test findfirst(2,A)     == (2,1)
@test findall(2,A)       == [(2,1),(1,2),(3,2)]
@test findfirst(2,A;returnTuple=false) == CartesianIndex(2,1)

factors =  ["aa","bb","cc"]
encoded  = integerEncoder(a,factors=factors)
@test encoded == [1,3,3,2,3,1]
decoded  = integerDecoder(encoded,factors)
@test a == decoded


# ==================================
# TEST 2: softMax
println("** Going through Test2 (softmax and other activation functions)...")
@test isapprox(softmax([2,3,4],β=0.1),[0.3006096053557272,0.3322249935333472,0.36716540111092544])

@test didentity(-1) == 1 && relu(-1) == 0 && drelu(-1) == 0 && celu(-1,α=0.1) == -0.09999546000702375 && dcelu(-1) == 0.36787944117144233 &&
      elu(-1,α=0.1) == -0.06321205588285576 && delu(-1) == 0.36787944117144233 && plu(-1) == -1 && dplu(-1) == 1

@test didentity(1) == 1 && relu(1) == 1 && drelu(1) == 1 && celu(1) == 1 && dcelu(1) == 1 &&
    elu(1) == 1 && delu(1) == 1 && plu(1) == 1 && dplu(1) == 1

@test dtanh(1) == 0.41997434161402614 && sigmoid(1) == 0.7310585786300049 == dsoftplus(1) && dsigmoid(1) == 0.19661193324148188 &&
      softplus(1) == 1.3132616875182228 && mish(1) == 0.8650983882673103 && dmish(1) == 1.0490362200997922

# ==================================
# TEST 3: autoJacobian
println("** Going through Test3 (autoJacobian)...")
@test isapprox(softmax([2,3,4],β=0.1),[0.3006096053557272,0.3322249935333472,0.36716540111092544])

#import BetaML.Utils: autoJacobian
@test autoJacobian(x -> (x[1]*2,x[2]*x[3]),[1,2,3]) == [2.0 0.0 0.0; 0.0 3.0 2.0]

b = softmax([2,3,4],β=1/2)
c = softmax([2,3.0000001,4],β=1/2)
# Skipping this test as gives problems in CI for Julia < 1.6
#autoGrad = autoJacobian(x->softmax(x,β=1/2),[2,3,4])
realG2 = [(c[1]-b[1])*10000000,(c[2]-b[2])*10000000,(c[3]-b[3])*10000000]
#@test isapprox(autoGrad[:,2],realG2,atol=0.000001)
manualGrad = dsoftmax([2,3,4],β=1/2)
@test isapprox(manualGrad[:,2],realG2,atol=0.000001)

# Manual way is hundred of times faster
#@benchmark autoJacobian(softMax2,[2,3,4])
#@benchmark dSoftMax([2,3,4],β=1/2)

# ==================================
# New test
println("** Testing cross-entropy...")
or = crossEntropy([0.8,0.001,0.001],[1.0,0,0],weight = [2,1,1])
@test or ≈ 0.4462871026284194
d = dCrossEntropy([0.8,0.001,0.001],[1.0,0,0],weight = [2,1,1])
δ = 0.001
dest = crossEntropy([0.8+δ,0.101,0.001],[1.0,0,0],weight = [2,1,1])
@test isapprox(dest-or, d[1]*δ,atol=0.0001)

# ==================================
# New test
println("** Testing permutations...")
y = ["a","a","a","b","b","c","c","c"]
yp = getPermutations(y,keepStructure=true)
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

accuracyConsideringClassLabels = accuracy(ŷ,y) # 2 out of 8
accuracyConsideringAnyClassLabel = accuracy(ŷ,y,ignoreLabels=true) # 6 out of 8

@test accuracyConsideringClassLabels == 2/8
@test accuracyConsideringAnyClassLabel == 6/8

# with categorical arrays..
y = CategoricalArray(y)
ŷ = CategoricalArray(ŷ)

accuracyConsideringClassLabels = accuracy(ŷ,y) # 2 out of 8
accuracyConsideringAnyClassLabel = accuracy(ŷ,y,ignoreLabels=true) # 6 out of 8

@test accuracyConsideringClassLabels == 2/8
@test accuracyConsideringAnyClassLabel == 6/8

x = [0.01 0.02 0.1 0.05 0.2 0.1  0.05 0.27  0.2;
     0.05 0.01 0.2 0.02 0.1 0.27 0.1  0.05  0.2]
y = [3,3]
@test [accuracy(x,y,tol=i) for i in 1:10] == [0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
x = [0.3 0.2 0.5; 0.5 0.25 0.25; 0.1 0.1 0.9]
y = [1, 3, 1]
@test accuracy(x,y) == 0.0
@test accuracy(x,y, ignoreLabels=true) == 1.0


yest = [0 0.12; 0 0.9]
y    = [1,2]
accuracy(yest,y)
yest = [0.0  0.0  0.0;0.0  0.0  0.0;0.0  0.0  0.706138]
y    = [2,2,1]
accuracy(yest,y)

# ==================================
# New test
println("** Going through testing scaling...")

skip = [1,4,5]
x = [1.1 4.1 8.1 3 8; 2 4 9 7 2; 7 2 9 3 1]
scaleFactors = getScaleFactors(x,skip=skip)
y  = scale(x,scaleFactors)
y2 = scale(x)
x2 = copy(x)
scale!(x2)
@test y2 == x2
@test all((sum(mean(y,dims=1)), sum(var(y,corrected=false,dims=1)) ) .≈ (11.366666666666667, 21.846666666666668))
x3 = scale(y,scaleFactors,rev=true)
@test x3 == x

# ==================================
# New test
println("** Testing batch()...")

@test size.(batch(10,3,rng=copy(TESTRNG)),1) == [3,3,3]
@test size.(batch(10,12,rng=copy(TESTRNG)),1) == [10]

# ==================================
# New test
println("** Testing meanRelError()...")

ŷ = [22 142 328; 3 9 31; 5 10 32; 3 10 36]
y = [20 140 330; 1 11 33; 3 8 30; 5 12 38]
p=2
(n,d) = size(y)

# case 1 - average of the relative error (records and dimensions normalised)
avgϵRel = sum(abs.((ŷ-y)./ y).^p)^(1/p)/(n*d)
#avgϵRel = (norm((ŷ-y)./ y,p)/(n*d))
meanRelError(ŷ,y,normDim=true,normRec=true,p=p) == avgϵRel
# case 2 - normalised by dimensions (i.e.  all dimensions play the same)
avgϵRel_byDim = (sum(abs.(ŷ-y) .^ (1/p),dims=1).^(1/p) ./ n) ./   (sum(abs.(y) .^ (1/p) ,dims=1) ./n)
avgϵRel = mean(avgϵRel_byDim)
@test meanRelError(ŷ,y,normDim=true,normRec=false,p=p) == avgϵRel
# case 3
avgϵRel_byRec = (sum(abs.(ŷ-y) .^ (1/p),dims=2).^(1/p) ./ d) ./   (sum(abs.(y) .^ (1/p) ,dims=2) ./d)
avgϵRel = mean(avgϵRel_byRec)
@test meanRelError(ŷ,y,normDim=false,normRec=true,p=p) == avgϵRel
# case 4 - average error relativized
avgϵRel = (sum(abs.(ŷ-y).^p)^(1/p) / (n*d)) / (sum( abs.(y) .^p)^(1/p) / (n*d))
#avgϵRel = (norm((ŷ-y),p)/(n*d)) / (norm(y,p) / (n*d))
@test meanRelError(ŷ,y,normDim=false,normRec=false,p=p) == avgϵRel


# ==================================
# New test
println("** Testing pca()...")

X = [1 10 100; 1.1 15 120; 0.95 23 90; 0.99 17 120; 1.05 8 90; 1.1 12 95]
out = pca(X,error=0.05)
@test out.error ≈ 1.0556269747774571e-5
@test sum(out.X) ≈ 662.3492034128955
#X2 = out.X*out.P'
@test out.explVarByDim ≈ [0.873992272007021,0.9999894437302522,1.0]
X = [1 8; 4.5 5.5; 9.5 0.5]
out = pca(X;K=2)
expectedX = [-4.58465   6.63182;-0.308999  7.09961; 6.75092   6.70262]
expectedP = [0.745691  0.666292;-0.666292  0.745691]
@test isapprox(out.X,expectedX,atol=0.00001) || isapprox(out.X, (.- expectedX),atol=0.00001) 
@test isapprox(out.P,expectedP,atol=0.00001) || isapprox(out.P, (.- expectedP),atol=0.00001)


# ==================================
# New test
println("** Testing accuracy() on probs given in a dictionary...")

ŷ = Dict("Lemon" => 0.33, "Apple" => 0.2, "Grape" => 0.47)
y = "Lemon"
@test accuracy(ŷ,y) == 0
@test accuracy(ŷ,y,tol=2) == 1
y = "Grape"
@test accuracy(ŷ,y) == 1
y = "Something else"
@test accuracy(ŷ,y) == 0

ŷ1 = Dict("Lemon" => 0.6, "Apple" => 0.4)
ŷ2 = Dict("Lemon" => 0.33, "Apple" => 0.2, "Grape" => 0.47)
ŷ3 = Dict("Lemon" => 0.2, "Apple" => 0.5, "Grape" => 0.3)
ŷ4 = Dict("Apple" => 0.2, "Grape" => 0.8)
ŷ = [ŷ1,ŷ2,ŷ3,ŷ4]
y = ["Lemon","Lemon","Apple","Lemon"]
@test accuracy(ŷ,y) == 0.5
@test accuracy(ŷ,y,tol=2) == 0.75

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
cm = ConfusionMatrix(ŷ,["Lemon","Lemon","Apple","Grape","Lemon"],rng=copy(FIXEDRNG))
@test cm.scores == [2 0 1; 0 1 0; 0 0 1]
@test cm.tp == [2,1,1] && cm.tn == [2,4,3] && cm.fp == [0,0,1] && cm.fn == [1, 0, 0]


# Example from https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
y = [0,1,2,2,0]
ŷ = [0,0,2,1,0]
labels = ["Class 0", "Class 1 with an extra long name super long", "Class 2"]
#y = integerDecoder(y .+ 1,labels)
#ŷ = integerDecoder(ŷ .+ 1,labels)
cm = ConfusionMatrix(ŷ,y,labels=labels)

@test cm.precision ≈ [0.6666666666666666, 0.0, 1.0]
@test cm.recall ≈ [1.0, 0.0, 0.5]
@test cm.specificity ≈ [0.6666666666666666, 0.75, 1.0]
@test cm.f1Score ≈ [0.8, 0.0, 0.6666666666666666]
@test cm.meanPrecision == (0.5555555555555555, 0.6666666666666666)
@test cm.meanRecall == (0.5, 0.6)
@test cm.meanSpecificity == (0.8055555555555555, 0.8166666666666667)
@test cm.meanF1Score == (0.48888888888888893, 0.5866666666666667)
@test cm.accuracy == 0.6
@test cm.misclassification == 0.4

@test BetaML.Utils.print(cm,["report"]) == nothing
@test BetaML.Utils.println(cm) == nothing

# ==================================
# New test
println("** Testing classCounts()...")

@test classCountsWithLabels(["a","b","a","c","d"]) == Dict("a"=>2,"b"=>1,"c"=>1,"d"=>1)
@test classCountsWithLabels(['a' 'b'; 'a' 'c';'a' 'b']) == Dict(['a', 'b'] => 2,['a', 'c'] => 1)
@test classCounts(["a","b","a","c","d"],classes=["a","b","c","d","e"]) == [2,1,1,1,0]
@test collect(classCounts(['a' 'b'; 'a' 'c';'a' 'b'])) in [[2,1],[1,2]] # Order doesn't matter


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
println("** Testing meanDicts()...")

a = Dict('a'=> 0.2,'b'=>0.3,'c'=>0.5)
b = Dict('a'=> 0.3,'b'=>0.1,'d'=>0.6)
c = Dict('b'=>0.6,'e'=>0.4)
d = Dict('a'=>1)
dicts = [a,b,c,d]
@test meanDicts(dicts) == Dict('a' => 0.375,'c' => 0.125,'d' => 0.15,'e' => 0.1,'b' => 0.25)
@test meanDicts(dicts,weights=[1,1,97,1]) == Dict('a' => 0.015, 'b' => 0.586, 'c' => 0.005, 'd' =>0.006, 'e' =>0.388)

a = Dict(1=> 0.1,2=>0.4,3=>0.5)
b = Dict(4=> 0.3,1=>0.1,2=>0.6)
c = Dict(5=>0.6,4=>0.4)
d = Dict(2=>1)
dicts = [a,b,c,d]
@test meanDicts(dicts)  == Dict(4 => 0.175, 2 => 0.5, 3 => 0.125, 5 => 0.15, 1 => 0.05)

# ==================================
# New test
println("** Testing partition()...")

m1 = [1:10 11:20 31:40]
m2 = convert(Array{Float64,2},[41:50 51:60])
m3 = makeMatrix(collect(61:70))
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
sampleIterator = SamplerWithData(KFold(nSplits=3,nRepeats=1,shuffle=true,rng=copy(TESTRNG)),data,2)
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
sampleIterator = SamplerWithData(KFold(nSplits=3,nRepeats=2,shuffle=false,rng=copy(TESTRNG)),data,1)
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


println("** Testing crossValidation...")

X = [11:19 21:29 31:39 41:49 51:59 61:69]
Y = [1:9;]
sampler = KFold(nSplits=3,nRepeats=1,shuffle=true,rng=copy(TESTRNG))
(μ,σ) = crossValidation([X,Y],sampler) do trainData,valData,rng
        (xtrain,ytrain) = trainData; (xval,yval) = valData
        trainedModel = buildForest(xtrain,ytrain,30,rng=rng)
        predictions = predict(trainedModel,xval,rng=rng)
        ϵ = meanRelError(predictions,yval,normRec=false)
        return ϵ
    end

@test (μ,σ) == (0.3202242202242202, 0.04307662219315022)


# ==================================
# New test
println("** Testing shuffle()...")

a = [1 2 3; 10 20 30; 100 200 300; 1000 2000 4000; 10000 20000 40000]; b = [4,40,400,4000,40000]
out = shuffle([a,b],rng=copy(FIXEDRNG))
@test out[1] ==  [100 200 300; 1 2 3;10000 20000 40000; 10 20 30; 1000 2000 4000] && out[2] == [400,4,40000,40,4000]
out2 = shuffle(copy(FIXEDRNG),[a,b])
@test out2 == out


a = [1 2 3 4 5; 10 20 30 40 50]; b = [100 200 300 400 500]
out = shuffle([a,b],rng=copy(FIXEDRNG),dims=2)
@test out[1] == [3 1 5 2 4; 30 10 50 20 40] && out[2] == [300 100 500 200 400]


a = [1 2 30; 10 20 30]; b = [100 200 300];
(aShuffled, bShuffled) = shuffle([a,b],dims=2)


a = [1 2 3; 10 20 30]; b = [4,40]
out = shuffle([a,b],rng=copy(FIXEDRNG))


data = [a,b]
dims = 1
rng=Random.GLOBAL_RNG
Ns = [size(m,dims) for m in data]
length(Set(Ns)) == 1 || @error "In `shuffle(arrays)` all individual arrays need to have the same size on the dimension specified"
N    = Ns[1]
ridx = Random.shuffle(rng, 1:N)
out = similar(data)
for (i,a) in enumerate(data)
i = 1
a = data[1]
aidx = convert(Vector{Union{UnitRange{Int64},Vector{Int64}}},[1:i for i in size(a)])
aidx[dims] = ridx
out[i] = a[aidx...]
Vector{Union{UnitRange{Int64},Vector{Int64}}}
end
return out



# ==================================
# New test
println("** Testing generateParallelRngs()...")
x = rand(copy(TESTRNG),100)

function innerFunction(bootstrappedx; rng=Random.GLOBAL_RNG)
     sum(bootstrappedx .* rand(rng) ./ 0.5)
end
function outerFunction(x;rng = Random.GLOBAL_RNG)
    masterSeed = rand(rng,100:9999999999999) # important: with some RNG it is important to do this before the generateParallelRngs to guarantee independance from number of threads
    rngs       = generateParallelRngs(rng,Threads.nthreads()) # make new copy instances
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
poolSize = 3

out = pool1d(x,poolSize)
@test out == [2.0,3.0,4.0,5.0]
out = pool1d(x,poolSize;f=maximum)
@test out == [3,4,5,6]

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
