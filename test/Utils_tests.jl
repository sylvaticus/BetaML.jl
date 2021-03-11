using Test, Statistics, CategoricalArrays
#using StableRNGs
#rng = StableRNG(123)
using BetaML.Utils

println("*** Testing individual utility functions (module `Utils`)...")

# ==================================
# TEST 1: oneHotEncoder
println("Going through Test1 (oneHotEncoder)...")

a = [[1,3,4],[1,4,2,2,3],[2,3]]
b = [2,1,5,2,1]
c = 2
ae = oneHotEncoder(a,5,count=true)
be = oneHotEncoder(b,6,count=true)
ce = oneHotEncoder(c,6)
@test sum(ae*be*ce') == 4

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

encoded  = integerEncoder(a)
uniquea  = unique(a)
decoded  = integerDecoder(encoded,a,unique=false)
decoded2 = integerDecoder(encoded,uniquea)
@test a == decoded == decoded2



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

@test size.(batch(10,3,rng=FIXEDRNG),1) == [3,3,3]
@test size.(batch(10,12,rng=FIXEDRNG),1) == [10]

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
ŷ = [ŷ1,ŷ2,ŷ3,ŷ4]
@test mode(ŷ) == ["Lemon","Grape","Apple","Grape"]

# ==================================
# New test
println("** Testing classCounts()...")

a = ["a","b","a","c","d"]
@test classCounts(["a","b","a","c","d"]) == Dict("a"=>2,"b"=>1,"c"=>1,"d"=>1)
@test classCounts(['a' 'b'; 'a' 'c';'a' 'b']) == Dict(['a', 'b'] => 2,['a', 'c'] => 1)


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
out = partition([m1,m2,m3,m4],parts,shuffle=true,rng=FIXEDRNG)
@test size(out,1) == 4 && size(out[1][3]) == (4,3)
x = [1:10 11:20]
y = collect(31:40)
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3], shuffle=false,rng=FIXEDRNG)
@test xtest[2,2] == 19 && ytest[2] == 39
