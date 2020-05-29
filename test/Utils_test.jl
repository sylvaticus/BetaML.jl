using Test, Statistics

using Bmlt.Utils

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
# TEST 2: softMax
println("** Going through Test2 (softMax)...")
@test isapprox(softMax([2,3,4],β=0.1),[0.3006096053557272,0.3322249935333472,0.36716540111092544])


# ==================================
# TEST 3: autoJacobian
println("** Going through Test3 (softMax, dSoftMax and autoJacobian)...")
@test isapprox(softMax([2,3,4],β=0.1),[0.3006096053557272,0.3322249935333472,0.36716540111092544])

#import Bmlt.Utils: autoJacobian
@test autoJacobian(x -> (x[1]*2,x[2]*x[3]),[1,2,3]) == [2.0 0.0 0.0; 0.0 3.0 2.0]

b = softMax([2,3,4],β=1/2)
c = softMax([2,3.0000001,4],β=1/2)
softMax2(x) = softMax(x,β=1/2)
autoGrad = autoJacobian(softMax2,[2,3,4])
realG2 = [(c[1]-b[1])*10000000,(c[2]-b[2])*10000000,(c[3]-b[3])*10000000]
@test isapprox(autoGrad[:,2],realG2,atol=0.000001)
manualGrad = dSoftMax([2,3,4],β=1/2)
@test isapprox(manualGrad[:,2],realG2,atol=0.000001)

# Manual way is hundred of times faster
#@benchmark autoJacobian(softMax2,[2,3,4])
#@benchmark dSoftMax([2,3,4],β=1/2)

# ==================================
# New test
println("** Going through testing accuracy...")

x = [0.01 0.02 0.1 0.05 0.2 0.1  0.05 0.27  0.2;
     0.05 0.01 0.2 0.02 0.1 0.27 0.1  0.05  0.2]
y = [3,3]
@test [accuracy(x,y,tol=i) for i in 1:10] == [0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# ==================================
# New test
println("** Going through testing scaling...")

skip = [1,4,5]
x = [1.1 4.1 8.1 3 8; 2 4 9 7 2; 7 2 9 3 1]
scaleFactors = getScaleFactors(x,skip=skip)
y  = scale(x,scaleFactors)
y2 = scale(x)
scale!(x)

@test all((sum(mean(y,dims=1)), sum(var(y,corrected=false,dims=1)) ) .≈ (11.366666666666667, 21.846666666666668))


# ==================================
# New test
println("** Testing batch()...")

@test size.(batch(10,3),1) == [3,3,3]
@test size.(batch(10,12),1) == [10]
