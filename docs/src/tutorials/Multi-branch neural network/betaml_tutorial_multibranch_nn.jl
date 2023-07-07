# # [Predictions using a multiple branches neural network](@id multibranch_nn_tutorial)
# Often we can "divide" our feature sets in different groups, where for each group we have many, many variables which we don't know the importance in prediction but for which using a fully dense layer would result in too computational burder.
# For example we want to predict forest trees growth based on soil characteristics, climate characteristics and a bunch of other data (species, age, density..) 
#
# A soil (or climate) database has possibly hundreds of variables, how can we summarise them in a few ones that encode all the "soil" information ?
# Sure we could run PCA or a clustering analysis but a better way is to let our model itself find a way to _encode_ the soil infformation in a vector in a way that it is optimal for our prediction objective, i.e. we target the encoding task to our prediction one.
# Hence we run a multi-branch neural network where one branch is given by the soil variables - it starts from all the hundreds of variables and ends in a few neuron outputs, an other branch in a similar way is for the climate variables, we merge them in a branch to consider the soil-weather interrelation (for example it is well known that water retention capacity of a sandy soil is quite different from that of a clay soil) and finally we merge this branch with the other variables branch to arrive to a single predicted output.
# In this example we concentrate on building, training and predicting a multi-branch neural network. See the other examples for cross-validation, hyperparameters tuning, scaling, overfitting, etc.


#
# Data origin:
# - for now we work on simulated random data


# ## Library and data loading
using Dates                                                              #src
println(now(), " ", "*** Starting bike demand regression tutorial..." )  #src

# Activating the local environment specific to 
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))

# We first load all the packages we are going to use
using  LinearAlgebra, Random, Statistics, StableRNGs, DataFrames, BenchmarkTools, BetaML

using  Test     #src
println(now(), " ", "- Loading, plotting, wrangling data..." )  #src

# Here we are explicit and we use our own fixed RNG:
seed = 123 # The table at the end of this tutorial has been obtained with seeds 123, 1000 and 10000
AFIXEDRNG = StableRNG(seed)

# Generate random data..
N         = 100 # records
soilD     = 20   # dimensions of the soil database
climateD  = 30   # dimensions of the climate database
othervarD = 10   # dimensions of the other variables database

soilX     = rand(StableRNG(seed),N,soilD)
climateX  = rand(StableRNG(seed+10),N,climateD)
othervarX = rand(StableRNG(seed+20),N,othervarD)
X = hcat(soilX,climateX,othervarX)
Y         = rand(StableRNG(seed+30),N)

# layer 1
l1_soil    = DenseLayer(20,30,f=relu,rng=copy(AFIXEDRNG))
l1_climate = ReplicatorLayer(30)
l1_oth     = ReplicatorLayer(10)
l1         = GroupedLayer([l1_soil,l1_climate,l1_oth])
# layer 2
l2_soil    = DenseLayer(30,30,f=relu,rng=copy(AFIXEDRNG))
l2_climate = DenseLayer(30,40,f=relu,rng=copy(AFIXEDRNG))
l2_oth     = ReplicatorLayer(10)
l2         = GroupedLayer([l2_soil,l2_climate,l2_oth])
# layer 3
l3_soil    = DenseLayer(30,4,f=relu,rng=copy(AFIXEDRNG)) # encoding of soil properties
l3_climate = DenseLayer(40,4,f=relu,rng=copy(AFIXEDRNG)) # encoding of climate properties
l3_oth     = DenseLayer(10,15,f=relu,rng=copy(AFIXEDRNG))                         
l3         = GroupedLayer([l3_soil,l3_climate,l3_oth])
# layer 4
l4_soilclim = DenseLayer(8,15,f=relu,rng=copy(AFIXEDRNG))
l4_oth      = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG))                         
l4          = GroupedLayer([l4_soilclim,l4_oth])
# layer 5
l5_soilclim = DenseLayer(15,6,f=relu,rng=copy(AFIXEDRNG))  # encoding of soil and climate properties together
l5_oth      = DenseLayer(15,6,f=relu,rng=copy(AFIXEDRNG))  # encoding of other vars                       
l5          = GroupedLayer([l5_soilclim,l5_oth])
# layer 6
l6          = DenseLayer(12,15,f=relu,rng=copy(AFIXEDRNG)) 
# layer 7
l7          = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG)) 
# layer 8
l8          = DenseLayer(15,1,f=relu,rng=copy(AFIXEDRNG)) 

layers = [l1,l2,l3,l4,l5,l6,l7,l8]
m      = NeuralNetworkEstimator(layers=layers,opt_alg=ADAM(),epochs=100,verbosity=HIGH,rng=copy(AFIXEDRNG))
Ŷ      = fit!(m,X,Y)
rme    = relative_mean_error(Y,Ŷ)

# -------- works
X    = rand(50,30)
Y    = rand(50)
l1_1 = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG))
l1_2 = ReplicatorLayer(10)
l1_3 = ReplicatorLayer(5)
l1   = GroupedLayer([l1_1,l1_2,l1_3])
l2_1 = DenseLayer(15,5,f=relu,rng=copy(AFIXEDRNG))
l2_2 = DenseLayer(10,15,f=relu,rng=copy(AFIXEDRNG))
l2_3 = DenseLayer(5,10,f=relu,rng=copy(AFIXEDRNG))
l2   = GroupedLayer([l2_1,l2_2,l2_3])
l3   = DenseLayer(30,30,f=relu,rng=copy(AFIXEDRNG))
l4   = DenseLayer(30,1,f=relu,rng=copy(AFIXEDRNG))
layers = [l1,l2,l3,l4]
m = buildNetwork(layers,squared_cost)
predict(m,X)
loss(m,X,Y)
get_gradient(m,X[1,:],Y[1,:])
train!(m,X,Y,epochs=400)
Ŷ = predict(m,X)
relative_mean_error(Y,Ŷ)

# -------- works
X    = rand(50,30)
Y    = rand(50)
l1_1 = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG))
l1_2 = ReplicatorLayer(10)
l1_3 = ReplicatorLayer(5)
l1   = GroupedLayer([l1_1,l1_2,l1_3])
l2_1 = DenseLayer(15,5,f=relu,rng=copy(AFIXEDRNG))
l2_2 = DenseLayer(10,15,f=relu,rng=copy(AFIXEDRNG))
l2_3 = DenseLayer(5,10,f=relu,rng=copy(AFIXEDRNG))
l2   = GroupedLayer([l2_1,l2_2,l2_3])
l3_1 = DenseLayer(20,30,f=relu,rng=copy(AFIXEDRNG))
l3_2 = DenseLayer(10,10,f=relu,rng=copy(AFIXEDRNG))
l3   = GroupedLayer([l3_1,l3_2])
l4   = DenseLayer(40,1,f=relu,rng=copy(AFIXEDRNG))
layers = [l1,l2,l3,l4]
m = buildNetwork(layers,squared_cost)
predict(m,X)
loss(m,X,Y)
get_gradient(m,X[1,:],Y[1,:])
train!(m,X,Y,epochs=400)
Ŷ = predict(m,X)
relative_mean_error(Y,Ŷ)

# -------- testing
X    = rand(50,30)
Y    = rand(50)
l1_1 = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG))
l1_2 = ReplicatorLayer(10)
l1_3 = ReplicatorLayer(5)
l1   = GroupedLayer([l1_1,l1_2,l1_3])
l2_1 = DenseLayer(15,5,f=relu,rng=copy(AFIXEDRNG))
l2_2 = DenseLayer(10,15,f=relu,rng=copy(AFIXEDRNG))
l2_3 = DenseLayer(5,10,f=relu,rng=copy(AFIXEDRNG))
l2   = GroupedLayer([l2_1,l2_2,l2_3])
l3_1 = DenseLayer(20,30,f=relu,rng=copy(AFIXEDRNG))
l3_2 = DenseLayer(10,10,f=relu,rng=copy(AFIXEDRNG))
l3   = GroupedLayer([l3_1,l3_2])
l4   = DenseLayer(40,1,f=relu,rng=copy(AFIXEDRNG))
layers = [l1,l2,l3,l4]
m = buildNetwork(layers,squared_cost)
predict(m,X)
loss(m,X,Y)
get_gradient(m,X[1,:],Y[1,:])
train!(m,X,Y,epochs=400)
Ŷ = predict(m,X)
relative_mean_error(Y,Ŷ)