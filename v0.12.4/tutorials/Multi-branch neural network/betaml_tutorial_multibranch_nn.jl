# # [A deep neural network with multi-branch architecture](@id multibranch_nn_tutorial)

# Often we can "divide" our feature sets into different groups, where for each group we have many, many variables whose importance in prediction we don't know, but for which using a fully dense layer would be too computationally expensive.
# For example, we want to predict the growth of forest trees based on soil characteristics, climate characteristics and a bunch of other data (species, age, density...). 
# 
# A soil (or climate) database may have hundreds of variables, how can we reduce them to a few that encode all the "soil" information?
# Sure, we could do a PCA or a clustering analysis, but a better way is to let our model itself find a way to _encode_ the soil information into a vector in a way that is optimal for our prediction goal, i.e. we target the encoding task at our prediction goal.
# 
# So we run a multi-branch neural network where one branch is given by the soil variables - it starts from all the hundreds of variables and ends in a few neuron outputs, another branch in a similar way is for the climate variables, we merge them in a branch to take into account the soil-weather interrelation (for example, it is well known that the water retention capacity of a sandy soil is quite different from that of a clay soil) and finally we merge this branch with the other variable branch to arrive at a single predicted output.
# In this example we focus on building, training and predicting a multi-branch neural network. See the other examples for cross-validation, hyperparameter tuning, scaling, overfitting, encoding, etc.

#
# Data origin:
# - while we hope to apply this example soon on actual real world data, for now we work on synthetic random data just to assess the validity of the network configuration.


# ## Library and data generation
using Dates                                                       #src
println(now(), " ", "*** Starting multi-branch nn tutorial..." )  #src

# Activating the local environment specific to the tutorials
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))

# We first load all the packages we are going to use
using  StableRNGs, BetaML, Plots

using  Test     #src
println(now(), " ", "- Generating data and implementing the model..." )  #src

# Here we are explicit and we use our own fixed RNG:
seed      = 123 
AFIXEDRNG = StableRNG(seed)

# Here we generate the random data..
N         = 100 # records
soilD     = 20   # dimensions of the soil database
climateD  = 30   # dimensions of the climate database
othervarD = 10   # dimensions of the other variables database

soilX     = rand(StableRNG(seed),N,soilD)
climateX  = rand(StableRNG(seed+10),N,climateD)
othervarX = rand(StableRNG(seed+20),N,othervarD)
X         = hcat(soilX,climateX,othervarX)
Y         = rand(StableRNG(seed+30),N)

# ## Model definition

# ![Neural Network model](imgs/multibranch_nn.png)
# 
# In the figure above, each circle represents a multi-neuron layer, with the number of neurons (output dimensions) written inside. Dotted circles are `RreplicatorLayer`s, which simply "pass through" the information to the next layer.
# Red layers represent the layers responsible for the final step in encoding the information for a given branch. Subsequent layers will use this encoded information (i.e. decode it) to finally provide the prediction for the branch.  
# We create a first branch for the soil variables, a second for the climate variables and finally a third for the other variables. We merge the soil and climate branches in layer 4 and the resulting branch and the other variables branch in layer 6. Finally, the single neuron layer 8 provides the prediction.
#         
# The weights along the whole chain can be learned using the traditional backpropagation algorithm.

# The whole model can be implemented with the following code:

# - layer 1:
l1_soil    = DenseLayer(20,30,f=relu,rng=copy(AFIXEDRNG))
l1_climate = ReplicatorLayer(30)
l1_oth     = ReplicatorLayer(10)
l1         = GroupedLayer([l1_soil,l1_climate,l1_oth])
# - layer 2:
l2_soil    = DenseLayer(30,30,f=relu,rng=copy(AFIXEDRNG))
l2_climate = DenseLayer(30,40,f=relu,rng=copy(AFIXEDRNG))
l2_oth     = ReplicatorLayer(10)
l2         = GroupedLayer([l2_soil,l2_climate,l2_oth])
# - layer 3:
l3_soil    = DenseLayer(30,4,f=relu,rng=copy(AFIXEDRNG)) # encoding of soil properties
l3_climate = DenseLayer(40,4,f=relu,rng=copy(AFIXEDRNG)) # encoding of climate properties
l3_oth     = DenseLayer(10,15,f=relu,rng=copy(AFIXEDRNG))                         
l3         = GroupedLayer([l3_soil,l3_climate,l3_oth])
# - layer 4:
l4_soilclim = DenseLayer(8,15,f=relu,rng=copy(AFIXEDRNG))
l4_oth      = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG))                         
l4          = GroupedLayer([l4_soilclim,l4_oth])
# - layer 5:
l5_soilclim = DenseLayer(15,6,f=relu,rng=copy(AFIXEDRNG))  # encoding of soil and climate properties together
l5_oth      = DenseLayer(15,6,f=relu,rng=copy(AFIXEDRNG))  # encoding of other vars                       
l5          = GroupedLayer([l5_soilclim,l5_oth])
# - layer 6:
l6          = DenseLayer(12,15,f=relu,rng=copy(AFIXEDRNG)) 
# - layer 7:
l7          = DenseLayer(15,15,f=relu,rng=copy(AFIXEDRNG)) 
# - layer 8:
l8          = DenseLayer(15,1,f=relu,rng=copy(AFIXEDRNG)) 

# Finally we put the layers together and we create our `NeuralNetworkEstimator` model:
layers = [l1,l2,l3,l4,l5,l6,l7,l8]
m      = NeuralNetworkEstimator(layers=layers,opt_alg=ADAM(),epochs=100,rng=copy(AFIXEDRNG))

# ## Fitting the model 
println(now(), " ", "- model fitting..." )  #src
# We are now ready to fit the model to the data. By default BetaML models return directly the predictions of the trained data as the output of the fitting call, so there is no need to separate call `predict(m,X)`. 
Ŷ      = fit!(m,X,Y)

# ## Model quality assessment
println(now(), " ", "- assessing the model quality..." )  #src
# We can compute the relative mean error between the "true" Y and the Y estimated by the model.
rme    = relative_mean_error(Y,Ŷ)
@test rme <0.1 #src

# Of course we know there is no actual relation here between the X and The Y, as both are randomly generated, the result above just tell us that the network has been able to find a path between the X and Y that has been used for training, but we hope that in the real application this learned path represent a true, general relation beteen the inputs and the outputs.

# Finally we can also plot Y again Ŷ and visualize how the average loss reduced along the training:
scatter(Y,Ŷ,xlabel="vol observed",ylabel="vol estimated",label=nothing,title="Est vs. obs volumes")

#-
loss_per_epoch = info(m)["loss_per_epoch"]

plot(loss_per_epoch, xlabel="epoch", ylabel="loss per epoch", label=nothing, title="Loss per epoch")

println(now(), " ", "- Ended multi-branch nn example." )  #src