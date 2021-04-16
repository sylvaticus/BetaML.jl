# # [Getting started](@id getting_started)

# ## Work in progress

# This document presents some general information concerning BetaML. For detailed information on the algorithms provided by the Toolkit refer to the individual module API or to the tutorial below:
# -  [Regression tutorial](@ref regression_tutorial) - Arguments: _Decision trees, Random forests, neural networks, hyper-parameter tuning_
# -  [Classification tutorial](@ref classification_tutorial) - Arguments: _Decision trees and random forests, neural networks (softmax), pre-processing workflow, confusion matrix_
# -  [Clustering tutorial](@ref classification_clustering) - Arguments: _k-means, kmedoids, generative gaussain models, cross-validation_

# ## [Dealing with stochasticity](@id dealing_with_stochasticity)

# Most models have some stochastic components and support a `rng` parameter. By default, the outputs of these models will hence not be absolutelly equal on each run. If you want to be sure that the output of a model remain constant given the same inputs you can pass a fixed Random Number Generator to the `rng` parameter. Use it with:
#
# - `myAlgorithm(;rng=FIXEDRNG)`               # always produce the same sequence of results on each run of the script ("pulling" from the same rng object on different calls)
# - `myAlgorithm(;rng=StableRNG(SOMEINTEGER))` # always produce the same result (new rng object on each call)
#
# In particular, use `rng=StableRNG(FIXEDSEED)` to retrieve the exacty output as in the documentation or in the unit tests.
#
#
# Most of the stochasticity appears in _training_ a model. However in few cases (e.g. decision trees with missing values) some stocasticity appears also in _predicting_ new data with a trained model. In such cases the model doesn't stire the random seed, so that you can choose at _predict_ time to use a fixed or a variable random seed.



#src # -------------------------------------------------------------------------
#src # ## This is markdown title
#src # This is also markdown

#src ## This is a normal comment

#src a = 1
#src b = a + 1
#src println("*** I am a line of code that is executed, and b is $b")
#src b

#src # A markdown cell

#src using BenchmarkTools
#src c = b + 1
#src c

#src #-
#src k = 100000
#src @btime sum(1:k);


#src #-

#src a = [1,2]
#src using Test  #src
#src @test c == 3; #src

#src #-
#src println("hello")
#src @test c == 3;  #src
