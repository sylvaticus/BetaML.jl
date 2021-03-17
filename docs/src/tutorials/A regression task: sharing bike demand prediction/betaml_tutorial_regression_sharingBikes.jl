
# # A regression task: the prediction of  bike  sharing demand
# The task is to estimate the influence of several variables (like the weather, the season, the day of the week..) on the demand of shared bicycles, so that the authority in charge of the service can organise the service in the best way.
#
# Data origin:
# - original full dataset (by hour, not used here): https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
# - simplified dataset (by day, with some simple scaling): https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec
# - description: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf
# - data: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip

# Note that even if we are estimating a time serie, we are not using here a recurrent neural network as we assume the temporal dependence to be negligible (i.e. $Y_t = f(X_t)$ alone).

# ## Library and data loading

using LinearAlgebra, Random, DataFrames, CSV, StatsPlots,  BetaML
import Distributions: Uniform
import DecisionTree # for comparision

using BetaML  # only needed to get the relative path in importing data
baseDir = joinpath(dirname(pathof(BetaML)),"..","docs","src","tutorials","A regression task: sharing bike demand prediction")

# Data loading and cleaning..
data = CSV.File(joinpath(baseDir,"data","bike_sharing_day.csv"),delim=',') |> DataFrame
describe(data)

# The variable we want to learn to predict is `cnt`, the total demand of bikes for a given day. Even if it is indeed an integer, we treat it as a continuous variable, so each single prediction will be a scalar $$Y \in R$$.


# ## Data preparation
# The first step is to prepare the data for the analysis. This indeed depends already on the model we want to employ, as some models "accept" everything as input, no matter if the data is numerical or categorical, if it has missing values or not... Other models are instead more exigents, and require more or less work.
# Here we use Decision Tree and Random Forest models that belong to the first category, those of the models for which any kind of input is fine.
# Hence, the only things we have to do is to select the variables in input and those representing our output:

x    = convert(Matrix,hcat(data[:,[:instant,:season,:yr,:mnth,:holiday,:weekday,:workingday,:weathersit,:temp,:atemp,:hum,:windspeed]]))
y    = data[:,16]

# We can now split the dataset between the data we will use for training the algorithm (xtrain/ytrain), those for selecting the hyperparameters (xval/yval) and finally those for testing the quality of the algoritm with the optimal hyperparameters (xtest/ytest). For doing that we use the function `Utils.partition` specifying the share we want to use for these three different jobs. As our data represent indeed a time serie, we want our model to be able to predict _future_ demand of bike sharing from _past_, observed rented bikes, so we do not shuffle the datasets as it would be the default.

((xtrain,xval,xtest),(ytrain,yval,ytest)) = Utils.partition([x,y],[0.75,0.125,1-0.75-0.125],shuffle=false)
(ntrain, nval, ntest) = size.([ytrain,yval,ytest],1)

# We can now "tune" our model so-called hyperparameters, i.e. choose the best exogenous parameters of our algorithm, where "best" refer to some minimisation of a "loss" function between the true and the predicted value, where a dedicated "validation" subset of the input is used (xval and yval) in our case.
# BetaML doesn't have a diedicated function for hyperparameters optimisation, but it is easy to write some custom julia code, at least for a simple grid-based "search". Indeed the reason that a dedicated function exists in other Machine Learning libraries is that loops in other languages are slow, but this is not a problem in Julia, so you can retain the flexibility to write the kind of hyperparameter tuning that best fits your needs.
# Below if an example of a possible such function. Note there are more "elegant" ways to code it, but this one does the job. We will see the various functions inside `tuneHyperParameters()` in a moment. For now let's going just to observe that `tuneHyperParameters` just loops over all the possible hyperparameter and select the one where the error between xval and yval is minimised.

function tuneHyperParameters(model,xtrain,ytrain,xval,yval;maxDepthsRange=15:15,maxFeaturesRange=size(xtrain,2):size(xtrain,2),nTreesRange=20:20,βRange=0:0,minRecordsRange=2:2,repetitions=5,rng=Random.GLOBAL_RNG)
    # We start with an infinititly high error
    bestMre         = +Inf
    bestMaxDepth    = 1
    bestMaxFeatures = 1
    bestMinRecords  = 2
    bestNTrees      = 1
    bestβ           = 0
    # We loop over all possible hyperparameter combinations...
    for maxDepth in maxDepthsRange
        for maxFeatures in maxFeaturesRange
            for minRecords in minRecordsRange
                for nTrees in nTreesRange
                   for β in βRange
                       totAttemptError = 0.0
                       # We run several repetitions with the same hyperparameter combination to account for stochasticity...
                       for r in 1:repetitions
                          if model == "DecisionTree"
                             # here we train the Decition Tree model
                             myTrainedModel = buildTree(xtrain,ytrain, maxDepth=maxDepth,maxFeatures=maxFeatures,minRecords=minRecords,rng=rng)
                          else
                             # Here we train the Random Forest model
                             myTrainedModel = buildForest(xtrain,ytrain,nTrees,maxDepth=maxDepth,maxFeatures=maxFeatures,minRecords=minRecords,β=β,rng=rng)
                          end
                          # Here we make prediciton with this trained model and we compute its error
                          ŷval   = predict(myTrainedModel, xval,rng=rng)
                          mreVal = meanRelError(ŷval,yval)
                          totAttemptError += mreVal
                       end
                       avgAttemptedDepthError = totAttemptError / repetitions
                       # Select this specific combination of hyperparameters if its error is the lowest
                       if avgAttemptedDepthError < bestMre
                         bestMre         = avgAttemptedDepthError
                         bestMaxDepth    = maxDepth
                         bestMaxFeatures = maxFeatures
                         bestNTrees      = nTrees
                         bestβ           = β
                         bestMinRecords  = minRecords
                       end
                   end
               end
           end
      end
    end

    return (bestMre,bestMaxDepth,bestMaxFeatures,bestMinRecords,bestNTrees,bestβ)
end

# We can now run the hyperparameter optimisation function with some "reasonable" range:
(bestMre,bestMaxDepth,bestMaxFeatures,bestMinRecords) = tuneHyperParameters("DecisionTree",xtrain,ytrain,xval,yval,
           maxDepthsRange=3:7,maxFeaturesRange=10:12,minRecordsRange=2:6,repetitions=5,rng=copy(FIXEDRNG))

# Now that we have the "optimal" hyperparameters we can build ("train") our model using them:
myTree = buildTree(xtrain,ytrain, maxDepth=bestMaxDepth, maxFeatures=bestMaxFeatures,minRecords=bestMinRecords,rng=copy(FIXEDRNG))

# The above function produce a DecisionTree object that can now be used to make predictions given some features, i.e. given some X matrix of (number of observations x dimensions) predict the relative Y vector of scalers in R.
(ŷtrain,ŷval,ŷtest) = predict.([myTree], [xtrain,xval,xtest])

# Note that the above code uses the "dot syntax" to "broadcast" `predict()` over an array of label matrices. It is exactly equivalent to:

ŷtrain = predict(myTree, xtrain)
ŷval   = predict(myTree, xval)
ŷtest  = predict(myTree, xtest)

# Also note that we use the fully qualified name `predict` as we loaded other


mreTrain = meanRelError(ŷtrain,ytrain)



mreVal = meanRelError(ŷval,yval)


mreTest  = meanRelError(ŷtest,ytest)



scatter(ytrain,ŷtrain,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in training period")

scatter(yval,ŷval,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in validation period")



scatter(ytest,ŷtest,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in testing period")

# Full period plot (2 years)
ŷtrainfull = vcat(ŷtrain,fill(missing,nval+ntest))
ŷvalfull   = vcat(fill(missing,ntrain), ŷval, fill(missing,ntest))
ŷtestfull  = vcat(fill(missing,ntrain+nval), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷvalfull ŷtestfull], label=["obs" "train" "val" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period")


# Focus on testing data
stc = 620
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷvalfull[stc:endc] ŷtestfull[stc:endc]], label=["obs" "val" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period")


(bestMre,bestMaxDepth,bestMaxFeatures,bestMinRecords,bestNTrees,bestβ) = tuneHyperParameters("RandomForest",xtrain,ytrain,xval,yval,
          maxDepthsRange=size(x,1):size(x,1),maxFeaturesRange=Int(round(sqrt(size(x,2)))):Int(round(sqrt(size(x,2)))),
          minRecordsRange=4:6,nTreesRange=15:5:25,βRange=[0;10;50;100:100:500],repetitions=5,rng=copy(FIXEDRNG))

myForest = buildForest(xtrain,ytrain, bestNTrees, maxDepth=bestMaxDepth,maxFeatures=bestMaxFeatures,minRecords=bestMinRecords,β=bestβ,oob=true,rng=copy(FIXEDRNG))
oobError = myForest.oobError # note: Here the oob reported is different than the testing error. Why? Because the division between training/validation a,d testing is not random, but has a temporal basis. It seems in this example that data in validation/testing feel a different pattern/variance than those in training

(ŷtrain,ŷval,ŷtest)         = predict.([myForest], [xtrain,xval,xtest])
(mreTrain, mreVal, mreTest) = meanRelError.([ŷtrain,ŷval,ŷtest],[ytrain,yval,ytest])


scatter(ytrain,ŷtrain,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in training period")

scatter(yval,ŷval,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in validation period")

scatter(ytest,ŷtest,xlabel="daily rides",ylabel="est. daily rides",label=nothing,title="Est vs. obs in testing period")

# Full period plot (2 years)
ŷtrainfull = vcat(ŷtrain,fill(missing,nval+ntest))
ŷvalfull = vcat(fill(missing,ntrain), ŷval, fill(missing,ntest))
ŷtestfull = vcat(fill(missing,ntrain+nval), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷvalfull ŷtestfull], label=["obs" "train" "val" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period")

# Focus on testing data
stc = 620
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷvalfull[stc:endc] ŷtestfull[stc:endc]], label=["obs" "val" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period")

# ##Comparision with DecisionTree

n_subfeatures=-1; n_trees=30; partial_sampling=1; max_depth=26
min_samples_leaf=2; min_samples_split=2; min_purity_increase=0.0; seed=3

model = DecisionTree.build_forest(ytrain, convert(Matrix,xtrain),
                     n_subfeatures,
                     n_trees,
                     partial_sampling,
                     max_depth,
                     min_samples_leaf,
                     min_samples_split,
                     min_purity_increase;
                     rng = seed)

(ŷtrain,ŷval,ŷtest) = DecisionTree.apply_forest.([model],[xtrain,xval,xtest])
(mreTrain, mreVal, mreTest) = meanRelError.([ŷtrain,ŷval,ŷtest],[ytrain,yval,ytest])

ŷtrainfull = vcat(ŷtrain,fill(missing,nval+ntest))
ŷvalfull = vcat(fill(missing,ntrain), ŷval, fill(missing,ntest))
ŷtestfull = vcat(fill(missing,ntrain+nval), ŷtest)
plot(data[:,:dteday],[data[:,:cnt] ŷtrainfull ŷvalfull ŷtestfull], label=["obs" "train" "val" "test"], legend=:topleft, ylabel="daily rides", title="Daily bike sharing demand observed/estimated across the\n whole 2-years period")

# Focus on testing data
stc = 620
endc = size(x,1)
plot(data[stc:endc,:dteday],[data[stc:endc,:cnt] ŷvalfull[stc:endc] ŷtestfull[stc:endc]], label=["obs" "val" "test"], legend=:bottomleft, ylabel="Daily rides", title="Focus on the testing period")
