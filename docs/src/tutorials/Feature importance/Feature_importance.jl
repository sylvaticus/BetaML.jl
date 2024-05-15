# # [Understanding variable importance in black-box machine learning models](@id variable_importance_tutorial)

# Often we want to understand the contribution of different variables (x columns) to the prediction accuracy of a black-box machine learning model.
# To this end, BetaML 0.12 introduces [`FeatureRanker`](@ref), a flexible variable ranking estimator that employs multiple variable importance metrics. 
# `FeatureRanker` helps to determine the importance of features in predictions from any black-box machine learning model (not necessarily the BetaML suit), internally using cross-validation to assess the quality of the predictions (`metric="mda"`), or the contribution of the variable to the variance of the predictions (`metric="sobol"`), with or without a given variable.

# By default, it ranks variables (columns) in a single pass without retraining on each one. However, it is possible to specify the model to use multiple passes (where on each pass the less important variable is permuted). This helps to assess importance in the presence of highly correlated variables.
# While the default strategy is to simply (temporarily) permute the "test" variable and predict the modified data set, it is possible to refit the model to be evaluated on each variable ("permute and relearn"), of course at a much higher computational cost.
# However, if the ML model to be evaluated supports ignoring variables during prediction (as BetaML tree models do), it is possible to specify the keyword argument for such an option in the target model prediction function and avoid refitting.

# In this tutorial we will use `FeatureRanker` first with some synthetic data, and then with the Boston dataset to determine the most important variables in determining house prices.
# We will compare the results with Shapley values using the [`ShapML`](https://github.com/nredell/ShapML.jl) package. 

# Let's start by activating the local environment specific to the BetaML documentation and loading the necessary packages:
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))
using  Statistics, Random, Pipe, StableRNGs, HTTP, CSV, DataFrames, Plots, BetaML
import Distributions: Normal, Uniform, quantile
import ShapML
Random.seed!(123)

# ## Example with synthetic data

# In this example, we generate a dataset of 5 random variables, where `x1` is the most important in determining `y`, `x2` is somewhat less important, `x3` has interaction effects with `x1`, while `x4` and `x5` do not contribute at all to the calculation of `y`.
# We also add `x6` as a highly correlated variable to `x1`, but note that `x4` also does not contribute to `y`:

N     = 2000
xa    = rand(Uniform(0.0,10.0),N,5)
xb    = xa[:,1] .* rand.(Normal(1,0.5)) 
x     = hcat(xa,xb)  
y     = [10*r[1]-r[2]+0.1*r[3]*r[1] for r in eachrow(x) ];

# Aside of `y`, that is numerical, we create also a categorical version to test classification and a further one-hot version to test neural networks models that, for classification tasks, work using one-hot encoded variables:

ysort = sort(y)
ycat  = [(i < ysort[Int(round(N/3))]) ?  "c" :  ( (i < ysort[Int(round(2*N/3))]) ? "a" : "b")  for i in y]
yoh    = fit!(OneHotEncoder(),ycat)

# We first try a Random Forest regressor. The BetaML `RandomForestEstimator` model supports a `predict` function with the option to ignore specific dimensions. This allow us to "test" the various variables without retraining the model:

fr = FeatureRanker(model=RandomForestEstimator(),nsplits=5,nrepeats=1,recursive=false,metric="mda",ignore_dims_keyword="ignore_dims")
rank = fit!(fr,x,y) # As for the other BetaML models, `fit!` by default returns the predictions, in this case the ranking, avoiding a `predict` call

# As expected, the ranking shows `x1` as the most important variable. Let's look in detail at the metrics that we can obtain by querying the model with `info(fr)`:

loss_by_col        = info(fr)["loss_by_col"]
sobol_by_col       = info(fr)["sobol_by_col"]
loss_by_col_sd     = info(fr)["loss_by_col_sd"]
sobol_by_col_sd    = info(fr)["sobol_by_col_sd"]
loss_fullmodel     = info(fr)["loss_all_cols"]
loss_fullmodel_sd  = info(fr)["loss_all_cols_sd"]
ntrials_per_metric = info(fr)["ntrials_per_metric"]

# Since we choosed `mda` as the reported metric, we must have that the reported rank is equal to the sortperm of `loss_by_col`:

sortperm(loss_by_col) == rank

# We can plot the loss per (omitted) column... 
bar(string.(rank),loss_by_col[rank],label="Loss by col", yerror=quantile(Normal(1,0),0.975) .* (loss_by_col_sd[rank]./sqrt(ntrials_per_metric)))

# ..and the sobol values:
bar(string.(sortperm(sobol_by_col)),sobol_by_col[sortperm(sobol_by_col)],label="Sobol index by col", yerror=quantile(Normal(1,0),0.975) .* (sobol_by_col_sd[sortperm(sobol_by_col)]./sqrt(ntrials_per_metric)))

# As we can see from the graphs, the model did a good job of identifying the first variable as the most important one, ignoring the others and even giving a very low importance to the correlated one.

# ### Comparision with the Shapley values

# For Shapley values we need first to have a trained model
m = RandomForestEstimator()
fit!(m,x,y)

# We need then to wrap the predict function, accounting with the fact that BetaML models works with standard arrays, while `ShapML` assume data in DataFrame format:
function predict_function(model, data)
  data_pred = DataFrame(y_pred = BetaML.predict(model, Matrix(data)))
  return data_pred
end

# We set up other data related to the simulation..
explain   = DataFrame(x[1:300, :],:auto) 
reference = DataFrame(x,:auto) 

sample_size = 60  # Number of Monte Carlo samples.

# and finally compute the stochastic Shapley values per individual record:
data_shap = ShapML.shap(explain = explain,
                        reference = reference,
                        model = m,
                        predict_function = predict_function,
                        sample_size = sample_size,
                        seed = 1
                        )
# We aggregate the Shape values by feature
shap_aggregated =combine(groupby(data_shap,[:feature_name])) do subdf 
            (mean_effect = mean(abs.(subdf.shap_effect)), std = std(abs.(subdf.shap_effect)), n = size(subdf,1)  )
end    
shap_values = shap_aggregated.mean_effect

bar(string.(sortperm(shap_values)),shap_values[sortperm(shap_values)],label="Shapley values by col", yerror=quantile(Normal(1,0),0.975) .* (shap_aggregated.std[sortperm(shap_values)]./ sqrt.(shap_aggregated.n)))

# Note that the output using the Sobol index and the Shapley values are very similar. This shoudn't come as a surprice, as the two metrics are related.

# ### Classifications

# For classification tasks, the usage of `FeatureRanker` doesn't change:
fr = FeatureRanker(model=RandomForestEstimator(),nsplits=3,nrepeats=2,recursive=true,metric="mda",ignore_dims_keyword="ignore_dims")
rank = fit!(fr,x,ycat)

#-

fr = FeatureRanker(model=NeuralNetworkEstimator(verbosity=NONE),nsplits=3,nrepeats=1,recursive=false,metric="sobol",refit=false)
rank = fit!(fr,x,yoh) 

# ## Determinant of house prices in the Boston alrea

# We start this example by first loading the data from a CSV file and splitting the data in features and labels:
#src dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
#src data    = @pipe HTTP.get(dataURL).body |> CSV.File(_, delim=' ', header=false, ignorerepeated=true) |> DataFrame
data = CSV.File(joinpath(@__DIR__,"data","housing.data"), delim=' ', header=false, ignorerepeated=true) |> DataFrame

var_names = [
  "CRIM",    # per capita crime rate by town
  "ZN",      # proportion of residential land zoned for lots over 25,000 sq.ft.
  "INDUS",   # proportion of non-retail business acres per town
  "CHAS",    # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
  "NOX",     # nitric oxides concentration (parts per 10 million)
  "RM",      # average number of rooms per dwelling
  "AGE",     # proportion of owner-occupied units built prior to 1940
  "DIS",     # weighted distances to five Boston employment centres
  "RAD",     # index of accessibility to radial highways
  "TAX",     # full-value property-tax rate per $10,000
  "PTRATIO", # pupil-teacher ratio by town
  "B",       # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  "LSTAT",   # % lower status of the population
]
y_name = "MEDV" # Median value of owner-occupied homes in $1000's

# Our features are a set of 13 explanatory variables, while the label that we want to estimate is the average housing prices:
x = Matrix(data[:,1:13])
y = data[:,14]

# We use a Random Forest model as regressor and we compute the variable importance for this model as we did for the synthetic data:
fr = FeatureRanker(model=RandomForestEstimator(),nsplits=3,nrepeats=2,recursive=false)
rank = fit!(fr,x,y)

loss_by_col        = info(fr)["loss_by_col"]
sobol_by_col       = info(fr)["sobol_by_col"]
loss_by_col_sd     = info(fr)["loss_by_col_sd"]
sobol_by_col_sd    = info(fr)["sobol_by_col_sd"]
loss_fullmodel     = info(fr)["loss_all_cols"]
loss_fullmodel_sd  = info(fr)["loss_all_cols_sd"]
ntrials_per_metric = info(fr)["ntrials_per_metric"]

# Finally we can plot the variable importance:
bar(var_names[sortperm(loss_by_col)], loss_by_col[sortperm(loss_by_col)],label="Loss by var", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (loss_by_col_sd[sortperm(loss_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.5])
vline!([loss_fullmodel], label="Loss with all vars",linewidth=2)
vline!([loss_fullmodel-quantile(Normal(1,0),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
        loss_fullmodel+quantile(Normal(1,0),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
], label=nothing,linecolor=:black,linestyle=:dot,linewidth=1)

#-
bar(var_names[sortperm(sobol_by_col)],sobol_by_col[sortperm(sobol_by_col)],label="Sobol index by col", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (sobol_by_col_sd[sortperm(sobol_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.4])

# As we can see, the two analyses agree on the most important variables, showing that the size of the house (number of rooms), the percentage of low-income population in the neighbourhood and, to a lesser extent, the distance to employment centres are the most important explanatory variables of house price in the Boston area.