# [Understanding variable importance in black box machine learning models](@id variable_importance_tutorial)

# Often you want to understand the contribution of different variables (x columns) to the prediction accuracy of a black-box machine learning model.
# To this end, BetaML 0.12 introduces [`FeatureRanker`](@ref), a flexible variable ranking estimator using multiple variable importance metrics. `FeatureRanker' helps to determine the importance of features in predictions from any black-box machine learning model (not necessarily the BetaML suit), internally using cross-validation to assess the quality of the prediction with or without a given variable.

# By default, it ranks variables (columns) in a single pass without retraining on each one, but it is possible to specify the model to use multiple passes (where on each pass the less important variable is permuted). This helps to assess importance in the presence of highly correlated variables.
# While the default strategy is to simply (temporarily) permute the "test" variable and predict the modified data set, it is possible to refit the model to be evaluated on each variable ("permute and relearn"), of course at a much higher computational cost.
# However, if the ML model to be evaluated supports ignoring variables during prediction (e.g. BetaML tree models), it is possible to specify the keyword argument for such an option in the target model prediction function and avoid refitting.

# In this tutorial we will use `FeatureRanker` first with some synthetic data and then with the Boston dataset to determine the most important variables in determining house prices.
# We will compare the results with Sharpley values using the [`ShapML`] package (https://github.com/nredell/ShapML.jl). 

# Activating the local environment specific to BetaML documentation
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))
using  Statistics, Pipe, StableRNGs, HTTP, CSV, DataFrames, Plots, BetaML
import Distributions: Normal, Uniform, quantile
import ShapML

## Example with synthetic data

# Here we generate a dataset of 5 random variables, where `x1` is the most important in determining `y`, `x2` is somewhat less important, `x3` has interaction effects with `x1`, while `x4` and `x5` do not contribute at all to the calculation of `y`.
# We also add `x6` as a highly correlated variable to `x1`, but note that `x4` also does not contribute to `y`:

TEMPRNG = StableRNG(123)
N     = 2000
xa    = rand(TEMPRNG,Uniform(0.0,10.0),N,5)
xb    = xa[:,1] .* rand.(Normal(1,0.5)) 
x     = hcat(xa,xb)  
y     = [10*r[1]-r[2]+0.1*r[3]*r[1] for r in eachrow(x) ];

# Aside of y, we create also a categorical version to test classification and a further one-hot version to test neural networks models that, for classification tasks, work using one-hot encoded variables:

ysort = sort(y)
ycat  = [(i < ysort[Int(round(N/3))]) ?  "c" :  ( (i < ysort[Int(round(2*N/3))]) ? "a" : "b")  for i in y]
yoh    = fit!(OneHotEncoder(),ycat)

# We first try a Random Forest regressor. The BetaML `RandomForestEstimator` model support a `predict` function with the option to ignore specific dimensions. This allow us to "test" the various variables without retraining the model:

fr = FeatureRanker(model=RandomForestEstimator(rng=TEMPRNG),nsplits=5,nrepeats=1,recursive=false,ranking_metric="mda",ignore_dims_keyword="ignore_dims")
rank = fit!(fr,x,y) # As for the other BetaML models, fit! by default returns the predictions, in this case the ranking, avoiding a `predict` call

# As expected, the ranking shows `x1` as the most important variable. Let's look in detail at the metrics we can get by querying the model with `info(fr)`:

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

# As can be seen from the graphs, the model did a good job of identifying the first variable as the most important one, ignoring the others and even giving a very low importance to the correlated one.

### Comparision with the Shapley values

# For Shapley values we need first to have a trained model
m = RandomForestEstimator(rng=TEMPRNG)
fit!(m,xtrain,ytrain)

function predict_function(model, data)
  data_pred = DataFrame(y_pred = BetaML.predict(model, Matrix(data)))
  return data_pred
end

explain   = DataFrame(x[1:300, :],:auto) 
reference = DataFrame(x,:auto) 

sample_size = 60  # Number of Monte Carlo samples.

# Compute stochastic Shapley values.
data_shap = ShapML.shap(explain = explain,
                        reference = reference,
                        model = m,
                        predict_function = predict_function,
                        sample_size = sample_size,
                        seed = 1
                        )

shap_aggregated =combine(groupby(data_shap,[:feature_name])) do subdf 
            (mean_effect = mean(abs.(subdf.shap_effect)), std = std(abs.(subdf.shap_effect)), n = size(subdf,1)  )
end    
shap_values = shap_aggregated.mean_effect

bar(string.(sortperm(shap_values)),shap_values[sortperm(shap_values)],label="Shapley values by col", yerror=quantile(Normal(1,0),0.975) .* (shap_aggregated.std[sortperm(shap_values)]./ sqrt.(shap_aggregated.n)))

# Note that the value of x1 is similar for the Sobol index and the Shapley values. This shoudn't come as a surprice, as the two metrics are related.

### Classifications

# For classification tasks, the usage of `FeatureRanker` doesn't change:
fr = FeatureRanker(model=RandomForestEstimator(rng=TEMPRNG),nsplits=3,nrepeats=2,recursive=true,ranking_metric="mda",ignore_dims_keyword="ignore_dims")
rank = fit!(fr,x,ycat)

#-

fr = FeatureRanker(model=NeuralNetworkEstimator(verbosity=NONE,rng=TEMPRNG),nsplits=3,nrepeats=1,recursive=false,ranking_metric="sobol",refit=false)
rank = fit!(fr,x,yoh) 

## Determinant of house prices in the Boston alrea

# We first load the data from internet
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_, delim=' ', header=false, ignorerepeated=true) |> DataFrame

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

# Our features are a set of 13 explanatory variables, while the label is the average housing cost:
x = Matrix(data[:,1:13])
y = data[:,14]

# We use a Random Forest model as regressor and we compute the variable importance as we did for the synthetic data:

fr = FeatureRanker(model=RandomForestEstimator(rng=TEMPRNG),nsplits=3,nrepeats=2,recursive=false,ignore_dims_keyword="ignore_dims")
rank = fit!(fr,x,y)

loss_by_col        = info(fr)["loss_by_col"]
sobol_by_col       = info(fr)["sobol_by_col"]
loss_by_col_sd     = info(fr)["loss_by_col_sd"]
sobol_by_col_sd    = info(fr)["sobol_by_col_sd"]
loss_fullmodel     = info(fr)["loss_all_cols"]
loss_fullmodel_sd  = info(fr)["loss_all_cols_sd"]
ntrials_per_metric = info(fr)["ntrials_per_metric"]


bar(var_names[sortperm(loss_by_col)], loss_by_col[sortperm(loss_by_col)],label="Loss by col", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (loss_by_col_sd[sortperm(loss_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.5])

#-
bar(var_names[sortperm(sobol_by_col)],sobol_by_col[sortperm(sobol_by_col)],label="Sobol index by col", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (sobol_by_col_sd[sortperm(sobol_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.4])

# As we can see, the two analyses agree on the most important variables, showing the size of the house, the percentage of the population with a low income and, to a lesser extent, the distance to employment centres as the most important explanatory variables.