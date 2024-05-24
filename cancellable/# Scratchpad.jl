# Scratchpad

x = [0.12 0.31 0.29 3.21 0.21;
     0.44 1.21 1.18 13.54 0.85
     0.22 0.61 0.58 6.43 0.42;
     0.35 0.93 0.91 10.04 0.71;
     0.51 1.47 1.46 16.12 0.99;
     0.35 0.93 0.91 10.04 0.71;
     0.51 1.47 1.46 16.12 0.99;
     0.22 0.61 0.58 6.43 0.42;
     0.12 0.31 0.29 3.21 0.21;
     0.44 1.21 1.18 13.54 0.85];
m    = AutoEncoder(encoded_size=2,layers_size=15,epochs=400,autotune=false,rng=copy(TESTRNG)) 
x_reduced = fit!(m,x)
x̂ = inverse_predict(m,x_reduced)
x̂sum = sum(x̂)

x = vcat(rand(copy(TESTRNG),0:0.001:0.6,30,5), rand(copy(TESTRNG),0.4:0.001:1,30,5))
m = AutoEncoder(rng=copy(TESTRNG), verbosity=NONE)
x_reduced = fit!(m,x)
x̂ = inverse_predict(m,x_reduced)
x̂sum = sum(x̂)

l2loss_by_cv2(AutoEncoder(rng=copy(TESTRNG), verbosity=NONE),(x,),rng=copy(TESTRNG))

m = AutoEncoder(rng=copy(TESTRNG), verbosity=NONE)
sampler = KFold(nsplits=5,nrepeats=1,rng=copy(TESTRNG))
(μ,σ) = cross_validation([x],sampler) do trainData,valData,rng
    (xtrain,) = trainData; (xval,) = valData
    fit!(m,xtrain)
    x̂val_red = predict(m,xval)
    x̂val     = inverse_predict(m,x̂val_red)
    ϵ        = norm(xval .- x̂val)/size(xval,1) 
    println(ϵ) # different
    reset!(m)
    return ismissing(ϵ) ? Inf : ϵ 
end

function l2loss_by_cv2(m,data;nsplits=5,nrepeats=1,rng=Random.GLOBAL_RNG)
    x= data[1]
    sampler = KFold(nsplits=nsplits,nrepeats=nrepeats,rng=rng)
    (μ,σ) = cross_validation([x],sampler) do trainData,valData,rng
        (xtrain,) = trainData; (xval,) = valData
        fit!(m,xtrain)
        x̂val     = inverse_predict(m,xval)
        ϵ        = norm(xval .- x̂val)/size(xval,1) 
        reset!(m)
        return ismissing(ϵ) ? Inf : ϵ 
    end
    return μ
end



using Random, Pipe, HTTP, CSV, DataFrames, Plots, BetaML
import Distributions: Normal, quantile
Random.seed!(123)

# We download the Boston house prices dataset from interet and split it into x and y
dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_, delim=' ', header=false, ignorerepeated=true) |> DataFrame

data = CSV.File(joinpath("docs","src","tutorials","Feature importance", "data","housing.data"), delim=' ', header=false, ignorerepeated=true) |> DataFrame

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

# We use a Random Forest model as regressor and we compute the variable importance for this model :
fr = FeatureRanker(model=RandomForestEstimator(),nsplits=3,nrepeats=2,recursive=false, ignore_dims_keyword="ignore_dims")
rank = fit!(fr,x,y)

loss_by_col        = info(fr)["loss_by_col"]
sobol_by_col       = info(fr)["sobol_by_col"]
loss_by_col_sd     = info(fr)["loss_by_col_sd"]
sobol_by_col_sd    = info(fr)["sobol_by_col_sd"]
loss_fullmodel     = info(fr)["loss_all_cols"]
loss_fullmodel_sd  = info(fr)["loss_all_cols_sd"]
ntrials_per_metric = info(fr)["ntrials_per_metric"]

# Finally we can plot the variable importance, first using the loss metric ("mda") and then the sobol one:
bar(var_names[sortperm(loss_by_col)], loss_by_col[sortperm(loss_by_col)],label="Loss by var", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (loss_by_col_sd[sortperm(loss_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.5])
vline!([loss_fullmodel], label="Loss with all vars",linewidth=2)
vline!([loss_fullmodel-quantile(Normal(1,0),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
        loss_fullmodel+quantile(Normal(1,0),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
], label=nothing,linecolor=:black,linestyle=:dot,linewidth=1)
savefig("loss_by_var.png")
#-
bar(var_names[sortperm(sobol_by_col)],sobol_by_col[sortperm(sobol_by_col)],label="Sobol index by col", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (sobol_by_col_sd[sortperm(sobol_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.4])
savefig("sobol_ny_var.png")
# As we can see, the two analyses agree on the most important variables, showing that the size of the house (number of rooms), the percentage of low-income population in the neighbourhood and, to a lesser extent, the distance to employment centres are the most important explanatory variables of house price in the Boston area.