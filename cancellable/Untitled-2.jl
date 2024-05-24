
using Random, LinearAlgebra, Plots
Random.seed!(123)
# Syntetic data generation
# x1: high importance, x2: little importance, x3: mixed effects with x1, x4: highly correlated with x1 but no effects on Y, x5 and x6: no effects on Y 
TEMPRNG = copy(Random.GLOBAL_RNG)
N     = 2000
D     = 6
nAttempts = 30
xa    = rand(TEMPRNG,0:0.0001:10,N,3)
xb    = (xa[:,1] .* 0.5 .* rand(TEMPRNG,0.7:0.001:1.3)) .+ 10 
xc    = rand(TEMPRNG,0:0.0001:10,N,D-4)
x     = hcat(xa,xb,xc)  
y     = [10*r[1]-r[2]+0.1*r[3]*r[1] for r in eachrow(x) ]
((xtrain,xtest),(ytrain,ytest)) = BetaML.partition([x,y],[0.8,0.2],rng=TEMPRNG)


# full cols model:
m = RandomForestEstimator(n_trees=100,rng=TEMPRNG)
m = DecisionTreeEstimator(rng=TEMPRNG)
m = NeuralNetworkEstimator(verbosity=NONE,rng=TEMPRNG)
fit!(m,xtrain,ytrain)
ŷtrain = predict(m,xtrain)
loss = norm(ytrain-ŷtrain)/length(ytrain) # this is good

ŷtest = predict(m,xtest)
loss = norm(ytest-ŷtest)/length(ytest) # this is good

loss_by_cols  = zeros(D)
sobol_by_cols = zeros(D)
loss_by_cols2  = zeros(D)
sobol_by_cols2 = zeros(D)
diffest_bycols = zeros(D)
loss_by_cols_test  = zeros(D)
sobol_by_cols_test = zeros(D)
loss_by_cols2_test  = zeros(D)
sobol_by_cols2_test = zeros(D)
diffest_bycols_test = zeros(D)
for a in 1:nAttempts
    println("Running attempt $a...")
    for d in 1:D
        println("- doing modelling without dimension $d ....")
        xd_train = hcat(xtrain[:,1:d-1],shuffle(TEMPRNG,xtrain[:,d]),xtrain[:,d+1:end])
        xd_test = hcat(xtest[:,1:d-1],shuffle(TEMPRNG,xtest[:,d]),xtest[:,d+1:end])  
        #md = RandomForestEstimator(n_trees=100,rng=TEMPRNG)
        #md = DecisionTreeEstimator(rng=TEMPRNG)
        md = NeuralNetworkEstimator(verbosity=NONE,rng=TEMPRNG)
        fit!(md,xd_train,ytrain)
        ŷdtrain           = predict(md,xd_train)
        #ŷdtrain2          = predict(m,xtrain,ignore_dims=d)
        ŷdtest            = predict(md,xd_test)
        #ŷdtest2           = predict(m,xtest,ignore_dims=d)  
        if a == 1
            loss_by_cols[d]   = norm(ytrain-ŷdtrain)/length(ytrain)
            sobol_by_cols[d]  = sobol_index(ŷtrain,ŷdtrain) 
            #loss_by_cols2[d]  = norm(ytrain-ŷdtrain2)/length(ytrain)
            #sobol_by_cols2[d] = sobol_index(ŷtrain,ŷdtrain2) 
            #diffest_bycols[d] = norm(ŷdtrain-ŷdtrain2)/length(ytrain)
            loss_by_cols_test[d]   = norm(ytest-ŷdtest)/length(ytest)
            sobol_by_cols_test[d]  = sobol_index(ŷtest,ŷdtest) 
            #loss_by_cols2_test[d]  = norm(ytest-ŷdtest2)/length(ytest)
            #sobol_by_cols2_test[d] = sobol_index(ŷtest,ŷdtest2) 
            #diffest_bycols_test[d] = norm(ŷdtest-ŷdtest2)/length(ytest)
        else
            loss_by_cols[d]   = online_mean(norm(ytrain-ŷdtrain)/length(ytrain); mean=loss_by_cols[d],n=a-1)
            sobol_by_cols[d]  = online_mean(sobol_index(ŷtrain,ŷdtrain) ; mean=sobol_by_cols[d],n=a-1)
            #loss_by_cols2[d]  = online_mean(norm(ytrain-ŷdtrain2)/length(ytrain); mean=loss_by_cols2[d],n=a-1)
            #sobol_by_cols2[d] = online_mean(sobol_index(ŷtrain,ŷdtrain2) ; mean=sobol_by_cols2[d],n=a-1)
            #diffest_bycols[d] = online_mean(norm(ŷdtrain-ŷdtrain2)/length(ytrain); mean=diffest_bycols[d],n=a-1)
            loss_by_cols_test[d]   = online_mean(norm(ytest-ŷdtest)/length(ytest); mean=loss_by_cols_test[d],n=a-1)
            sobol_by_cols_test[d]  = online_mean(sobol_index(ŷtest,ŷdtest) ; mean=sobol_by_cols_test[d],n=a-1)
            #loss_by_cols2_test[d]  = online_mean(norm(ytest-ŷdtest2)/length(ytest); mean=loss_by_cols2_test[d],n=a-1)
            #sobol_by_cols2_test[d] = online_mean(sobol_index(ŷtest,ŷdtest2) ; mean=sobol_by_cols2_test[d],n=a-1)
            #diffest_bycols_test[d] = online_mean(norm(ŷdtest-ŷdtest2)/length(ytest); mean=diffest_bycols_test[d],n=a-1)
        end
    end
end
# Expected order: ~ [{5,6,4},{3,2},1] good
#                 ~ [{5,6},{4,3,2,1}] still good but don't see corelation
bar(string.(sortperm(loss_by_cols)),loss_by_cols[sortperm(loss_by_cols)],label="loss_by_cols train")
bar(string.(sortperm(sobol_by_cols)),sobol_by_cols[sortperm(sobol_by_cols)],label="sobol_by_cols train")
bar(string.(sortperm(loss_by_cols2)),loss_by_cols2[sortperm(loss_by_cols2)],label="loss_by_cols2 train")
bar(string.(sortperm(sobol_by_cols2)),sobol_by_cols2[sortperm(sobol_by_cols2)],label="sobol_by_cols2 train")
bar(string.(sortperm(loss_by_cols_test)),loss_by_cols_test[sortperm(loss_by_cols_test)],label="loss_by_cols test")
bar(string.(sortperm(sobol_by_cols_test)),sobol_by_cols_test[sortperm(sobol_by_cols_test)],label="sobol_by_cols test")
bar(string.(sortperm(loss_by_cols2_test)),loss_by_cols2_test[sortperm(loss_by_cols2_test)],label="loss_by_cols2 test")
bar(string.(sortperm(sobol_by_cols2_test)),sobol_by_cols2_test[sortperm(sobol_by_cols2_test)],label="sobol_by_cols2 test")




d = 5
xd_train = hcat(xtrain[:,1:d-1],shuffle(xtrain[:,d]),xtrain[:,d+1:end])
md = RandomForestEstimator(n_trees=50)
fit!(md,xd_train,ytrain)
ŷdtrain            = predict(md,xd_train)
loss_d = norm(ytrain-ŷdtrain)/length(ytrain)