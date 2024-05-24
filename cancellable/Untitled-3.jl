
# Syntetic data generation
# x1: high importance, x2: little importance, x3: mixed effects with x1, x4: highly correlated with x1 but no effects on Y, x5 and x6: no effects on Y 
using Random
TESTRNG = Random.GLOBAL_RNG
N     = 2000
D     = 6
nAttempts = 10
xa    = rand(copy(TESTRNG),0:0.0001:10,N,3)
xb    = (xa[:,1] .* 0.5 .* rand(0.8:0.001:1.2)) .+ 10 
xc    = rand(copy(TESTRNG),0:0.0001:10,N,D-4)
x     = hcat(xa,xb,xc)  
y     = [10*r[1]-r[2]+0.05*r[3]*r[1] for r in eachrow(x) ]
((xtrain,xtest),(ytrain,ytest)) = BetaML.partition([x,y],[0.8,0.2],rng=copy(TESTRNG))


# full cols model:
m = RandomForestEstimator(n_trees=50)
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
for a in 1:nAttempts
    println("Running attempt $a...")
    for d in 1:D
        println("- doing modelling without dimension $d ....")
        xd_train = hcat(xtrain[:,1:d-1],shuffle(xtrain[:,d]),xtrain[:,d+1:end])
        xd_test = hcat(xtest[:,1:d-1],shuffle(xtest[:,d]),xtest[:,d+1:end])  
        md = RandomForestEstimator(n_trees=50)
        fit!(md,xd_train,ytrain)
        ŷdtrain            = predict(md,xd_train)
        ŷdtrain2           = predict(m,xtrain,ignore_dims=d)
        ŷdtest            = predict(md,xd_test)
        ŷdtest2           = predict(m,xtest,ignore_dims=d)  
        if a == 1
            #=
            loss_by_cols[d]   = norm(ytest-ŷdtest)/length(ytest)
            sobol_by_cols[d]  = sobol_index(ŷtest,ŷdtest) 
            loss_by_cols2[d]  = norm(ytest-ŷdtest2)/length(ytest)
            sobol_by_cols2[d] = sobol_index(ŷtest,ŷdtest2) 
            diffest_bycols[d] = norm(ŷdtest-ŷdtest2)/length(ytest)
            =#
            loss_by_cols[d]   = norm(ytrain-ŷdtrain)/length(ytrain)
            sobol_by_cols[d]  = sobol_index(ŷtrain,ŷdtrain) 
            loss_by_cols2[d]  = norm(ytrain-ŷdtrain2)/length(ytrain)
            sobol_by_cols2[d] = sobol_index(ŷtrain,ŷdtrain2) 
            diffest_bycols[d] = norm(ŷdtrain-ŷdtrain2)/length(ytrain)
        else
            #=
            loss_by_cols[d]   = online_mean(norm(ytest-ŷdtest)/length(ytest); mean=loss_by_cols[d],n=a-1)
            sobol_by_cols[d]  = online_mean(sobol_index(ŷtest,ŷdtest) ; mean=sobol_by_cols[d],n=a-1)
            loss_by_cols2[d]  = online_mean(norm(ytest-ŷdtest2)/length(ytest); mean=loss_by_cols2[d],n=a-1)
            sobol_by_cols2[d] = online_mean(sobol_index(ŷtest,ŷdtest2) ; mean=sobol_by_cols2[d],n=a-1)
            diffest_bycols[d] = online_mean(norm(ŷdtest-ŷdtest2)/length(ytest); mean=diffest_bycols[d],n=a-1)
            =#
            loss_by_cols[d]   = online_mean(norm(ytrain-ŷdtrain)/length(ytrain); mean=loss_by_cols[d],n=a-1)
            sobol_by_cols[d]  = online_mean(sobol_index(ŷtrain,ŷdtrain) ; mean=sobol_by_cols[d],n=a-1)
            loss_by_cols2[d]  = online_mean(norm(ytrain-ŷdtrain2)/length(ytrain); mean=loss_by_cols2[d],n=a-1)
            sobol_by_cols2[d] = online_mean(sobol_index(ŷtrain,ŷdtrain2) ; mean=sobol_by_cols2[d],n=a-1)
            diffest_bycols[d] = online_mean(norm(ŷdtrain-ŷdtrain2)/length(ytrain); mean=diffest_bycols[d],n=a-1)
        end
    end
end
# Expected order: ~ [5,6,4,3,2,1]

d = 5
xd_train = hcat(xtrain[:,1:d-1],shuffle(xtrain[:,d]),xtrain[:,d+1:end])
md = RandomForestEstimator(n_trees=50)
fit!(md,xd_train,ytrain)
ŷdtrain            = predict(md,xd_train)
loss_d = norm(ytrain-ŷdtrain)/length(ytrain)