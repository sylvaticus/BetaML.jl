
module Api




# Shared api trough the modules, i.e. names used by more than one module
# Modules are free to use other functions but these are defined here to avoid name conflicts
# and allows instead Multiple Dispatch to handle them

export BetaMLModel, predict, fit, fit!, partition, info

abstract type BetaMLModel end

predict()   = nothing
fit()       = nothing
fit!()      = nothing
partition() = nothing
info()      = nothing
end
