
module Api

# Shared api trough the modules, i.e. names used by more than one module
# Modules are free to use other functions but these are defined here to avoid name conflicts
# and allows instead Multiple Dispatch to handle them

export predict, fit, fit!, partition

predict()   = nothing
fit()       = nothing
fit!()      = nothing
partition() = nothing

end
