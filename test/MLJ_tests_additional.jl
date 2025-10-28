import MLJTestIntegration
import Dates
# Complete MLJ test integration suit for BetaML models
# These tests may take some time to run, so they are not enable by deafult

st_time = Dates.now()
st_time_str = Dates.format(st_time, "HH:MM")  
@info "Start time: $(st_time_str)"
@info("Running complete MLJ integration test suit for BetaML models (may take time)... ")
for i in 1:length(BetaML.MLJ_INTERFACED_MODELS)
    m =  BetaML.MLJ_INTERFACED_MODELS[i]
    in(m,[BetaML.Bmlj.MultitargetNeuralNetworkRegressor,BetaML.Bmlj.MultitargetGaussianMixtureRegressor]) && continue # skip these ones    
    @info "*******************\n**** Testing model: $(m) ..."
    MLJTestIntegration.test(m; mod=@__MODULE__, level=4, throw=true, verbosity=2)
end
end_time = Dates.now()
end_time_str = Dates.format(end_time, "HH:MM")  

t_length = end_time - st_time
hours = Int(floor(t_length.value ./ (1000*3600))) 
minutes = Int(floor((t_length.value - hours*1000*3600) ./ (1000*60)))
seconds = Int(floor((t_length.value - hours*1000*3600 - minutes*1000*60) ./ 1000))

t_length_str = "$(hours)h:$(minutes)m:$(seconds)s"
@info "End time: $(end_time_str) (elapsed: $(t_length_str))"