@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    x    = [1 3 10; 0.8 2.8 8; 1.2 3.2 12; 2 6 20; 1.8 5 18; 2.2 7 22; 0.5 1.5 5; 0.45 1.3 4; 0.55 1.8 6]
    y    = [0.5, 0.45, 0.55, 1, 0.9, 1.1, 0.25, 0.23, 0.27]
    ycat = ["b","b","b","a","a","a","c","c","c"]
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        @info "Beginning BetaML PrecompileTool workflow...."
        yoh  = fit!(OneHotEncoder(verbosity=NONE),ycat)
        fit!(NeuralNetworkEstimator(verbosity=NONE,epochs=10),x,y)
        fit!(NeuralNetworkEstimator(verbosity=NONE,epochs=10),x,yoh)
        fit!(RandomForestEstimator(verbosity=NONE,n_trees=5),x,y)
        fit!(RandomForestEstimator(verbosity=NONE,n_trees=5),x,ycat)
        fit!(PerceptronClassifier(verbosity=NONE,epochs=10),x,ycat)
        fit!(KernelPerceptronClassifier(verbosity=NONE,epochs=10),x,ycat)
        fit!(PegasosClassifier(verbosity=NONE,epochs=10),x,ycat)
        fit!(KMeansClusterer(verbosity=NONE),x)
        fit!(KMedoidsClusterer(verbosity=NONE),x)
        fit!(GMMClusterer(verbosity=NONE,tol=0.01),x)
        @info "...done BetaML PrecompileTool workflow."
    end
end
