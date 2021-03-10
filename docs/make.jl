using Documenter, Literate, BetaML.Utils, BetaML.Nn, BetaML.Perceptron, BetaML.Clustering, BetaML.Trees

push!(LOAD_PATH,"../src/")
makedocs(sitename="BetaML.jl Documentation",
         #root = "../",
         #source = "src",
         #build = "build",
         pages = [
            "Index" => "index.md",
            "Perceptron" => "Perceptron.md",
            "Trees" => "Trees.md",
            "Nn"   => "Nn.md",
            "Clustering" => "Clustering.md",
            "Utils" => "Utils.md",
            "Examples" => "Examples.md"
         ],
         format = Documenter.HTML(prettyurls = false)
)
deploydocs(
    repo = "github.com/sylvaticus/BetaML.jl.git",
)
