using Documenter, BetaML.Utils, BetaML.Nn, BetaML.Perceptron, BetaML.Clustering

push!(LOAD_PATH,"../src/")
makedocs(sitename="BetaML.jl Documentation",
         #root = "../",
         #source = "src",
         #build = "build",
         pages = [
            "Index" => "index.md",
            "Perceptron" => "Perceptron.md",
            "Nn"   => "Nn.md",
            "Clustering" => "Clustering.md",
            "Utils" => "Utils.md",
            "Notebooks" => "Notebooks.md"
         ],
         format = Documenter.HTML(prettyurls = false)
)
deploydocs(
    repo = "github.com/sylvaticus/BetaML.jl.git",
)
