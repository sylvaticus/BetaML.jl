using Documenter, Bmlt.Utils, Bmlt.Nn, Bmlt.Perceptron, Bmlt.Clustering

push!(LOAD_PATH,"../src/")
makedocs(sitename="Beta Machine Learning Toolkit Documentation",
         #root = "../",
         #source = "src",
         #build = "build",
         pages = [
            "Index" => "index.md",
            "Perceptron" => "Perceptron.md",
            "Nn"   => "Nn.md",
            "Clustering" => "Clustering.md"
         ],
         format = Documenter.HTML(prettyurls = false)
)
