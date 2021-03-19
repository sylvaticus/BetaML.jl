using Documenter, Literate, BetaML, Test

push!(LOAD_PATH,"../src/")


const _TUTORIAL_DIR = joinpath(@__DIR__, "src", "tutorials")
const _TUTORIAL_SUBDIR = [
    "Getting started",
    "A regression task: sharing bike demand prediction",
    "A classification task when labels are known: determining the plant species giving floreal measures",
]

function link_example(content)
    edit_url = match(r"EditURL = \"(.+?)\"", content)[1]
    footer = match(r"^(---\n\n\*This page was generated using)"m, content)[1]
    content = replace(
        content, footer => "[View this file on Github]($(edit_url)).\n\n" * footer
    )
    return content
end

function _file_list(full_dir, relative_dir, extension)
    return map(
        file -> joinpath(relative_dir, file),
        filter(file -> endswith(file, extension), sort(readdir(full_dir))),
    )
end

"""
    _include_sandbox(filename)
Include the `filename` in a temporary module that acts as a sandbox. (Ensuring
no constants or functions leak into other files.)
"""
function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function literate_directory(dir)
    rm.(_file_list(dir, dir, ".md"))
    for filename in _file_list(dir, dir, ".jl")
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        @testset "$(filename)" begin
            _include_sandbox(filename)
        end
        Literate.markdown(
            filename,
            dir;
            documenter = true,
            postprocess = link_example,
        )
    end
    return nothing
end


literate_directory.(joinpath.(_TUTORIAL_DIR, _TUTORIAL_SUBDIR))


makedocs(sitename="BetaML.jl Documentation",
         #root = "../",
         #source = "src",
         #build = "build",
         #=
         format = Documenter.HTML(
             # See https://github.com/JuliaDocs/Documenter.jl/issues/868
             prettyurls = get(ENV, "CI", nothing) == "true",
             analytics = "UA-44252521-1", # set it on Google Analytics
             collapselevel = 1,
         ),
         =#
         # `strict = true` causes Documenter to throw an error if the Doctests fail.
         #strict = true,

         authors = "Antonello Lobianco",
         pages = [
            "Index" => "index.md",
            "Perceptron" => "Perceptron.md",
            "Trees" => "Trees.md",
            "Nn"   => "Nn.md",
            "Clustering" => "Clustering.md",
            "Utils" => "Utils.md",
            "Tutorials" => map(
                subdir -> subdir => map(
                    file -> joinpath("tutorials", subdir, file),
                    filter(
                        file -> endswith(file, ".md"),
                        sort(readdir(joinpath(_TUTORIAL_DIR, subdir))),
                    ),
                ),
                _TUTORIAL_SUBDIR,
            ),
            "Examples" => "Examples.md"
         ],
         format = Documenter.HTML(prettyurls = false)
)
deploydocs(
    repo = "github.com/sylvaticus/BetaML.jl.git",
)
