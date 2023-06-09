using eeML
using Documenter

DocMeta.setdocmeta!(eeML, :DocTestSetup, :(using eeML); recursive=true)

makedocs(;
    modules=[eeML],
    authors="Eric Ekholm",
    repo="https://github.com/ekholme/eeML.jl/blob/{commit}{path}#{line}",
    sitename="eeML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ekholme.github.io/eeML.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ekholme/eeML.jl",
    devbranch="master",
)
