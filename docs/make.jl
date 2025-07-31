using eeML
using Documenter

DocMeta.setdocmeta!(eeML, :DocTestSetup, :(using eeML); recursive=true)

makedocs(;
    modules=[eeML],
    authors="Eric Ekholm <eric.ekholm@gmail.com> and contributors",
    sitename="eeML.jl",
    format=Documenter.HTML(;
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
