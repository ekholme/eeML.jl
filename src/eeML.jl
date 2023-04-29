module eeML

#dependencies
using Statistics
using Random
using LinearAlgebra
using ForwardDiff

#includes
include("types.jl")
include("linear_regression.jl")
include("logistic_regression.jl")
include("losses.jl")

#exports
export
EEmodel,
LinearRegression,
LogisticRegression,

train!
binary_crossentropy

end
