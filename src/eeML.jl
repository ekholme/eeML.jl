module eeML

#dependencies
using Statistics
using Random
using LinearAlgebra

#includes
include("types.jl")
include("linear_regression.jl")


#exports
export
EEmodel,
LinearRegression,

train!

# Write your package code here.

end
