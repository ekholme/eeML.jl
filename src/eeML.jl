module eeML

using Statistics, LinearAlgebra, Random #std lib

export

    #types
    LinearRegression,

    #functions
    fit!,
    predict,
    rmse,
    r_squared

#includes
include("linear_regression.jl")
include("loss_funcs.jl")

end
