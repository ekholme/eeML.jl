module eeML

using Statistics, LinearAlgebra, Random #std lib
using ForwardDiff
using Distributions

export

    #types
    LinearRegression,
    LogisticRegression,

    #functions
    fit!,
    predict,
    rmse,
    r_squared,
    mse,
    binary_crossentropy,
    sigmoid,
    gradient_descent

#includes
include("linear_regression.jl")
include("logistic_regression.jl")
include("utils.jl")
include("loss_funcs.jl")
include("optimizers.jl")
include("score.jl")

end
