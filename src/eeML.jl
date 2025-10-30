module eeML

using Statistics, LinearAlgebra, Random #std lib
using ForwardDiff
using Distributions

export

    #types
    LinearRegression,
    LogisticRegression,
    KnnRegression,
    KnnClassification,

    #functions
    fit!,
    predict,

    #loss funcs
    rmse,
    mse,
    binary_crossentropy,

    #scoring funcs
    r_squared,
    accuracy,

    #utils
    sigmoid,
    mode,
    gradient_descent

#includes
include("linear_regression.jl")
include("logistic_regression.jl")
include("utils.jl")
include("loss_funcs.jl")
include("optimizers.jl")
include("score.jl")
include("knn.jl")

end
