using Distances
using Statistics: mean

"""
    KnnRegression

A k-Nearest Neighbors regressor model. The struct is mutable and stores the training data.

# Fields
- `X::AbstractMatrix{<:Real}`: The matrix of training features.
- `y::AbstractVector{<:Real}`: The vector of training target values.
"""
mutable struct KnnRegression
    X::AbstractMatrix{<:Real}
    y::AbstractVector{<:Real}
end

"""
    KnnRegression()

Constructs an untrained `KnnRegression` model. The training data fields `X` and `y` are initialized as empty and will be populated by the `fit!` function.
"""
function KnnRegression()
    return KnnRegression(Matrix{Float64}(undef, 0, 0), Float64[])
end

"""
    fit!(model::KnnRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})

"Trains" the `KnnRegression` model by storing the feature matrix `X` and target vector `y`.

In k-NN, fitting is a lazy process that simply consists of memorizing the training data.

# Arguments
- `model::KnnRegression`: The model to be trained.
- `X::AbstractMatrix{<:Real}`: The matrix of training features.
- `y::AbstractVector{<:Real}`: The vector of training target values.

# Returns
- `KnnRegression`: The trained model containing the training data.
"""
function fit!(model::KnnRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    # check that dimensions match
    size(X, 1) == length(y) || throw(DimensionMismatch("Number of rows in X must match the length of y."))

    model.X = X
    model.y = y
    return model
end

"""
    predict(model::KnnRegression, X::AbstractMatrix{<:Real}, k::Int=5)

Make predictions using a trained `KnnRegression` model.

For each sample in `X`, the prediction is the mean of the target values of its `k` nearest neighbors in the training data.

# Arguments
- `model::KnnRegression`: The trained model containing the training data.
- `X::AbstractMatrix{<:Real}`: The matrix of features for which to make predictions.
- `k::Int=5`: The number of nearest neighbors to consider.

# Returns
- `Vector{Float64}`: The vector of predicted values.
"""
function predict(model::KnnRegression, X::AbstractMatrix{<:Real}, k::Int=5)
    # check that model is fitted
    isempty(model.y) && error("Model has not been trained. Call fit! on the model first.")

    #check inputs
    _check_knn_inputs(model.X, X, k)

    #calculate distance matrix using pairwise euclidean distance
    dist_mat = pairwise(Euclidean(), X', model.X')

    #create an empty array to store predictions
    preds = zeros(size(X, 1))

    #predict the knn regression for each row in X
    for i in eachindex(preds)
        ind = sortperm(dist_mat[i, :])
        preds[i] = mean(model.y[ind[1:k]])
    end

    return preds
end

# KnnClassification ----------------------

"""
    KnnClassification

A k-Nearest Neighbors classifier model. The struct is mutable and stores the training data.

# Fields
- `X::AbstractMatrix{<:Real}`: The matrix of training features.
- `y::AbstractVector{<:Any}`: The vector of training labels.
"""
mutable struct KnnClassification
    X::AbstractMatrix{<:Real}
    y::AbstractVector{<:Any}
end

"""
    KnnClassification()

Constructs an untrained `KnnClassification` model. The training data fields `X` and `y` are initialized as empty and will be populated by the `fit!` function.
"""
function KnnClassification()
    return KnnClassification(Matrix{Float64}(undef, 0, 0), Any[])
end

"""
    fit!(model::KnnClassification, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Any})

"Trains" the `KnnClassification` model by storing the feature matrix `X` and label vector `y`.

In k-NN, fitting is a lazy process that simply consists of memorizing the training data.

# Arguments
- `model::KnnClassification`: The model to be trained.
- `X::AbstractMatrix{<:Real}`: The matrix of training features.
- `y::AbstractVector{<:Any}`: The vector of training labels.

# Returns
- `KnnClassification`: The trained model containing the training data.
"""
function fit!(model::KnnClassification, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Any})
    # check that dimensions match
    size(X, 1) == length(y) || throw(DimensionMismatch("Number of rows in X must match the length of y."))

    model.X = X
    model.y = y
    return model
end

"""
    predict(model::KnnClassification, X::AbstractMatrix{<:Real}, k::Int=5)

Make predictions using a trained `KnnClassification` model.

For each sample in `X`, the prediction is the most frequent class (mode) among its `k` nearest neighbors in the training data.

# Arguments
- `model::KnnClassification`: The trained model containing the training data.
- `X::AbstractMatrix{<:Real}`: The matrix of features for which to make predictions.
- `k::Int=5`: The number of nearest neighbors to consider.

# Returns
- `AbstractVector`: The vector of predicted labels, with the same element type as `model.y`.
"""
function predict(model::KnnClassification, X::AbstractMatrix{<:Real}, k::Int=5)
    #check that model has been fit
    isempty(model.y) && error("Model has not been trained. Call fit! on the model first.")

    #check inputs
    _check_knn_inputs(model.X, X, k)

    dist_mat = pairwise(Euclidean(), X', model.X')
    preds = Vector{eltype(model.y)}(undef, size(X, 1))


    for i in eachindex(preds)
        ind = sortperm(dist_mat[i, :])
        # note that mode() is defined in my utils.jl file
        preds[i] = mode(model.y[ind[1:k]])
    end

    return preds
end


# Helpers -----------------
"""
    _check_knn_inputs(train_x, test_x, k)

Internal helper to perform input validation for k-NN prediction.

Checks that the number of features in the test set matches the training set,
and that `k` is a valid number (i.e., between 1 and the number of training samples).
"""
function _check_knn_inputs(train_x::AbstractMatrix{<:Real}, test_x::AbstractMatrix{<:Real}, k::Int)
    size(train_x, 2) == size(test_x, 2) || throw(DimensionMismatch("Number of columns in X must match the number of columns in the training data."))

    #check k
    n_samples = size(train_x, 1)
    1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of rows in the training data ($n_samples)."))

end