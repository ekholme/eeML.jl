using Distances
using Statistics: mean

mutable struct KnnRegression
    X::AbstractMatrix{<:Real}
    y::AbstractVector{<:Real}
end

function KnnRegression()
    return KnnRegression(Matrix{Float64}(undef, 0, 0), Float64[])
end

function fit!(model::KnnRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    # check that dimensions match
    size(X, 1) == length(y) || throw(DimensionMismatch("Number of rows in X must match the length of y."))

    model.X = X
    model.y = y
    return model
end

function predict(model::KnnRegression, X::AbstractMatrix{<:Real}, k::Int=5)
    # check that model is fitted
    isempty(model.y) && error("Model has not been trained. Call fit! on the model first.")

    # check that dimensions match
    size(X, 2) == size(model.X, 2) || throw(DimensionMismatch("Number of columns in X must match the number of columns in the training data."))

    # check k
    n_samples = size(model.X, 1)
    1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of training samples ($n_samples), but got $k."))

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