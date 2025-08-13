mutable struct LogisticRegression
    β::Vector{Float64}
end

function LogisticRegression()
    return LogisticRegression(Float64[])
end

"""
    fit!(model::LogisticRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer}; kwargs...)

Train the `LogisticRegression` model using the feature matrix `X` and target vector `y`.

This function uses `gradient_descent` to find the optimal coefficients `β` that minimize the binary cross-entropy loss.

# Arguments
- `model::LogisticRegression`: The model to be trained.
- `X::AbstractMatrix{<:Real}`: The matrix of features. It's common to include a column of ones for an intercept term.
- `y::AbstractVector{<:Integer}`: The vector of target binary labels (0 or 1).

# Keywords
- `kwargs...`: Keyword arguments passed directly to the `gradient_descent` optimizer (e.g., `learning_rate`, `max_iter`).

# Returns
- `LogisticRegression`: The trained model.
"""
function fit!(model::LogisticRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer}; kwargs...)
    size(X, 1) == length(y) || throw(DimensionMismatch("Number of rows in X must match the length of y."))

    # This is the closure. The `loss` function "closes over" X and y.
    # It takes only one argument, `β`, as required by the optimizer.
    function loss(β)
        ŷ = sigmoid(X * β)
        return binary_crossentropy(y, ŷ)
    end

    # Initialize parameters and run the optimizer
    init_β = zeros(size(X, 2))
    model.β = gradient_descent(loss, init_β; kwargs...)

    return model
end

"""
    predict(model::LogisticRegression, X::AbstractMatrix{<:Real}; threshold::Union{Float64, Nothing}=0.5)

Make predictions using a trained `LogisticRegression` model. Returns class labels by default.

# Arguments
- `model::LogisticRegression`: The trained model.
- `X::AbstractMatrix{<:Real}`: The matrix of features for which to make predictions.
- `threshold::Union{Float64, Nothing}=0.5`: The probability threshold for classifying as 1. If `nothing`, raw probabilities are returned.
"""
function predict(model::LogisticRegression, X::AbstractMatrix{<:Real}; threshold::Union{Float64,Nothing}=0.5)
    isempty(model.β) && error("Model has not been trained. Call fit! on the model first.")
    size(X, 2) == length(model.β) || throw(DimensionMismatch("Number of features in X must match the number of coefficients in the model."))

    probabilities = sigmoid(X * model.β)
    isnothing(threshold) && return probabilities
    return (probabilities .>= threshold)
end
