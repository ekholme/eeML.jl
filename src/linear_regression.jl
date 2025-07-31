"""
    LinearRegression

A linear regression model object. The struct is mutable so that the `fit!` function can update the coefficients `β`.

# Fields
- `β::Vector{Float64}`: The coefficients of the linear model. It is empty until `fit!` is called.
"""
mutable struct LinearRegression
    β::Vector{Float64}
end

"""
    LinearRegression()

Constructs an untrained `LinearRegression` model. The coefficients `β` are initialized as an empty vector and will be populated by the `fit!` function.

# Examples
```jldoctest
julia> model = LinearRegression()
LinearRegression(Float64[])
```
"""
function LinearRegression()
    return LinearRegression(Float64[])
end


"""
    fit!(model::LinearRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})

Train the `LinearRegression` model using the feature matrix `X` and target vector `y`.

The coefficients `β` are calculated by solving the least-squares problem `Xβ = y`. This implementation uses Julia's `\\` operator, which is numerically stable and efficient. This function mutates the `model` object.

# Arguments
- `model::LinearRegression`: The model to be trained.
- `X::AbstractMatrix{<:Real}`: The matrix of features. It's common to include a column of ones for an intercept term.
- `y::AbstractVector{<:Real}`: The vector of target values.

# Returns
- `LinearRegression`: The trained model.
"""
function fit!(model::LinearRegression, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    size(X, 1) == length(y) || throw(DimensionMismatch("Number of rows in X must match the length of y."))
    # The \ operator is more numerically stable and efficient than inv(X'*X)*X'*y
    model.β = X \ y
    return model
end

"""
    predict(model::LinearRegression, X::AbstractMatrix{<:Real})

Make predictions using a trained `LinearRegression` model.

# Arguments
- `model::LinearRegression`: The trained model containing coefficients `β`.
- `X::AbstractMatrix{<:Real}`: The matrix of features for which to make predictions.
"""
function predict(model::LinearRegression, X::AbstractMatrix{<:Real})
    isempty(model.β) && error("Model has not been trained. Call fit! on the model first.")
    size(X, 2) == length(model.β) || throw(DimensionMismatch("Number of features in X must match the number of coefficients in the model."))
    return X * model.β
end

