"""
    rmse(y::AbstractVector{<:Real}, ŷ::AbstractVector{<:Real})

Calculate the Root Mean Squared Error (RMSE) between the true values `y` and the predicted values `ŷ`.

RMSE is calculated as `sqrt(mean((y - ŷ).^2))`.

# Arguments
- `y::AbstractVector{<:Real}`: The vector of true values.
- `ŷ::AbstractVector{<:Real}`: The vector of predicted values.

# Returns
- `Float64`: The Root Mean Squared Error.

# Examples
```jldoctest
julia> rmse([1, 2, 3], [1.1, 2.2, 2.9])
0.14142135623730964
```
"""
function rmse(y::Vector{<:Real}, ŷ::Vector{<:Real})
    #check that y and ŷ are the same length, otherwise throw an error
    if length(y) != length(ŷ)
        throw(ArgumentError("y and ŷ must be the same length"))
    end

    return sqrt(sum((y .- ŷ) .^ 2) / length(y))
end

"""
    mse(y::AbstractVector{<:Real}, ŷ::AbstractVector{<:Real})

Calculate the Mean Squared Error (MSE) between the true values `y` and the predicted values `ŷ`.

MSE is calculated as `mean((y - ŷ).^2)`. This is a common loss function for regression problems.

# Arguments
- `y::AbstractVector{<:Real}`: The vector of true values.
- `ŷ::AbstractVector{<:Real}`: The vector of predicted values.

# Returns
- `Float64`: The Mean Squared Error.
"""
function mse(y::AbstractVector{<:Real}, ŷ::AbstractVector{<:Real})
    length(y) == length(ŷ) || throw(ArgumentError("y and ŷ must be the same length"))
    return sum((y .- ŷ) .^ 2) / length(y)
end

"""
    binary_crossentropy(y::AbstractVector{<:Integer}, ŷ::AbstractVector{<:Real})

Calculate the binary cross-entropy loss, commonly used for binary classification.

Loss is calculated as `-mean(y .* log.(ŷ) .+ (1 .- y) .* log.(1 .- ŷ))`. A small epsilon `eps` is used to avoid `log(0)`.

# Arguments
- `y::AbstractVector{<:Integer}`: The vector of true binary labels (0 or 1).
- `ŷ::AbstractVector{<:Real}`: The vector of predicted probabilities (between 0 and 1).

# Returns
- `Float64`: The binary cross-entropy loss.
"""
function binary_crossentropy(y::AbstractVector{<:Integer}, ŷ::AbstractVector{<:Real})
    length(y) == length(ŷ) || throw(ArgumentError("y and ŷ must be the same length"))
    # Add a small epsilon to prevent log(0) which is -Inf
    ϵ = 1e-15
    ŷ_clipped = clamp.(ŷ, ϵ, 1 - ϵ)
    return -mean(y .* log.(ŷ_clipped) .+ (1 .- y) .* log.(1 .- ŷ_clipped))
end