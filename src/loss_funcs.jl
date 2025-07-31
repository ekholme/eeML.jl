"""
    rmse(y::Vector{<:Real}, ŷ::Vector{<:Real})

Calculate the Root Mean Squared Error (RMSE) between the true values `y` and the predicted values `ŷ`.

RMSE is calculated as `sqrt(mean((y - ŷ).^2))`.

# Arguments
- `y::Vector{<:Real}`: The vector of true values.
- `ŷ::Vector{<:Real}`: The vector of predicted values.

# Returns
- `Float64`: The Root Mean Squared Error.

# Examples
```jldoctest
julia> rmse([1, 2, 3], [1.1, 2.2, 2.9])
0.15275252316519468
```
"""
function rmse(y::Vector{<:Real}, ŷ::Vector{<:Real})
    #check that y and ŷ are the same length, otherwise throw an error
    if length(y) != length(ŷ)
        throw(ArgumentError("y and ŷ must be the same length"))
    end

    return sqrt(sum((y .- ŷ) .^ 2) / length(y))
end