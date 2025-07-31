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


"""
    r_squared(y::Vector{<:Real}, ŷ::Vector{<:Real})

Calculate the R-squared (coefficient of determination) value.

R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It is calculated as `1 - (SS_res / SS_tot)`, where `SS_res` is the sum of squared residuals and `SS_tot` is the total sum of squares.

# Arguments
- `y::AbstractVector{<:Real}`: The vector of true values.
- `ŷ::AbstractVector{<:Real}`: The vector of predicted values.

# Returns
- `Float64`: The R-squared value.

# Examples
```jldoctest
julia> r_squared([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8])
0.98
```
"""
function r_squared(y::AbstractVector{<:Real}, ŷ::AbstractVector{<:Real})
    #check lengths
    if length(y) != length(ŷ)
        throw(ArgumentError("y and ŷ must be the same length"))
    end

    ȳ = Statistics.mean(y)
    ss_resid = sum((y .- ŷ) .^ 2)
    sst = sum((y .- ȳ) .^ 2)

    rsq = 1 - (ss_resid / sst)
    return rsq
end
# not really a loss function, but something we can optimize for at least in theory?
# this might get reorganized later