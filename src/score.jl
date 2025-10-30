# todo -- implement scoring functions
# e.g. accuracy, precision

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

"""
    accuracy(y::AbstractVector, ŷ::AbstractVector)

Calculate the classification accuracy.

Accuracy is the proportion of correct predictions, calculated as `(number of correct predictions) / (total number of predictions)`.

# Arguments
- `y::AbstractVector`: The vector of true labels.
- `ŷ::AbstractVector`: The vector of predicted labels.

# Returns
- `Float64`: The accuracy score, a value between 0.0 and 1.0.

# Examples
```jldoctest
julia> accuracy([1, 2, 3, 4], [1, 2, 4, 4])
0.75
```
"""
function accuracy(y::AbstractVector{<:Any}, ŷ::AbstractVector{<:Any})
    #check lengths
    if length(y) != length(ŷ)
        throw(ArgumentError("y and ŷ must be the same length"))
    end

    n = length(y)
    acc = sum(y .== ŷ) / n

    return acc
end