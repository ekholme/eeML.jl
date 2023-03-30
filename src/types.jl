
abstract type EEmodel end

"""
    LinearRegression

A container for a linear regression model.
Contains the predictor matrix, the targets, the coefficients, and the standard errors of the coefficients.

# Fields

- `𝐗::Matrix{<:Real}`: the predictor matrix
- `y::AbstractVector{<:Real}`: the target vector
- `β̂::Vector{<:Real}`: the estimated coefficients
- `SEᵦ::Vector{<:Real}`: the standard errors of the estimated coefficients
"""
mutable struct LinearRegression <: EEmodel 
    𝐗::Matrix{<:Real}
    y::AbstractVector{<:Real}
    β̂::Vector{<:Real}
    SEᵦ::Vector{<:Real}
end

#constructor function to intialize without betas and std errs
function LinearRegression(;𝐗, y)
    b = Vector{Float64}(undef, size(𝐗)[2])
    err = Vector{Float64}(undef, length(b))
    return LinearRegression(𝐗, y, b, err)
end