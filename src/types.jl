
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

"""
    LogisticRegression

A container for a logistic regression model.
Contains the predictor matrix, the targets, and the coefficients

# Fields

- `𝐗::Matrix{<:Real}`: the predictor matrix
- `y::AbstractVector{<:Bool}`: the target vector
- `β̂::Vector{<:Real}`: the estimated coefficients
"""
mutable struct LogisticRegression <: EEmodel
    𝐗::Matrix{<:Real}
    y::AbstractVector{Bool}
    β̂::Vector{<:Real}
end
#todo -- include betas and std errs