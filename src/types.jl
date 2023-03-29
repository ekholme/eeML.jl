
abstract type EEmodel end

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