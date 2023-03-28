
abstract type EEmodel end

struct LinearRegression <: EEmodel 
    𝐗::Matrix{<:Real}
    y::AbstractVector{<:Real}
end

z = LinearRegression(𝐗, y)