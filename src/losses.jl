"""
    binary_crossentropy

    binary_crossentropy(X::Matrix{<:Real}, y::AbstractVector{Bool}, b::Vector{<:Real})

A function to calculate the binary cross entropy loss
"""
function binary_crossentropy(X::Matrix{<:Real}, y::AbstractVector{Bool}, b::Vector{<:Real})

    ŷ = sigmoid.(X*b)

    ŷ = clamp.(ŷ, 1e-10, 1-1e-10)

    return mean(-(y .* log.(ŷ) .+ (1 .- y) .* log.(1 .- ŷ)))
end
