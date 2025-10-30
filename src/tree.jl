mutable struct Node{T}
    featid::Int
    featval::T
end

mutable struct TreeRegression
    X::AbstractMatrix{<:Real}
    y::AbstractVector{<:Real}

end

#resume here

# use mse as the loss function for regression trees