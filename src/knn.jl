mutable struct KnnRegression
    X::AbstractMatrix{<:Real}
    y::AbstractArray{<:Real}
end

function fit!(model::KnnRegression, X::AbstractMatrix{<:Real}, y::AbstractArray{<:Real})
    model.X = X
    model.y = y
    return model
end

function predict(model::KnnRegression, X::AbstractMatrix{<:Real}, k::Int=5)
    nobs = length(X)
    for i in 1:nobs
        row = X[i, :]
        distances = Vector{Float64}(undef, nobs)
        #RESUME HERE -- THIS MIGHT NOT BE THE RIGHT APPROACH. I CAN PROBABLY JUST DIRECTLY COMPUTE THE DISTANCES USING BOTH THE X AND MODEL.X MATRICES
    end
end