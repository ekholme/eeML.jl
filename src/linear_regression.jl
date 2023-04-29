"""
    LinearRegression

    LinearRegression(𝐗::Matrix{<:Real}, y::AbstractVector{<:Real}})

A constructor function for a new LinearRegression object

# Arguments
- `𝐗`: the predictor matrix
- `y`: the target vector
"""
function LinearRegression(;𝐗, y)
    b = Vector{Float64}(undef, size(𝐗)[2])
    err = Vector{Float64}(undef, length(b))
    return LinearRegression(𝐗, y, b, err)
end

"""
    train!

    train!(model::LinearRegression)

In-place training of a linear regression model

# Arguments
- `model::LinearRegression`: a LinearRegression model (which includes a predictor matrix and a target vector) to be fit
"""
function train!(model::LinearRegression)
    𝐗 = model.𝐗
    y = model.y

    β̂ = (𝐗'*𝐗)^-1*(𝐗'*y)

    # calculate mean-square error
    mse = sum((y .- 𝐗*β̂).^2) 

    #calculate standard errors of betas
    err_b = sqrt.(diag(mse .* (𝐗'*𝐗)^-1))

    model.β̂ = β̂
    model.SEᵦ = err_b;
end
