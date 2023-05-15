"""
    LogisticRegression(𝐗::Matrix{<:Real}, y::AbstractVector{<:Real}})

A constructor function for a new LogisticRegression object

# Arguments
- `𝐗`: the predictor matrix
- `y`: the target vector
"""
function LogisticRegression(;𝐗, y)
    b = Vector{Float64}(undef, size(𝐗)[2])
    return LogisticRegression(𝐗, y, b)
end


"""
    train!(model::LogisticRegression)

In-place training of a logistic regression model

#Arguments
- `model::LogisticRegression`: a LogisticRegression model (which includes a predictor matrix and a target vector) to be fit
"""
function train!(model::LogisticRegression; lr = .01, max_iter = 1_000, tol = .01, noisy = false)
    #training via gradient descent
    β = randn(size(model.𝐗, 2) + 1)
    iter = 0
    err = 1e10

    X = hcat(ones(size(model.𝐗, 1)), model.𝐗)
    y = model.y

   d(b) = ForwardDiff.gradient(params -> binary_crossentropy(X, y, params), b)
   
   while err > tol && iter < max_iter
    β -= lr*d(β)
    err = binary_crossentropy(X, y, β)
        if (noisy == true)
            println("Iteration $(iter): current error is $(err)")
        end
    iter += 1
    end
    model.β̂ = β
end


#sigmoid function
#internal helper function
function sigmoid(X)
    return exp(X) / (1 + exp(X))
end

