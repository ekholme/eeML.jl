using ForwardDiff, LinearAlgebra

"""
    gradient_descent(loss_func, initial_β; learning_rate=0.01, tol=1e-6, max_iter=1_000, verbose=false)

Perform gradient descent to find the parameters `β` that minimize the `loss_func`.

This is a general-purpose optimizer. The `loss_func` must be a function that takes a single vector argument (the parameters `β`) and returns a scalar loss. For supervised learning problems, you should create a closure that captures your data (`X`, `y`).

# Arguments
- `loss_func`: A callable function that accepts a vector `β` and returns a scalar loss.
- `initial_β::AbstractVector{<:Real}`: The starting values for the parameters `β`.

# Keywords
- `learning_rate::Float64=0.01`: The step size for each iteration.
- `tol::Float64=1e-6`: The tolerance for convergence. The algorithm stops when the L2 norm of the change in `β` is less than this value.
- `max_iter::Int=1_000`: The maximum number of iterations.
- `verbose::Bool=false`: If `true`, prints the iteration number and loss at each step.

# Returns
- `Vector{Float64}`: The optimized parameters `β`.

# Example

### 1. Linear Regression

```jldoctest; setup = :(using Random; Random.seed!(123))
# Define a mean squared error loss function for a linear model
X = hcat(ones(100), rand(100, 1)) # Add intercept
true_β = [1.5, -3.0]
y = X * true_β + 0.2 * randn(100)
loss(β) = sum((y - X * β).^2) / length(y)

# Initial guess for parameters
initial_β = rand(2)

# Run optimizer
β_optimized = gradient_descent(loss, initial_β, learning_rate=0.1);
isapprox(β_optimized, true_β, atol=0.1)

# output

true
```

### 2. Logistic Regression

For logistic regression, we create a closure around the binary cross-entropy loss. This is exactly what the `fit!` method for `LogisticRegression` does internally.
```julia
# The loss function captures X and y from its environment
loss_logistic(β) = binary_crossentropy(y_binary, sigmoid(X * β))

# Then you would call the optimizer:
# β_logistic = gradient_descent(loss_logistic, initial_β_logistic)
```
"""
function gradient_descent(loss_func, initial_β::AbstractVector{<:Real}; learning_rate=0.01, tol=1e-6, max_iter=1_000, verbose=false)
    β = convert(Vector{Float64}, initial_β) # Work with a mutable Float64 copy

    # Define the gradient function using ForwardDiff
    g = b -> ForwardDiff.gradient(loss_func, b)

    iter = 0
    for i in 1:max_iter
        iter = i
        grad = g(β)
        β_new = β - learning_rate * grad

        # Check for convergence using the L2 norm of the change in β
        if norm(β_new - β) < tol
            verbose && println("Converged after $i iterations.")
            return β_new
        end
        β = β_new

        verbose && println("Iteration: $i, Loss: $(loss_func(β))")
    end

    @warn "Maximum iterations ($max_iter) reached without convergence."
    return β
end