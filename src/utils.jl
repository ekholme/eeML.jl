"""
    sigmoid(x)

Calculate the sigmoid function `1 / (1 + exp(-x))`.

This function is broadcasted element-wise for array inputs.
"""
function sigmoid(x::Real)
    return 1 / (1 + exp(-x))
end
sigmoid(x::AbstractArray) = sigmoid.(x)