"""
    sigmoid(x)

Calculate the sigmoid function `1 / (1 + exp(-x))`.

This function is broadcasted element-wise for array inputs.
"""
function sigmoid(x::Real)
    return 1 / (1 + exp(-x))
end
sigmoid(x::AbstractArray) = sigmoid.(x)


"""
    mode(x::AbstractVector{<:Any})

Retrieve the mode of the input vector. This function will only return the first mode if there are multiple modes.
"""
function mode(x::AbstractVector{<:Any})
    #check for empty x
    isempty(x) && error("The input vector cannot be empty.")

    d = Dict{eltype(x),Int}()
    for i in x
        d[i] = get(d, i, 0) + 1
    end

    max_count = 0
    if !isempty(d)
        max_count = maximum(values(d))
    end

    # note that this returns only the first mode if there are multiple
    for (k, v) in d
        if v == max_count
            return k
        end
    end
end