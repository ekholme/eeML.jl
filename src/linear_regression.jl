
function train!(model::LinearRegression)
    𝐗 = model.𝐗
    y = model.y

    return (𝐗, y)
    # β̂ = (𝐗'*𝐗)^-1*𝐗*y

    # return β̂
    #calculate mean-square error
    # mse = sum((y .- 𝐗*β̂).^2) 

    # return mse
    # err_b = sqrt.(diag(mse .* (𝐗'*𝐗)^-1))

# return (β=β̂, std_err = err_b)
end
