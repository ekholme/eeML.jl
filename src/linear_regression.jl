
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
