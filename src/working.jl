using eeML
using Distributions

#demo linear regression
X = randn(1000, 3)
b = [1, 2, 3]
ϵ = rand(Normal(0, .5), 1000)
y = X*b + ϵ

z = LinearRegression(𝐗 = X, y = y)

train!(z)

