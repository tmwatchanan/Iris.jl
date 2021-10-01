### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 206aad4b-fd9b-4648-9fc8-522c77780768
using Random

# ╔═╡ d2d84ac0-2185-11ec-36d6-ffa0e3e50e84
md"""
# Watchanan Chantapakul (wcgzm)
"""

# ╔═╡ eb9dcbe6-d98c-4922-ba23-01eec2601ad3
md"""
## Assignment 4: Polynomial Regression
- Write programs in Matlab, R, C/C++, Java, Perl, or Python to implement the analytical (e.g. matrix-based) or iterative (e.g. gradient descent) linear regression algorithm to solve a polynomial regression problem of predicting the weight (y) of woman from their height (x).
- The relationship between y and x is assumed to be ``y = b_0 + b_1 * x + b_2 * X^2  + noise``. The height of the women is 1.47, 1.5, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.7, 1.73, 1.75, 1.78, 1.8, 1.83.  The corresponding weight of the women is 52.21, 53.12, 54.48, 55.84, 57.2, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.1, 69.92, 72.19, 74.46.
- Don’t directly call linear regression functions in any software to obtain the results. Turn in your programs and execution results.
------
"""

# ╔═╡ fc7d4e74-db46-4a8b-a9b1-c45e88315250
md"""
Import some libraries.
"""

# ╔═╡ e254e24b-4a33-457c-b39e-59a2ef0f40c6
md"""
Create two variables, `heights` and `weights`, with the provided data.
"""

# ╔═╡ 6ab5076c-28e7-4971-99d9-32b6b003c7bd
heights = [1.47, 1.5, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.7, 1.73, 1.75, 1.78, 1.8, 1.83]

# ╔═╡ d6fdafcc-2932-4f5a-a14a-a1f946b91c1b
weights = [52.21, 53.12, 54.48, 55.84, 57.2, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.1, 69.92, 72.19, 74.46]

# ╔═╡ a6a11584-5a13-4c33-aa5c-fea1548f9b97
md"""
We create a matrix $X$ containings all the training patterns. We need to have an extra column preceding the data filled with unities. The added column helps us to do matrix multiplication between the inputs and weights with ease.
"""

# ╔═╡ fc9a6862-0a0e-4582-ab39-752af2b78271
X = [ones(length(heights)) heights heights.^2]

# ╔═╡ 030fa246-48b5-46da-93f0-509f2c49b1fa
md"""
Our target values (weights to be predicted) are contained in the matrix $Y$.
"""

# ╔═╡ 8b74e67a-ebaa-4bdf-8634-819cb9e1bfe4
Y = weights

# ╔═╡ a00154e2-8c08-4eea-ba9b-1df60c14a940
md"""
### Least Square Method

Let $f(\vec{x})$ be a linear regression model defined with a set of weights ($\vec{\beta}$) that takes an input vector $\vec{x}$ as follows:

$f(\vec{x}) = \vec{x} \vec{\beta} = y$

where $\vec{x} = \begin{bmatrix} x_0 & x_1 & \dots & x_p \end{bmatrix}$, and $\vec{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \dots \\ \beta_p \end{bmatrix}$.

We would like to find a set of weights ($\hat{\beta}$) that minimizes the error, specifically in this case, sum of squared residuals. A residual is the difference between the predicted value and the true value or $f(\vec{x}) - \hat{y}$. Mathematically speaking, we need to find:

$\hat{\beta} = \underset{\vec{\beta}}{\mathrm{argmin}} J(\vec{\beta})$

$\begin{align}
J(\vec{\beta})
&=\underbrace{\sum_{i=1}^{n} (X_i \vec{\beta} - Y_i)^2}_\text{sum of squared residuals} \\
&= (X \vec{\beta} - Y)^\mathsf{T} (X \vec{\beta} -\ Y) \\
&= \left[(X \vec{\beta})^\mathsf{T} (X \vec{\beta}) \right] + \left[(X \vec{\beta})^\mathsf{T} (-Y)\right] + \left[(-Y)^{\mathsf{T}} (X \vec{\beta})\right] +\left[(-Y)^{\mathsf{T}} (-Y)\right] \\
&= \vec{\beta}^\mathsf{T} X^\mathsf{T} X \vec{\beta} - \vec{\beta}^\mathsf{T} X^\mathsf{T} Y - Y^\mathsf{T} X \vec{\beta} + Y^\mathsf{T} Y \\
\end{align}$

In order to tackle an optimization problem, one of the commonly used way is to use the knowledge from calculus where we set the derivative to zero to find the extrema of the function. We arrive at:

$\begin{align}
\frac{\partial J(\vec{\beta})}{\partial \vec{\beta}} &= 0 \\
\frac{\partial \left(\vec{\beta}^\mathsf{T} X^\mathsf{T} X \vec{\beta} - \vec{\beta}^\mathsf{T} X^\mathsf{T} Y - Y^\mathsf{T} X \vec{\beta} + \overbrace{Y^\mathsf{T} Y}^\text{constant} \right)}{\partial \vec{\beta}} &= 0 \\
2X^\mathsf{T} X \vec{\beta} - X^\mathsf{T} Y - (Y^\mathsf{T} X)^\mathsf{T} + 0  &= 0 \\
2X^\mathsf{T} X \vec{\beta} - X^\mathsf{T} Y - X^\mathsf{T} Y &= 0 \\
2X^\mathsf{T} X \vec{\beta} - 2X^\mathsf{T} Y &= 0 \\
2X^\mathsf{T} X \vec{\beta} &= 2X^\mathsf{T} Y \\
X^\mathsf{T} X \vec{\beta} &= X^\mathsf{T} Y \\
\underbrace{(X^\mathsf{T} X)^{-1} X^\mathsf{T} X}_{\text{cancelled}} \vec{\beta} &= (X^\mathsf{T} X)^{-1} X^\mathsf{T} Y \\
\hat{\beta} &= \underset{p \times p}{(X^{\mathsf{T}}X)}^{-1} \underset{p \times 1}{X^\mathsf{T} Y}
\end{align}$

where
- The matrix $X$ contains independent variables
- The matrix $Y$ contains dependent variables
- ``p`` is the number of parameters.

So, we can use the equation $\hat{\beta} = (X^{\mathsf{T}}X)^{-1} X^\mathsf{T} Y$ to find the weights. Note that, it requires $(X^{\mathsf{T}}X)$ to be invertible which happens when the $(X^{\mathsf{T}}X)$ is a full rank matrix.


"""

# ╔═╡ d8d45e93-a889-49ad-ab0f-0950c2ca586d
β̂ = (X' * X)^(-1) * X' * Y

# ╔═╡ 5f7753df-5afd-4e02-a146-cd2a6c6d5902
size(X), size(β̂)

# ╔═╡ 6fdc2c00-4fcf-4dcc-9da7-aead992daa84
X * β̂

# ╔═╡ c494c820-e67f-4301-8abf-d0fa59a8bf9f
md"""
After solving for $\hat{\beta}$ using the least square method, we get the following weights:

``\hat{\beta}_0`` = $(β̂[1])

``\hat{\beta}_1`` = $(β̂[2])

``\hat{\beta}_2`` = $(β̂[3])

The $\hat{\beta}_0$ is the bias or intercept.
"""

# ╔═╡ ae685d58-39e2-4123-ba0c-17e58b3e8ebe
error_least_square = sum((X * β̂ - Y).^2)

# ╔═╡ 754b0222-8ff8-4d3d-b3f9-ab1b5f1a09aa
md"""
We get the sum squared error of $error_least_square which is very good (the lower, the better).
"""

# ╔═╡ bd8c1b84-0a29-4cc7-a633-30b001d5215d
md"""
### Gradient Descent Method

$\begin{align}
\frac{\partial J(\vec{\beta})}{\partial \vec{\beta}}
&= \frac{\partial \left(\vec{\beta}^\mathsf{T} X^\mathsf{T} X \vec{\beta} - \vec{\beta}^\mathsf{T} X^\mathsf{T} Y - Y^\mathsf{T} X \vec{\beta} + \overbrace{Y^\mathsf{T} Y}^\text{constant} \right)}{\partial \vec{\beta}} \\
&= 2X^\mathsf{T} X \vec{\beta} - X^\mathsf{T} Y - (Y^\mathsf{T} X)^\mathsf{T} + 0 \\
&= 2X^\mathsf{T} X \vec{\beta} - X^\mathsf{T} Y - X^\mathsf{T} Y \\
&= 2X^\mathsf{T} X \vec{\beta} - 2X^\mathsf{T} Y \\
&= 2 \left(X^\mathsf{T} X \vec{\beta} - X^\mathsf{T} Y \right) \\
&= 2 X^\mathsf{T} \left(X \vec{\beta} - Y \right) \\
\end{align}$

Gradint descent algorithm is an iterative algorithm that we use for finding local minimum of a differentiable function which is defined as:

$\begin{align}
\vec{\beta}(t+1)
&= \vec{\beta}(t) - \frac{\alpha}{2} \frac{\partial J(\vec{\beta})}{\partial \vec{\beta}} \\
&= \vec{\beta}(t) - \frac{\alpha}{2} 2 X^\mathsf{T} \left(X \vec{\beta} - Y \right) \\
&= \vec{\beta}(t) - \alpha X^\mathsf{T} \left(X \vec{\beta} - Y \right) \\
\end{align}$

So, we are going to use the equation above to update the weights $\vec{\beta}$ in each iteration until it reaches the maximum training iteration $T$ (our stopping criterion).

"""

# ╔═╡ caf9f75e-37f6-49c1-b7e9-1045850ff55a
begin
	Random.seed!(8725)
	w⃗ = rand(3)
end

# ╔═╡ 71006099-049d-4878-bdd6-804402c142f6
w⃗

# ╔═╡ d95ce2da-f985-4b92-bcb1-5a310586438b
md"""
Define a learning rate $\alpha$ for gradient descent:
"""

# ╔═╡ 1afa8578-6026-4f18-91fb-af9ec19d1d3b
α = 0.01

# ╔═╡ c47dbedd-6f2f-4b13-b12e-743966ea900b
T₁ = 1000

# ╔═╡ 56971fdd-8956-4912-9aa0-88e92bd53f01
md"""
We use the learning rate ``\alpha =`` $α and the maximum number of iterations/epochs ``T_1 =`` $T₁
"""

# ╔═╡ e1d6e8c5-2894-48ae-8278-f7fc0e6f5516
begin
	w⃗_gradient_descent = w⃗
	for t = 1:T₁
		w⃗_gradient_descent = w⃗_gradient_descent - α * X' * (X * w⃗_gradient_descent - Y)
	end
end

# ╔═╡ 5a2c12f7-a870-4786-98e3-ddabb99e2b30
w⃗_gradient_descent

# ╔═╡ a14a4ec8-8914-4cc0-97a0-64b40e63bafa
md"""
After solving for $\hat{\beta}$ using the gradient descent algorithm for $T₁ epochs, we get the following weights:

``\hat{\beta}_0`` = $(w⃗_gradient_descent[1])

``\hat{\beta}_1`` = $(w⃗_gradient_descent[2])

``\hat{\beta}_2`` = $(w⃗_gradient_descent[3])
"""

# ╔═╡ 763c0dc9-2eed-4349-8a55-972a75c6aab7
error_gradient_descent = sum((X * w⃗_gradient_descent - Y).^2)

# ╔═╡ 07972683-d848-46a4-ae6e-a5fb71305e54
md"""
It seems like the trained weights resulting from the gradient descent algorithm with 1,000 iterations does not produce a very good result in comparison to the least square method. The sum squared error is $error_gradient_descent.
"""

# ╔═╡ 07347674-3b85-4b77-aa22-643d1134a607
T₂ = 10000000

# ╔═╡ b6b78d48-b678-4ee2-9f67-91e8698f452e
md"""
Therefore, let's train it using more number of iterations/epochs. Now, we set ``T = `` $T₂.
"""

# ╔═╡ 3f2e24d1-b9c8-4ef2-854b-a5265cad236d
begin
	w⃗_trained = w⃗
	for t = 1:T₂
		w⃗_trained = w⃗_trained - α * X' * (X * w⃗_trained - Y)
	end
end

# ╔═╡ eca01260-1dcd-4662-8fac-c924eaaad11c
w⃗_trained

# ╔═╡ 6ff93c38-3a1b-4f9c-b386-2bb74c5fa978
md"""
After solving for $\hat{\beta}$ using the gradient descent method for $T₂ epochs, we get the following weights:

``\hat{\beta}_0`` = $(w⃗_trained[1])

``\hat{\beta}_1`` = $(w⃗_trained[2])

``\hat{\beta}_2`` = $(w⃗_trained[3])
"""

# ╔═╡ 96763296-7283-4975-b184-dc964d221802
error_gradient_descent_long = sum((X * w⃗_trained - Y).^2)

# ╔═╡ 088f2ad1-4849-44bc-9a97-0c34cece2a06
md"""
Wonderful! We get the result close to the least square approach. The sum squared error, $error_gradient_descent_long, is very low now. This means our model trained with the gradient descent is decent.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
"""

# ╔═╡ Cell order:
# ╟─d2d84ac0-2185-11ec-36d6-ffa0e3e50e84
# ╟─eb9dcbe6-d98c-4922-ba23-01eec2601ad3
# ╟─fc7d4e74-db46-4a8b-a9b1-c45e88315250
# ╠═206aad4b-fd9b-4648-9fc8-522c77780768
# ╟─e254e24b-4a33-457c-b39e-59a2ef0f40c6
# ╠═6ab5076c-28e7-4971-99d9-32b6b003c7bd
# ╠═d6fdafcc-2932-4f5a-a14a-a1f946b91c1b
# ╟─a6a11584-5a13-4c33-aa5c-fea1548f9b97
# ╠═fc9a6862-0a0e-4582-ab39-752af2b78271
# ╟─030fa246-48b5-46da-93f0-509f2c49b1fa
# ╠═5f7753df-5afd-4e02-a146-cd2a6c6d5902
# ╠═6fdc2c00-4fcf-4dcc-9da7-aead992daa84
# ╠═8b74e67a-ebaa-4bdf-8634-819cb9e1bfe4
# ╟─a00154e2-8c08-4eea-ba9b-1df60c14a940
# ╠═d8d45e93-a889-49ad-ab0f-0950c2ca586d
# ╟─c494c820-e67f-4301-8abf-d0fa59a8bf9f
# ╠═ae685d58-39e2-4123-ba0c-17e58b3e8ebe
# ╟─754b0222-8ff8-4d3d-b3f9-ab1b5f1a09aa
# ╟─bd8c1b84-0a29-4cc7-a633-30b001d5215d
# ╠═caf9f75e-37f6-49c1-b7e9-1045850ff55a
# ╠═71006099-049d-4878-bdd6-804402c142f6
# ╟─d95ce2da-f985-4b92-bcb1-5a310586438b
# ╠═1afa8578-6026-4f18-91fb-af9ec19d1d3b
# ╠═c47dbedd-6f2f-4b13-b12e-743966ea900b
# ╟─56971fdd-8956-4912-9aa0-88e92bd53f01
# ╠═e1d6e8c5-2894-48ae-8278-f7fc0e6f5516
# ╠═5a2c12f7-a870-4786-98e3-ddabb99e2b30
# ╟─a14a4ec8-8914-4cc0-97a0-64b40e63bafa
# ╠═763c0dc9-2eed-4349-8a55-972a75c6aab7
# ╟─07972683-d848-46a4-ae6e-a5fb71305e54
# ╠═07347674-3b85-4b77-aa22-643d1134a607
# ╟─b6b78d48-b678-4ee2-9f67-91e8698f452e
# ╠═3f2e24d1-b9c8-4ef2-854b-a5265cad236d
# ╠═eca01260-1dcd-4662-8fac-c924eaaad11c
# ╟─6ff93c38-3a1b-4f9c-b386-2bb74c5fa978
# ╠═96763296-7283-4975-b184-dc964d221802
# ╟─088f2ad1-4849-44bc-9a97-0c34cece2a06
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
