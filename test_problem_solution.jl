import LinearAlgebra
using Plots
using DelimitedFiles


struct Solver
    m::Number
    left::Real
    right::Real
    coeff::Array{Real}
    initial_conditions::Array{Real}
end

# Аналитическое решение
function exact_solution(t)
    return -exp(-3 * t) / 12 * t * (-36 - 54 * t + 16 * t ^ 2 + 129 * t ^ 3)
end

# Скалярное произведение
function dot(x, y)
    return sum(x .* y)
end

# Разница векторов
function diff(x, y)
    return x .- y
end

# Сумма вектор
function sums(x, y)
    return x .+ y
end

# Произведение вектора на скаляр
function product(vec, scal)
    return vec .* scal
end

# Вытащить i - столбец из матрицы
function column(matrix, i)
    return [row[i] for row in matrix]
end

function euler_method(block::Solver)
    n = length(block.initial_conditions)
    h = (block.right - block.left) / block.m
    A = zeros(n, block.m)

    # Реализация начальных условий
    for i in 1:n
        A[i, 1] = block.initial_conditions[i]
    end

    for i in 2:block.m
        for j in 1:n  
            if j != n 
                A[j, i] = A[j, i - 1] + h * A[j + 1, i - 1]
            else
                A[j, i] = A[j, i - 1] - h * coeff[2:end]'reverse(A[:, i - 1]) / coeff[1]
            end
        end
    end 

    return A[1, :]
end

function runge_kutta(block::Solver)
    n = length(block.initial_conditions)
    h = (block.right - block.left) / block.m
    A = zeros(n, block.m)
    B = zeros(block.m, n, 4)

    # Реализация начальных условий
    for i in 1:n
        A[i, 1] = block.initial_conditions[i]
    end

    for k in 1:(block.m - 1)
        for i in 1:4
            for j in 1:n
                if i == 1
                    if j != n
                        B[k, j, i] = h * A[j + 1, k]
                    else
                        B[k, j, i] = -h * coeff[2:end]'reverse(A[:, k]) / coeff[1]
                    end
                end
                if i == 2 || i == 3
                    if j != n
                        B[k, j, i] = h * (A[j + 1, k] + B[k, j + 1, i - 1] / 2)
                    else
                        B[k, j, i] =  -h * coeff[2:end]'reverse(sums(A[:, k], product(B[k, :, i - 1], 0.5))) / coeff[1]
                    end
                end
                if i == 4
                    if j != n
                        B[k, j, i] = h * (A[j + 1, k] + B[k, j + 1, i - 1])
                    else
                        B[k, j, i] =  -h * coeff[2:end]'reverse(sums(A[:, k], B[k, :, i - 1])) / coeff[1]
                    end
                end
            end
        end
        # Обновление на новом слое
        for j in 1:n
            A[j, k + 1] = A[j, k] + (B[k, j, 1] + 2 * B[k, j, 2] + 2 * B[k, j, 3] + B[k, j, 4]) / 6
        end
    end
       
    return A[1, :]
end


M = 200
a, b = 0, 5
x = range(a, stop=b, length=M)
coeff = [1, 15, 90, 270, 405, 243]
initial_conditions = [0, 3, -9, -8, 0]
example = Solver(M, a, b, coeff, initial_conditions)
plot(x, [runge_kutta(example) euler_method(example)], label=["Runge-Kutta" "Euler"], title="N=200",linewidth=1)
plot!(x, exact_solution.(x), label="Extract")
#writedlm("example_euler.txt", euler_method(example))
#writedlm("example_runge_kutta.txt", runge_kutta(example))