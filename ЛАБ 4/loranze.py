
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def simulate_lorenz(initial_state, t_span, t_steps):

    t_eval = np.linspace(t_span[0], t_span[1], t_steps)
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    return sol.t, sol.y


def plot_lorenz(sol1, sol2=None):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(sol1[0], sol1[1], sol1[2], label="Траєкторія 1", color='blue')

    if sol2 is not None:
        ax.plot(sol2[0], sol2[1], sol2[2], label="Траєкторія 2", color='red', linestyle='--')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Атрактор Лоренца")
    ax.legend()
    plt.show()


def plot_deviation(t, sol1, sol2):

    diff = np.sqrt(np.sum((sol1 - sol2)**2, axis=0))  # Евклідова відстань
    plt.figure(figsize=(10, 5))
    plt.plot(t, diff)
    plt.xlabel('Час')
    plt.ylabel('Відстань між траєкторіями')
    plt.title('Розбіжність траєкторій через малу зміну початкових умов')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Початкові умови
    initial_state1 = [1.0, 1.0, 1.0]
    initial_state2 = [1.001, 1.0, 1.0]  # Дуже близька умова

    # Інтервал часу
    t_span = (0, 40)
    t_steps = 10000

    # Симуляції
    t1, sol1 = simulate_lorenz(initial_state1, t_span, t_steps)
    t2, sol2 = simulate_lorenz(initial_state2, t_span, t_steps)

    # Побудова графіка траєкторій
    plot_lorenz(sol1, sol2)

    # Побудова графіка відхилення між траєкторіями
    plot_deviation(t1, sol1, sol2)
