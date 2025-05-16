import unittest
import numpy as np
from lorenz import lorenz, simulate_lorenz

class TestLorenzAttractor(unittest.TestCase):
    def test_lorenz_function(self):
        """Тестування функції lorenz для коректності обчислень похідних."""
        state = [1.0, 1.0, 1.0]
        t = 0
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        expected = [
            sigma * (state[1] - state[0]),  # dx/dt = sigma * (y - x)
            state[0] * (rho - state[2]) - state[1],  # dy/dt = x * (rho - z) - y
            state[0] * state[1] - beta * state[2]  # dz/dt = x * y - beta * z
        ]
        result = lorenz(t, state, sigma, rho, beta)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_lorenz_function_zero_state(self):
        """Тестування функції lorenz при нульовому стані."""
        state = [0.0, 0.0, 0.0]
        t = 0
        result = lorenz(t, state)
        expected = [0.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_simulate_lorenz_output_shape(self):
        """Тестування форми вихідних даних simulate_lorenz."""
        initial_state = [1.0, 1.0, 1.0]
        t_span = (0, 10)
        t_steps = 1000
        t, sol = simulate_lorenz(initial_state, t_span, t_steps)
        self.assertEqual(t.shape, (t_steps,))
        self.assertEqual(sol.shape, (3, t_steps))

    def test_simulate_lorenz_time_steps(self):
        """Тестування коректності часового вектора в simulate_lorenz."""
        initial_state = [1.0, 1.0, 1.0]
        t_span = (0, 10)
        t_steps = 1000
        t, _ = simulate_lorenz(initial_state, t_span, t_steps)
        expected_t = np.linspace(t_span[0], t_span[1], t_steps)
        np.testing.assert_array_almost_equal(t, expected_t, decimal=6)

    def test_simulate_lorenz_different_initial_conditions(self):
        """Тестування чутливості до початкових умов."""
        initial_state1 = [1.0, 1.0, 1.0]
        initial_state2 = [1.001, 1.0, 1.0]
        t_span = (0, 10)
        t_steps = 1000
        _, sol1 = simulate_lorenz(initial_state1, t_span, t_steps)
        _, sol2 = simulate_lorenz(initial_state2, t_span, t_steps)
        # Перевірка, що рішення різні
        self.assertFalse(np.array_equal(sol1, sol2))

if __name__ == '__main__':
    unittest.main()
