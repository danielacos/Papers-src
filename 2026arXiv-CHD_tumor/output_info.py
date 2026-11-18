import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

average_newton_iter = []
average_time = []

tau_plot_K_0_1 = []
tau_plot_K_1 = []
tau_plot_K_10 = []
tau_plot_K_0_1_symmetric = []
tau_plot_K_1_symmetric = []
tau_plot_K_10_symmetric = []

tests = ['reference_test', 'test_chi-0.01', 'test_chi-0.5', 'test_chi-1', 'test_P0-0.001', 'test_P0-0.05', 'test_P0-2']

for K in ["1e-1", "1", "10"]:
    for test in tests:
        file_path = f'tests/tests_irregular_shape_K-{K}/{test}'
        i = 0
        while i < 2:
            if i == 1: file_path += "_symmetric"

            data = pd.read_csv(file_path + "/output.txt", sep=' ', header=36, index_col=0, skipinitialspace=True)

            t = data.dropna(subset=['t'])['t'].to_numpy()

            newton_iter = data.dropna(subset=['n_iter'])['n_iter'].to_numpy()
            if i == 1:
                print(f"Test: {test} symmetric, K: {K}:")
            else:
                print(f"Test: {test}, K: {K}:")
            print(f"Max Newton iterations: {max(newton_iter)}")
            print(f"Min Newton iterations: {min(newton_iter)}")
            average_newton_iter.append(np.mean(newton_iter))
            print(f"Average Newton iterations: {average_newton_iter[-1]:.2f}\n")

            tau_u = -data.dropna(subset=['tau_u'])['tau_u'].to_numpy()
            tau_n = -data.dropna(subset=['tau_n'])['tau_n'].to_numpy()
            energy = data.dropna(subset=['energy'])['energy'].to_numpy()
            print(f"Range tau_u: ({min(tau_u):.2e}, {max(tau_u):.2e})")
            print(f"Range tau_n: ({min(tau_n):.2e}, {max(tau_n):.2e})")
            print(f"Range tau_u + tau_n: ({min(tau_u + tau_n):.2e}, {max(tau_u + tau_n):.2e})")
            print(f"Range energy: ({min(energy):.2e}, {max(energy):.2e})\n")

            with open(file_path + "/time.txt", "r") as f:
                last = f.read().splitlines()[-1]

            total_time = re.search(r"([\d:,]+) total", last).group(1)
            parts = [float(x.replace(",", ".")) for x in total_time.split(":")]
            total_time_seconds = sum(p * 60**i for i, p in enumerate(reversed(parts)))

            n_steps = len(newton_iter)
            print("Time steps:", n_steps)

            print(f"Total time: {total_time_seconds / 3600:.2f} h")
            average_time.append(total_time_seconds / n_steps)
            print(f"Average time per step: {average_time[-1]:.2f} s\n")

            i += 1

print(f"Maximum average Newton iterations for all tests: {max(average_newton_iter):.2f}")
print(f"Minimum average Newton iterations for all tests: {min(average_newton_iter):.2f}")
print(f"Average Newton iterations for all tests: {np.mean(average_newton_iter):.2f}")

print(f"Maximum average time per step for all tests: {max(average_time):.2f} s")
print(f"Minimum average time per step for all tests: {min(average_time):.2f} s")
print(f"Average time per step for all tests: {np.mean(average_time):.2f} s")