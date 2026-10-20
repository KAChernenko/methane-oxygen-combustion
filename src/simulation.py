"""Двухстадийная модель горения метано-кислородной смеси."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


R = 8.314
CP = 1005.0

FORMATION_ENTHALPY = {
    "CH4": -74.8e3,
    "O2": 0.0,
    "CO": -110.5e3,
    "CO2": -393.5e3,
    "H2O": -241.8e3,
}

SPECIES = ("CH4", "O2", "CO", "CO2", "H2O")


def reaction_rates(temperature: float, concentrations: np.ndarray) -> tuple[float, float]:
    """Вычисляет скорости двух глобальных химических реакций."""
    ch4, o2, co, _, h2o = np.maximum(concentrations, 0.0)

    a1 = 1.59e13
    ea1 = 4.78e4 * 4.184
    a2 = 3.98e12
    ea2 = 4.07e4 * 4.184

    w1 = a1 * np.exp(-ea1 / (R * temperature)) * ch4**0.7 * o2**0.8
    w2 = a2 * np.exp(-ea2 / (R * temperature)) * co * o2**0.25 * h2o**0.5
    return float(w1), float(w2)


def rhs(_: float, state: np.ndarray) -> np.ndarray:
    """Возвращает правую часть системы уравнений кинетики и теплового баланса."""
    temperature = max(float(state[0]), 1.0)
    w1, w2 = reaction_rates(temperature, state[1:])

    d_ch4 = -w1
    d_o2 = -1.5 * w1 - 0.5 * w2
    d_co = w1 - w2
    d_co2 = w2
    d_h2o = 2.0 * w1

    delta_h1 = (
        FORMATION_ENTHALPY["CO"]
        + 2.0 * FORMATION_ENTHALPY["H2O"]
        - FORMATION_ENTHALPY["CH4"]
        - 1.5 * FORMATION_ENTHALPY["O2"]
    )
    delta_h2 = (
        FORMATION_ENTHALPY["CO2"]
        - FORMATION_ENTHALPY["CO"]
        - 0.5 * FORMATION_ENTHALPY["O2"]
    )
    d_temperature = -(delta_h1 * w1 + delta_h2 * w2) / CP

    return np.array([d_temperature, d_ch4, d_o2, d_co, d_co2, d_h2o])


def rk4_step(time: float, step: float, state: np.ndarray) -> tuple[float, np.ndarray]:
    """Выполняет один шаг метода Рунге–Кутты четвёртого порядка."""
    k1 = rhs(time, state)
    k2 = rhs(time + step / 2.0, state + step * k1 / 2.0)
    k3 = rhs(time + step / 2.0, state + step * k2 / 2.0)
    k4 = rhs(time + step, state + step * k3)
    next_state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return time + step, next_state


def gear_solver(
    start: float,
    end: float,
    step: float,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Интегрирует систему схемой прототипа: разгон РК-4 и основной шаг Гира."""
    dimension = len(initial_state)
    time = start
    state = np.array(initial_state, dtype=float)
    corrected = np.zeros(dimension)
    history = np.zeros((3, dimension))
    four_steps = 4.0 * step

    times = [time]
    states = [state.copy()]

    for _ in range(3):
        time, state = rk4_step(time, step, state)
        times.append(time)
        states.append(state.copy())

    while time <= end:
        for index in range(3):
            history[index] = state
            time, state = rk4_step(time, step, state)

        derivative = rhs(time, state)
        previous = state.copy()

        for index in range(dimension):
            state[index] = (
                (history[0, index] - 10.0 * state[index]) / 3.0
                - 2.0 * history[1, index]
                + 6.0 * history[2, index]
                + four_steps * derivative[index]
            )
            derivative = rhs(time, state)

        for index in range(dimension):
            corrected[index] = (
                48.0 * previous[index]
                - 36.0 * history[2, index]
                + 16.0 * history[1, index]
                - 3.0 * (history[0, index] - four_steps * derivative[index])
            ) / 25.0
            history[0, index] = history[1, index]
            history[1, index] = history[2, index]
            history[2, index] = corrected[index]

        time += step
        times.append(time)
        states.append(state.copy())

    return np.array(times), np.array(states)


def save_results(times: np.ndarray, states: np.ndarray, output_dir: Path) -> None:
    """Сохраняет таблицу результатов и два графика."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "simulation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("time_s", "temperature_k", *SPECIES))
        writer.writerows(np.column_stack((times, states)))

    plt.figure(figsize=(10, 5))
    plt.plot(times, states[:, 0], color="red", label="Температура")
    plt.xlabel("Время, с")
    plt.ylabel("Температура, K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for index, species in enumerate(SPECIES, start=1):
        plt.plot(times, states[:, index], label=species)
    plt.xlabel("Время, с")
    plt.ylabel("Концентрация, моль")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "concentrations.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=780.0, help="начальная температура, K")
    parser.add_argument("--end-time", type=float, default=0.02, help="конечное время расчёта, с")
    parser.add_argument("--steps", type=int, default=2000, help="число шагов на заданном интервале")
    parser.add_argument("--output", type=Path, default=Path("results"), help="папка для результатов")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("Начальная температура должна быть положительной")
    if args.end_time <= 0 or args.steps <= 0:
        raise ValueError("Время расчёта и число шагов должны быть положительными")

    initial_state = np.array([args.temperature, 1.0, 2.0, 0.0, 0.0, 0.0])
    step = args.end_time / args.steps
    times, states = gear_solver(0.0, args.end_time, step, initial_state)
    save_results(times, states, args.output)

    print(f"Расчёт завершён: {len(times)} сохранённых состояний")
    print(f"Начальная температура: {args.temperature:.2f} K")
    print(f"Конечная температура: {states[-1, 0]:.2f} K")
    print(f"Результаты: {args.output.resolve()}")


if __name__ == "__main__":
    main()
