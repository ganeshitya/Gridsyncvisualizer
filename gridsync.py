import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Constants
fs = 1000  # Sampling frequency
T = 1.0    # Duration in seconds
t = np.linspace(0, T, int(fs*T), endpoint=False)

# UI
st.title("🌀 Grid Synchronization Visualizer")
algorithm = st.selectbox("Choose Sync Algorithm", ["PLL", "SRF-PLL", "SOGI-PLL", "DSOGI-PLL"])
freq_grid = st.slider("Grid Frequency (Hz)", 49.5, 50.5, 50.0, step=0.01)
noise_level = st.slider("Grid Noise Level", 0.0, 0.5, 0.05)

# Generate noisy grid signal
v_grid = np.sin(2 * np.pi * freq_grid * t) + noise_level * np.random.randn(len(t))

# Utility functions
def lowpass_moving_avg(signal, window=20):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def phase_unwrap(a, b):
    return np.unwrap(np.angle(np.exp(1j * a) / np.exp(1j * b)))

def estimate_frequency(theta):
    dtheta = np.diff(theta)
    freq = fs * dtheta / (2 * np.pi)
    freq = np.append(freq, freq[-1])  # match original length
    return freq

# Algorithm implementations
def run_pll():
    theta = 2 * np.pi * 50 * t
    return np.sin(theta), theta

def run_srf_pll():
    alpha = v_grid
    beta = np.sin(2 * np.pi * freq_grid * t + np.pi/2)  # approx quadrature

    theta = np.zeros_like(t)
    Kp, Ki = 100, 2000
    integrator = 0

    for i in range(1, len(t)):
        v_q = -alpha[i] * np.sin(theta[i-1]) + beta[i] * np.cos(theta[i-1])
        integrator += Ki * v_q / fs
        d_theta = Kp * v_q + integrator
        theta[i] = theta[i-1] + d_theta / fs

    return np.sin(theta), theta

def run_sogi_pll():
    alpha = v_grid
    omega = 2 * np.pi * 50
    k = 1.0  # SOGI gain

    x1, x2 = 0, 0
    y_alpha = np.zeros_like(alpha)
    y_beta = np.zeros_like(alpha)

    for i in range(1, len(alpha)):
        e = alpha[i] - x1
        dx1 = omega * x2 + k * omega * e
        dx2 = -omega * x1
        x1 += dx1 / fs
        x2 += dx2 / fs
        y_alpha[i] = x1
        y_beta[i] = x2

    # Standard PLL loop
    theta = np.zeros_like(t)
    integrator = 0
    Kp, Ki = 100, 2000

    for i in range(1, len(t)):
        v_q = -y_alpha[i] * np.sin(theta[i-1]) + y_beta[i] * np.cos(theta[i-1])
        integrator += Ki * v_q / fs
        d_theta = Kp * v_q + integrator
        theta[i] = theta[i-1] + d_theta / fs

    return np.sin(theta), theta

def run_dsogi_pll():
    alpha = v_grid
    omega = 2 * np.pi * 50
    k = 1.0

    x1_p, x2_p = 0, 0
    x1_n, x2_n = 0, 0
    y_alpha = np.zeros_like(alpha)
    y_beta = np.zeros_like(alpha)

    for i in range(1, len(alpha)):
        # Positive sequence
        ep = alpha[i] - x1_p
        dx1_p = omega * x2_p + k * omega * ep
        dx2_p = -omega * x1_p
        x1_p += dx1_p / fs
        x2_p += dx2_p / fs

        # Negative sequence
        en = alpha[i] - x1_n
        dx1_n = -omega * x2_n + k * omega * en
        dx2_n = omega * x1_n
        x1_n += dx1_n / fs
        x2_n += dx2_n / fs

        y_alpha[i] = x1_p - x1_n
        y_beta[i] = x2_p - x2_n

    # PLL
    theta = np.zeros_like(t)
    integrator = 0
    Kp, Ki = 100, 2000

    for i in range(1, len(t)):
        v_q = -y_alpha[i] * np.sin(theta[i-1]) + y_beta[i] * np.cos(theta[i-1])
        integrator += Ki * v_q / fs
        d_theta = Kp * v_q + integrator
        theta[i] = theta[i-1] + d_theta / fs

    return np.sin(theta), theta

# Execute chosen algorithm
if algorithm == "PLL":
    output, theta = run_pll()
elif algorithm == "SRF-PLL":
    output, theta = run_srf_pll()
elif algorithm == "SOGI-PLL":
    output, theta = run_sogi_pll()
elif algorithm == "DSOGI-PLL":
    output, theta = run_dsogi_pll()

# Ground truth
theta_grid = 2 * np.pi * freq_grid * t

# Phase error and frequency tracking
phase_error = phase_unwrap(theta, theta_grid)
freq_actual = estimate_frequency(theta)
freq_true = np.full_like(freq_actual, freq_grid)
freq_error = freq_actual - freq_true

# Settling time calculation
settling_time_idx = np.argmax(np.abs(freq_error) < 0.1)
settling_time = t[settling_time_idx] if settling_time_idx > 0 else None

# 📊 Plot
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

ax[0].plot(t, v_grid, label="Grid Signal", color='orange')
ax[0].plot(t, output, label=f"{algorithm} Output", color='blue')
ax[0].legend()
ax[0].set_title("Signal Comparison")

ax[1].plot(t, phase_error, color='red')
ax[1].set_title("Phase Error (rad)")

ax[2].plot(t, freq_error, color='purple')
ax[2].axhline(0, linestyle='--', color='gray')
if settling_time:
    ax[2].axvline(settling_time, color='green', linestyle='--', label=f"Settling Time ≈ {settling_time:.3f}s")
    ax[2].legend()
ax[2].set_title("Frequency Error (Hz)")
ax[2].set_xlabel("Time (s)")

st.pyplot(fig)
