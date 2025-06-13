import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Constants
fs = 1000  # sampling frequency
T = 1.0    # signal duration
t = np.linspace(0, T, int(fs*T), endpoint=False)

# UI
st.title("Grid Synchronization Simulator")
algorithm = st.selectbox("Choose Sync Algorithm", ["PLL", "SRF-PLL"])
freq_grid = st.slider("Grid Frequency (Hz)", 49.5, 50.5, 50.0)
noise_level = st.slider("Noise Level", 0.0, 0.5, 0.05)

# Generate noisy grid voltage
grid_signal = np.sin(2 * np.pi * freq_grid * t) + noise_level * np.random.randn(len(t))

# Helper: Moving average low-pass filter
def lowpass_moving_avg(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='same')

# PLL and SRF-PLL logic
if algorithm == "PLL":
    pll_output = np.sin(2 * np.pi * 50 * t)

elif algorithm == "SRF-PLL":
    # Approximate beta (90° shifted version of alpha)
    v_alpha = grid_signal
    v_beta = np.sin(2 * np.pi * freq_grid * t + np.pi / 2)

    theta = np.zeros_like(t)
    freq = 2 * np.pi * 50  # target frequency in rad/s
    Kp, Ki = 100, 2000     # control loop gains
    integrator = 0

    for i in range(1, len(t)):
        v_q = -v_alpha[i] * np.sin(theta[i-1]) + v_beta[i] * np.cos(theta[i-1])
        integrator += Ki * v_q / fs
        d_theta = Kp * v_q + integrator
        theta[i] = theta[i-1] + d_theta / fs

    pll_output = np.sin(theta)

# Phase error estimation
def phase_diff(phase1, phase2):
    return np.unwrap(phase1 - phase2)

true_phase = 2 * np.pi * freq_grid * t
measured_phase = 2 * np.pi * 50 * t if algorithm == "PLL" else theta
phase_error = phase_diff(measured_phase, true_phase)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t, grid_signal, label='Grid Signal', color='orange')
ax[0].plot(t, pll_output, label=f'{algorithm} Output', color='blue')
ax[0].legend()
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Signal Comparison")

ax[1].plot(t, phase_error, color='red')
ax[1].set_ylabel("Phase Error (rad)")
ax[1].set_xlabel("Time (s)")
ax[1].set_title("Phase Error vs Time")

st.pyplot(fig)
