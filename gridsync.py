import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

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

# Butterworth filter for SRF-PLL
def lowpass_filter(data, cutoff=20, fs=1000, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return lfilter(b, a, data)

# PLL: Basic sinusoidal tracking
if algorithm == "PLL":
    pll_output = np.sin(2 * np.pi * 50 * t)

# SRF-PLL: Using v_alpha and Hilbert transform for v_beta
elif algorithm == "SRF-PLL":
    # α-β transformation: v_beta = Hilbert(v_alpha)
    from scipy.signal import hilbert
    v_alpha = grid_signal
    v_beta = np.imag(hilbert(v_alpha))
    
    theta = np.zeros(len(t))
    freq_est = 2 * np.pi * 50
    Kp, Ki = 200, 10000  # SRF-PLL gains
    
    integrator = 0
    for i in range(1, len(t)):
        # Park Transformation (q-axis extraction)
        v_q = -v_alpha[i] * np.sin(theta[i-1]) + v_beta[i] * np.cos(theta[i-1])
        integrator += Ki * v_q / fs
        d_theta = Kp * v_q + integrator
        theta[i] = theta[i-1] + d_theta / fs
    
    pll_output = np.sin(theta)

# Phase error
def phase_diff(a, b):
    return np.unwrap(np.angle(np.exp(1j*a) / np.exp(1j*b)))

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
