#!/usr/bin/env python3
 
import math
import numpy as np
import matplotlib.pyplot as plt

# Given data
T0 = 90.0
T_ref = 60.0
t1 = 10.0
T_t1 = 88.0

# Infer k from the 10-minute measurement
k = (1.0 / t1) * math.log((T_t1 - T_ref) / (T0 - T_ref))

# Compute T(20)
t2 = 20.0
T_t2 = T_ref + (T0 - T_ref) * math.exp(k * t2)

# Time to reach 65 F
T_target = 65.0
t_target = math.log((T_target - T_ref) / (T0 - T_ref)) / k

# Create a smooth time array up to slightly beyond t_target
t_max = max(60.0, t_target * 1.1)  # show enough of the cooling curve
t = np.linspace(0, t_max, 400)
T = T_ref + (T0 - T_ref) * np.exp(k * t)

plt.figure(figsize=(7, 4.5))
plt.plot(t, T, label="Exact solution T(t)")
plt.scatter(
    [0, t1, t2, t_target], [T0, T_t1, T_t2, T_target], zorder=3, label="Key points"
)
plt.axhline(T_ref, linestyle="--", linewidth=1, label="Ambient (60°F)")
plt.axhline(T_target, linestyle=":", linewidth=1, label="Target (65°F)")
plt.axvline(t_target, linestyle=":", linewidth=1)
plt.title("Newton's Law of Cooling — Problem 16")
plt.xlabel("Time [minutes]")
plt.ylabel("Temperature [°F]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cooling_problem16.png", dpi=150)
plt.show()

print(f"T(20 min) = {T_t2:.1f} °F")
print(f"Time to reach 65 °F = {t_target:.1f} minutes (~{t_target/60:.1f} hours)")