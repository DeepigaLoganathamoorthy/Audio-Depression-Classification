import numpy as np
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
time = np.linspace(0, duration, num=n_samples)
plt.figure(figsize=(15, 5))
plt.plot(time, signal_array)
plt.title('<title>')
plt.ylabel(' signal wave')
plt.xlabel('time (s)')
plt.xlim(0, duration)
plt.show()
