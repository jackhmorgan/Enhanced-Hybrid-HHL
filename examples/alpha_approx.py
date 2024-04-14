import matplotlib.pyplot as plt
import numpy as np

T=8


def alpha(delta, T):

    coefficient = np.sqrt(2)*np.sin((np.pi)/(2*T))/T
    numerator = abs(np.cos(delta/(2*T))*np.cos(delta/2))
    denominator = abs(np.sin((delta+np.pi)/(2*T))*np.sin((delta-np.pi)/(2*T)))
    return coefficient*numerator/denominator

deltas = np.linspace(0, 2*np.pi, 100).tolist()
alphas = [alpha(delta,T) for delta in deltas]
approxes = [(np.pi*2-delta)/(np.pi*2) for delta in deltas]
mse = 0
for alpha, approx in zip(alphas, approxes):
    mse += alpha-approx
mse /= len(alphas)
print(mse)
plt.plot(deltas, alphas, label = r'$|\alpha_{k|l}|$')
plt.plot(deltas, approxes, label = r'$\tilde{\alpha_{k|l}}$')
plt.xlabel(r'$\delta$')  # LaTeX label for x-axis
plt.ylabel(r'$|\alpha_{k|l}|$')  # LaTeX label for y-axis
plt.title(r'$|\alpha_{k|l}|$ vs. $\tilde{\alpha}_{k|l}$')

# Set x-axis ticks to multiples of pi
plt.xticks(np.arange(0, 2*np.pi+0.01, np.pi/2), [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.show()