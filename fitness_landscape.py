import numpy as np
import matplotlib.pyplot as plt

n = 3
x, y = np.mgrid[-n:n:200j, -n:n:200j]
z = x * np.exp(-x ** 2 - y ** 2)
max_z = 0
max_z_index = (0, 0)
min_z = 0
min_z_index = (0, 0)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        if z[i][j] > max_z:
            max_z = z[i][j]
            max_z_index = (i, j)
        if z[i][j] < min_z:
            min_z = z[i][j]
            min_z_index = (i, j)
print(max_z, max_z_index)
print(min_z, min_z_index)
print(x[max_z_index], y[max_z_index])
print(x[min_z_index], y[min_z_index])

ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.5)
# ax.quiver([0], [0], [0], [0.71], [0], [0.43])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        if np.abs(z[i][j]) > 0.4:
            if np.random.rand() > 0.1:
                if z[i][j] > 0:
                    ax.scatter(x[i][j], y[i][j], z[i][j],
                               color='red', marker='+', s=100)
                else:
                    ax.scatter(x[i][j], y[i][j], z[i][j],
                               color='blue', marker='+', s=100)

ax.view_init(elev=10, azim=-90)
plt.tight_layout()
plt.savefig('fitness_landscape.png')
