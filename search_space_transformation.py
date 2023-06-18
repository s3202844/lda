import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

fig, ax = plt.subplots(figsize=(8, 8))
rect = patches.Rectangle((-100., -100.), 200, 200,
                         linewidth=3, edgecolor="blue", facecolor="none")
ax.add_patch(rect)
rect = patches.Rectangle((-130., -130.), 200, 200, linewidth=3,
                         edgecolor="red", facecolor="none", linestyle="--")
ax.add_patch(rect)
ax.plot([0], [0], color="blue", linewidth=3, label="original search space")
ax.plot([0], [0], color="red", linewidth=3, linestyle="--",
        label="search space after translation")
ax.axis('off')
ax.set_xlim(-150., 150.)
ax.set_ylim(-150., 150.)
ax.axhline(0., color='black', linewidth=0.5)
ax.axvline(0., color='black', linewidth=0.5)
plt.legend(loc="upper right", fontsize=20)
fig.savefig("transformation/translation_search.png")


fig, ax = plt.subplots(figsize=(8, 8))
rect = patches.Rectangle((-100., -100.), 200, 200,
                         linewidth=3, edgecolor="blue", facecolor="none")
ax.add_patch(rect)
rect = patches.Rectangle((-100., -100.), 200, 200, linewidth=3,
                         edgecolor="red", facecolor="none", linestyle="--")
angle = -30.
center = (0, 0)
trans = transforms.Affine2D().rotate_deg(angle)
rect.set_transform(trans + ax.transData)
ax.add_patch(rect)
ax.plot([0], [0], color="blue", linewidth=3, label="original search space")
ax.plot([0], [0], color="red", linewidth=3, linestyle="--",
        label="search space after rotation")
ax.axis('off')
ax.set_xlim(-150., 150.)
ax.set_ylim(-150., 150.)
ax.axhline(0., color='black', linewidth=0.5)
ax.axvline(0., color='black', linewidth=0.5)
plt.legend(loc="upper right", fontsize=20)
fig.savefig("transformation/rotation.png")


fig, ax = plt.subplots(figsize=(8, 8))
rect = patches.Rectangle((-100., -100.), 200, 200,
                         linewidth=3, edgecolor="blue", facecolor="none")
ax.add_patch(rect)
rect = patches.Rectangle((-120., -120.), 240, 240, linewidth=3,
                         edgecolor="red", facecolor="none", linestyle="--")
ax.add_patch(rect)
rect = patches.Rectangle((-70., -70.), 140, 140, linewidth=3,
                         edgecolor="red", facecolor="none", linestyle="--")
ax.add_patch(rect)
ax.plot([0], [0], color="blue", linewidth=3, label="original search space")
ax.plot([0], [0], color="red", linewidth=3, linestyle="--",
        label="search space after scaling")
ax.axis('off')
ax.set_xlim(-150., 150.)
ax.set_ylim(-150., 150.)
ax.axhline(0., color='black', linewidth=0.5)
ax.axvline(0., color='black', linewidth=0.5)
plt.legend(loc="upper right", fontsize=20)
fig.savefig("transformation/scaling_search.png")
