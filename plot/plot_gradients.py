from enum import Enum, auto
from pylab import plt, np
import matplotlib.colors
import matplotlib.animation
import matplotlib.pyplot
import matplotlib.cm


fig = plt.figure()

ax = fig.add_subplot(221)
ax.set_xticks([])
ax.set_yticks([])

x = np.random.random(size=(10, 10))
ax.imshow(x, cmap=matplotlib.cm.get_cmap("Blues"))

ax = fig.add_subplot(222)
x = np.random.random(size=(10, 10))
ax.imshow(x, cmap=matplotlib.cm.get_cmap("Oranges"))

ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(223)
x = np.random.random(size=(10, 10))
ax.imshow(x, cmap=matplotlib.cm.get_cmap("Greens"))

ax.set_xticks([])
ax.set_yticks([])

plt.show()
