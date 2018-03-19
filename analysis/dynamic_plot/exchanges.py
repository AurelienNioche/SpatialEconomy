from enum import Enum, auto
import numpy as np
import matplotlib.colors
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib.cm


class Mode(Enum):

    SAVE = auto()
    DISPLAY = auto()
    ON_KEY_PRESS = auto()


class Plot:

    matplotlib.pyplot.rcParams['toolbar'] = 'None'

    video_name = "SpatialEconomyExchanges.mp4"

    def __init__(self, data=None, mode=Mode.DISPLAY):

        if data is None:
            print("Demo data are used")
            self.data = np.random.random(size=(100, 3, 10, 10))
        else:
            self.data = data  # Shape is t_max, n_types, width, height

        self.mode = mode

        self.t = 0

        self.fig = plt.figure()

        cmaps = 'hot', 'hot', 'hot'  # "Blues", "Oranges", "Greens"

        self.im = []

        for (i, cmap) in enumerate(cmaps):

            ax = self.fig.add_subplot(2, 2, i+1)
            ax.set_xticks([])
            ax.set_yticks([])

            self.im.append(
                ax.imshow(data[0, i, :, :], cmap=matplotlib.cm.get_cmap(cmap),
                          vmin=0, vmax=np.max(data), aspect=1,
                          interpolation='none', origin='upper')
            )

        plt.tight_layout()

        self.run()

    def run(self):

        plt.xticks([]), plt.yticks([])

        if self.mode == Mode.DISPLAY:
            animation = matplotlib.animation.FuncAnimation(self.fig, self.time_step, interval=60)
            # 'animation =' is necessary

        elif self.mode == Mode.ON_KEY_PRESS:
            self.fig.canvas.mpl_connect('key_press_event', self.time_step)

        else:
            print("Creating video! Could need some time to complete!")

            ffmpeg_writer = matplotlib.animation.writers['ffmpeg']
            metadata = dict(title=self.video_name.split(".")[0], artist='Matplotlib',
                            comment='')
            writer = ffmpeg_writer(fps=15, metadata=metadata)

            n_frames = len(self.data)

            with writer.saving(self.fig, self.video_name, n_frames):
                for i in range(n_frames):
                    self.time_step()
                    writer.grab_frame()

        if self.mode != Mode.SAVE:
            plt.show()

    def time_step(self, *args):

        if self.t + 1 >= self.data.shape[0]:
            return

        self.t += 1
        print("t = {}".format(self.t), end="\r")

        for i in range(3):

            self.im[i].set_array(self.data[self.t, i])

            if self.mode == Mode.ON_KEY_PRESS:
                self.fig.canvas.draw()


def plot(data=None, mode=Mode.DISPLAY):

    Plot(data=data, mode=mode)


if __name__ == "__main__":

    plot()

