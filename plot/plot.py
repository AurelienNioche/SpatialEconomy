from enum import Enum, auto
from pylab import plt, np
import matplotlib.colors
import matplotlib.animation
import matplotlib.pyplot


class Mode(Enum):

    SAVE = auto()
    DISPLAY = auto()
    ON_KEY_PRESS = auto()


class Plot:

    matplotlib.pyplot.rcParams['toolbar'] = 'None'

    color_list = ['white', 'blue', 'red', 'purple']
    color_map = matplotlib.colors.ListedColormap(color_list)

    video_name = "SpatialEconomy.mp4"

    def __init__(self, data=np.random.randint(len(color_list), size=(10, 4, 4)), mode=Mode.DISPLAY):

        self.data = data
        self.mode = mode

        # New figure with white background
        self.fig = matplotlib.pyplot.figure(figsize=(10, 10), facecolor='white', dpi=72)

        # New self.axis over the whole figure and a 1:1 aspect ratio
        self.fig.subplots_adjust(top=.96, bottom=.02, left=.02, right=.98)
        self.ax = self.fig.add_subplot(111)

        self.im = self.ax.imshow(
            self.data[0], interpolation='none', aspect='auto', origin='upper',
            cmap=self.color_map, vmin=0, vmax=len(self.color_list))

        self.t = 0

        self.animation = None

        self.run()

    def run(self):

        plt.xticks([]), plt.yticks([])

        if self.mode == Mode.DISPLAY:
            self.animation = matplotlib.animation.FuncAnimation(self.fig, self.time_step, interval=60)

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

        self.im.set_array(self.data[self.t])

        if not self.t+1 >= len(self.data):
            self.t += 1

        if self.mode == Mode.ON_KEY_PRESS:
            self.fig.canvas.draw()


def main():

    Plot(mode=Mode.ON_KEY_PRESS)


if __name__ == "__main__":

    main()
