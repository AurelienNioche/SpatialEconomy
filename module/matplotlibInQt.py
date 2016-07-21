from matplotlib.backends import qt_compat
from PyQt5.QtWidgets import QWidget, QMainWindow, QVBoxLayout, QSizePolicy, \
    QLabel
from PyQt5.QtCore import Qt, QTimer, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from multiprocessing import Event

from module.graphics import shared_zeros

use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE


class InformationGetter(QThread):

    def __init__(self, shared_object, queue, shutdown):

        QThread.__init__(self)
        self.shared_object = shared_object
        self.queue = queue
        self.shutdown = shutdown

    def run(self):

        while not self.shutdown.is_set():

            new_value = self.queue.get()

            if new_value is not None:
                self.shared_object[:] = new_value

        print("InformationGetter finished to run.")


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, data=None, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('white')

        self.axes = fig.add_subplot(111)
        self.axes.hold(False)  # We want the axes cleared every time plot() is called

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.data = data


class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):

        MplCanvas.__init__(self, *args, **kwargs)
        self.compute_initial_figure()

    def compute_initial_figure(self):

        self.axes.imshow(self.data)
        self.draw()

    def update_figure(self):

        self.axes.imshow(self.data.T)
        self.draw()


class PlotWindow(QMainWindow):

    def __init__(self, map_limits, fig_names, fig_queues):

        QMainWindow.__init__(self)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        # self.setWindowTitle("application main window")

        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("* { background-color: white }")

        grid = QVBoxLayout(self.main_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        
        self.dc = []
        self.information_getters = []

        self.fig_queues = fig_queues

        self.shutdown = Event()
        
        for i in range(len(fig_names)):

            shared_object = shared_zeros(map_limits["width"], map_limits["height"])

            self.information_getters.append(InformationGetter(queue=fig_queues[i],
                                                              shared_object=shared_object,
                                                              shutdown=self.shutdown))
            
            self.dc.append(DynamicMplCanvas(self.main_widget, data=shared_object, width=5, height=4,
                                            dpi=100))

            label = QLabel(fig_names[i])
            label.setStyleSheet("QLabel { background-color: white }")
            label.setAlignment(Qt.AlignCenter)

            grid.addWidget(label)
            grid.addWidget(self.dc[i])
            grid.addSpacing(20)

        for i in self.information_getters:
            i.start()

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        timer = QTimer(self)
        timer.timeout.connect(self.update_display)
        timer.setInterval(0)
        timer.start()
        self.show()

    def update_display(self):

        for i in self.dc:
            
            i.update_figure()

        # self.label.setText("Trial: {}".format(self.data["trial"]))

    def closeEvent(self, event):

        self.shutdown.set()
        for i in self.fig_queues:
            i.put(None)
        Event().wait(0.5)
        print("Window's dead.")
        self.close()
