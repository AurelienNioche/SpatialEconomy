from multiprocessing import Event, Array, Manager
import numpy as np
import ctypes
from PyQt5.QtWidgets import QWidget, QMainWindow, QGridLayout
from PyQt5.QtGui import QPalette, QColor, QPainter, QBrush, QPen
from PyQt5.QtCore import QRect, Qt, QTimer, QEvent, QThread
from collections import OrderedDict


def shared_zeros(n1, n2):
    # create a  2D numpy array which can be then changed in different threads
    shared_array_base = Array(ctypes.c_double, n1 * n2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n1, n2)
    return shared_array


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

                self.shared_object.clear()
                self.shared_object.update(new_value)

        print("Economic InformationGetter finished to run.")


class Frame(QWidget):

    def __init__(self):

        QWidget.__init__(self)

        self.line_width = 0

        self.color = {"white": QColor(255, 255, 255),
                      "black": QColor(0, 0, 0),
                      "blue": QColor(114, 212, 247),
                      "green": QColor(18, 247, 41),
                      "grey": QColor(220, 220, 220)}

        self.painter = QPainter()
        self.pen = {}
        self.brush = {}

        self.create_background()
        self.prepare_pens_and_brushes()

    def create_background(self):

        pal = QPalette()
        pal.setColor(QPalette.Background, QColor(255, 255, 255))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def prepare_pens_and_brushes(self):

        for color in self.color:

            self.pen[color] = QPen()
            self.pen[color].setColor(self.color[color])

            self.brush[color] = QBrush()
            self.brush[color].setStyle(Qt.SolidPattern)
            self.brush[color].setColor(self.color[color])

        self.pen["transparent"] = QPen()
        self.pen["transparent"].setStyle(Qt.NoPen)

        self.brush["transparent"] = QBrush()
        self.brush["transparent"].setStyle(Qt.NoBrush)

        self.brush["texture"] = QBrush()

    def paintEvent(self, e):

        self.painter.begin(self)
        self.draw()
        self.painter.end()

    def adapt_line_width(self, window_size):

        self.line_width = int(1/60. * window_size["height"])
        for c in self.color:
            self.pen[c].setWidth(self.line_width)

    def draw(self):

        pass

    def adapt_size_to_window(self, window_size):

        pass


class Window(QMainWindow):

    def __init__(self, shutdown=Event()):

        super(Window, self).__init__()

        self.shutdown = shutdown

        self.setGeometry(600, 600, 600, 600)

        self.central_area = QWidget()
        self.setCentralWidget(self.central_area)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(0)
        self.central_area.setLayout(self.grid)

        self.frames = OrderedDict()

    def initialize(self):

        for i in self.frames.values():
            i.adapt_size_to_window({"width": self.width(), "height": self.height()})
            self.grid.addWidget(i, 0, 0)
            i.hide()

        self.show()

    def update_size(self):

        for i in self.frames.values():
            i.adapt_size_to_window({"width": self.width(), "height": self.height()})
        self.hide()
        self.show()

    def changeEvent(self, event):

        if event.type() == QEvent.WindowStateChange:

            QTimer().singleShot(0, self.update_size)

    def resizeEvent(self, event):

        QTimer().singleShot(0, self.update_size)


class EconomyFrame(Frame):

    def __init__(self, map_limits, queue, shutdown):

        Frame.__init__(self)

        self.circle = dict()

        self.map_limits = map_limits

        self.map_type_of_agent = Manager().dict()

        self.agent_type_color_map = \
            {0: "black",
             1: "blue",
             2: "green"}
        self.information_getter = InformationGetter(self.map_type_of_agent, queue, shutdown)
        self.information_getter.start()

        self.timer = QTimer()
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.repaint)
        self.timer.start()

    def draw(self):

        map_type_of_agent = self.map_type_of_agent.copy()
        
        for position, to_print in map_type_of_agent.items():
                
            rectangle = QRect(position[0]*self.circle["width"],
                              position[1]*self.circle["height"],
                              self.circle["width"],
                              self.circle["height"])

            agent_type = to_print[0]
            agent_object = to_print[1]

            set_angle = 0
            size = 180*16

            self.painter.setBrush(self.brush[self.agent_type_color_map[agent_type]])
            self.painter.drawPie(rectangle, set_angle, size)

            set_angle = 180 * 16
            size = 180 * 16
            self.painter.setBrush(self.brush[self.agent_type_color_map[agent_object]])
            self.painter.drawPie(rectangle, set_angle, size)

            # self.painter.drawEllipse(rectangle)

    def adapt_size_to_window(self, window_size):

        for i in ["width", "height"]:
            self.circle[i] = window_size[i] / float(self.map_limits[i])
            
        self.repaint()


class EcoWindow(Window):

    def __init__(self, map_limits, shutdown, queue):

        Window.__init__(self, shutdown)

        self.shutdown = shutdown
        self.queue = queue

        self.frames["economy"] = EconomyFrame(map_limits=map_limits, queue=self.queue, shutdown=self.shutdown)

        self.initialize()
        self.frames["economy"].show()

    def closeEvent(self, QCloseEvent):

        self.shutdown.set()
        self.queue.put(None)
        Event().wait(1)
        print("Economic window's dead")
        self.close()
