from multiprocessing import Event, Queue
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np
from module.graphics import EcoWindow
from module.matplotlibInQt import PlotWindow
import pickle, time
from module.converter import read


class GraphicManager(QThread):

    def __init__(self, map_queue, exchange_queues, map_limits):

        QThread.__init__(self)
        self.map_queue = map_queue
        self.exchange_queues = exchange_queues
        self.map_limits = map_limits
        
        self.speed = 0.01  # In seconds

    def run(self):

        date = open("../data/last.txt", mode='r').read()
        self.map = pickle.load(open("../data/map{}.p".format(date), mode='rb'))
        matrix_temp = list() 
        
        self.matrix_list = { "0": list(), "1": list(), "2": list() }
        
        for i in range(len(self.matrix_list)):
            
            self.matrix_list[str(i)] = \
                read(self.map_limits["height"], table_name="exchange_{i}".format(i=i),
                     database_name="array_exchanges{}".format(date))
        
        t_max = len(self.map) 
      
        for i in range(t_max):
            
                        
            Event().wait(self.speed)
            
            self.exchange_matrix = np.zeros((self.map_limits["width"], self.map_limits["height"]),
                                       dtype=[("0", float, 1),
                                              ("1", float, 1),
                                              ("2", float, 1)])
            
                        
            for j in [0, 1, 2]:
                
              
                self.exchange_matrix[str(j)][:] = self.matrix_list[str(j)][i][:]
                 
                
             
            
            for j in [0,1,2]:
                
                self.exchange_queues[j].put(self.exchange_matrix[str(j)])

            
            agent_dic = self.map[i]
           
            self.map_queue.put(agent_dic)


def main():

    map_limits = {"width": 25, "height": 25}

    shutdown = Event()
    map_queue = Queue()

    exchanges = [0, 1, 2]
    exchange_queues = []
    for i in exchanges:
        exchange_queues.append(Queue())

    app = QApplication(sys.argv)

    w = EcoWindow(
        queue=map_queue,
        map_limits=map_limits,
        shutdown=shutdown)

    w2 = PlotWindow(map_limits=map_limits,
                    fig_queues=exchange_queues, fig_names=['Exchange_{}'.format(i) for i in exchanges])

    graphic = GraphicManager(map_queue=map_queue, exchange_queues=exchange_queues, map_limits=map_limits)
    graphic.start()

    sys.exit(app.exec_())

if __name__ == "__main__":

    main()




