import numpy as np
from multiprocessing import Pool
# from writer_main_wisdom_of_woc import SimulationRunner, BackUp
import pickle
from datetime import datetime 


def simple_main():

    '''
    Simplest program use
    :return: None
    '''
   
    #------------------------------------#

    step = 5 
    mini = 50
    maxi = 70
    array = np.zeros(3)
    array[:] = mini 
    workforce_list = list()
    workforce_list.append(array.copy())


    while array[0] < 70:
        
        for i in range(len(array)):
           
            array[i] += step  
            workforce_list.append(array.copy())
           
    #------------------------------------#
     
    idx = 0
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    tau_list =[0.01, 0.02, 0.03, 0.04, 0.05]
    vision_list = [1, 3, 6, 12, 15]
    area_list = [1, 3, 6, 12, 15]
    stride = 1 
    t_max = 600
    width = 30
    height = 30
    parameters_list = list()
 
    
    #------------------------------------#
    
    for workforce in workforce_list:
        
        for alpha in alpha_list:
            
            for tau in tau_list:
                
                for vision in vision_list:
                    
                    for area in area_list:                        
                        
                        idx += 1
                        parameters = \
                            {
                                "workforce": np.array(workforce, dtype=int),
                                "alpha": alpha ,  # Set the coefficient learning
                                "tau": tau,  # Set the softmax parameter.
                                "t_max": t_max,  # Set the number of time units the simulation will run
                                               #by each agent a at each round 
                                "stride": stride, 
                                "vision": vision,  # Set the importance of other agents'results in
                                                # front of an indivual res
                                "area": area,  # set the perimeter within each agent can move 
                                "map_limits": {"width": width, "height": height},
                                "idx": idx 

                            }
                        
                    
                        parameters_list.append(parameters)
    
    
    #------------------------------------#
    
    parameters_dict = dict()    
    nb_sub_list = 10
    sub_part = int(len(parameters_list) / 10 )
        

    cursor = 0
    
    for i in range(nb_sub_list):
        
        part = parameters_list[cursor:cursor+sub_part]
        parameters_dict[i] = part 
        cursor += sub_part   

    
    while cursor < len(parameters_list):
        
        for i in range(nb_sub_list): 
            
            if cursor < len(parameters_list):
                parameters_dict[i].append(parameters_list[cursor])       
                cursor += 1
        
    
    date = str(datetime.now())[:-10].replace(" ", "_").replace(":", "-")
    
    for i in parameters_dict.keys():
        
        pickle.dump(parameters_dict[i], open( "../data/parameters_lists/slices_{d}_{i}.p".format(i=i, d=date), mode="wb"))
 
                        
if __name__ == "__main__":

    simple_main()


 
