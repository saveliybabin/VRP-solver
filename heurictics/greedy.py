import random
import pandas as pd
import numpy as np

def greedy_heurictics(df, dist_matrix, dim = 100, cap = 300, k_nei = 5, iters = 200):
    path_dict_best = {}
    dist_dict_best = {}
    total_dist_best = 100000
    
    for i in range(iters): #number of iterations
        dist_matrix_iter = dist_matrix.copy() # dist matrix
        depot = 0 # depot index
        path_dict = {} 
        dist_dict = {}
        global_check = True
        j = 0
        points = []
        points.append(0)
        total_dist = 0
        car_cap = cap

        while global_check:
            # first point is depot
            check = True
            curr_point = depot
            curr_cap = car_cap
            path = []
            path.append(curr_point)
            dist = 0

            #second point is random points
            if len(points) != dim:
                r = random.choice([i for i in range(dim) if i not in points])
                sorted_nei = list(dist_matrix_iter[r].sort_values().index)
                for pnt in points:
                    sorted_nei.remove(pnt)
                points.append(r)
                path.append(r)
                total_dist += dist_matrix_iter.iloc[curr_point, r]
                dist += dist_matrix_iter.iloc[curr_point, r]
                dist_matrix_iter.iloc[curr_point, r] = np.inf
                curr_point = r
                curr_cap -= df.loc[r, 'demand']
            else:
                check = False
                global_check = False

            #further nearest point
            while check:
                    p = 0
                    sorted_nei = list(dist_matrix_iter[curr_point].sort_values().index)
                    for pnt in points:
                        sorted_nei.remove(pnt)
                    random.shuffle(sorted_nei[:k_nei])
                    for i in sorted_nei:
                        if (curr_cap - df.loc[i, 'demand'] >= 0) and (dist_matrix_iter.iloc[curr_point, i] != np.inf):
                            path.append(i)
                            total_dist += dist_matrix_iter.iloc[curr_point, i]
                            dist += dist_matrix_iter.iloc[curr_point, i]
                            dist_matrix_iter.iloc[curr_point, i] = np.inf
                            curr_point = i
                            curr_cap -= df.loc[i, 'demand']
                            p = 1
                            points.append(i)
                            break
                    if p == 0:
                        path.append(0)
                        curr_cap -= df.loc[0, 'demand']
                        if curr_point != 0:
                            total_dist += dist_matrix_iter.iloc[curr_point, 0]
                            dist += dist_matrix_iter.iloc[curr_point, 0]
                        dist_matrix_iter.iloc[curr_point, 0] = np.inf
                        check = False
            if path == [0, 0] or path == [0]:
                global_check = False
            else:
                path_dict[j] = path
                dist_dict[j] = dist
                j+=1
        if total_dist_best >= total_dist:
            total_dist_best = total_dist
            path_dict_best = path_dict
            dist_dict_best = dist_dict
    return {'total_dist': total_dist_best, 'paths': path_dict_best, 'vehicles_dist': dist_dict_best}