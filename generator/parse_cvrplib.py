import numpy as np
import pandas as pd
import requests

def split(txt, seps):
    default_sep = seps[0]
    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]

def get_instance_from_CVRPLIB(set_data = 'P', model = "P-n101-k4"):
    file = set_data + '/' + model + ".vrp"
    res = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/' + file)
    lines = res.text.split('\n')
    solution = set_data + '/' + model + ".sol"
    res = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/' + solution)
    solution_lines = res.text.split('\n')
    cap = np.int(lines[5][len('CAPACITY : '):])
    dim = np.int(lines[3][len('DIMENSION : '):])
    cars = np.int(model.split('-')[-1][1:])
    baseline = np.int(solution_lines[-2][len('Cost '):])
    SYMBOLS = '\n'
    nodes = [item.translate(SYMBOLS).strip() for item in lines[7:7+dim]]
    demands = [item.translate(SYMBOLS).strip() for item in lines[7+dim+1:7+dim+1+dim]]
    for i in range(len(nodes)):
        nodes[i] = split(nodes[i], (',', ' '))
        demands[i] = split(demands[i], (',', ' '))
    df_nodes = pd.DataFrame(data = nodes, columns = ['id', 'latitude', 'longitude'])
    df_demands = pd.DataFrame(data = demands, columns = ['id', 'demand'])
    df = pd.merge(df_nodes, df_demands).drop('id', 1)
    df= df.astype(int)
    df['cap'] = cap
    df['cars'] = cars
    tour = []
    paths = {}
    car_index = 0
    for i in range(len(solution_lines)-2):
        parse_solution = split(solution_lines[i], ' ')
        tour.append(0)
        for j in range(len(parse_solution) - 3):
            tour.append(parse_solution[j+2])
        tour.append(0)
        paths[car_index] = tour
    return df, tour, paths