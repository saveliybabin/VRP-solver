from torch.utils.data import Dataset
import torch
import os
import pickle5 as pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search
from heurictics.op_tools import main
from generator.distance_matrix import distance_function
from tqdm.notebook import tqdm
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd


def nearest_neighbor_graph(nodes, neighbors, knn_strat):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W

class CVRP(object):
   

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, _ = dataset.size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
#         print(torch.arange(0, graph_size, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size))
#         print(sorted_pi[:, -graph_size:])
#         print((sorted_pi[:, :-graph_size] == 0).all())
        # Sorting it should give all zeros at front and then 1...n
    
        assert (
            torch.arange(0, graph_size, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset[:, :, 2][:, :1], -1.0),
                dataset[:, :, 2][:, 1:]
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset[:, :, 2][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = dataset#torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0][:, :2] -  dataset[:, :1, :2].reshape(-1, 2)).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1][:, :2] -  dataset[:, :1, :2].reshape(-1, 2)).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    import numpy as np
    depot, loc, demand, capacity, *args = args
    capacity = np.median(capacity)
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat))
    }

def make_instance_and_solution(self, args):
    import numpy as np
    depot, loc, demand, capacity, *args = args
    capacity = np.median(capacity)
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    all_loc = np.concatenate((np.array(depot).reshape(1, -1), np.array(loc)), 0)
    all_demand =  np.concatenate((np.zeros(1), np.array(demand)), 0)
    graph = torch.ByteTensor(nearest_neighbor_graph(all_loc, self.neighbors, self.knn_strat))
    dist = pd.DataFrame(distance_function(all_loc, is_coord = False).values * graph.numpy())
    nodes_cars_dist = {21: 5}
    solution = main(all_demand, dist, cars = nodes_cars_dist[all_loc.shape[0]], cap = capacity)
    np.fill_diagonal(solution['adj'], 1)
    solution['adj'] =  torch.tensor(solution['adj'], dtype=torch.float)
    solution['total_dist'] =  torch.tensor(solution['total_dist'], dtype=torch.float)
    solution['tour_nodes'] =  torch.tensor(solution['tour_nodes'], dtype=torch.float)
    
    output = {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'graph': graph
    }
    output.update(solution)
    return output

class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, supervised = True, neighbors = 0.2, knn_strat='percentage'):
        super(VRPDataset, self).__init__()
        
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            if supervised:
                self.data = [make_instance_and_solution(self, args) for args in tqdm(data[offset:offset+num_samples])]
            else:
                self.data = [make_instance(self, args) for args in data[offset:offset+num_samples]]
                

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    

# class SDVRP(object):

#     NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

#     VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

#     @staticmethod
#     def get_costs(dataset, pi):
#         batch_size, graph_size = dataset['demand'].size()

#         # Each node can be visited multiple times, but we always deliver as much demand as possible
#         # We check that at the end all demand has been satisfied
#         demands = torch.cat(
#             (
#                 torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
#                 dataset['demand']
#             ),
#             1
#         )
#         rng = torch.arange(batch_size, out=demands.data.new().long())
#         used_cap = torch.zeros_like(dataset['demand'][:, 0])
#         a_prev = None
#         for a in pi.transpose(0, 1):
#             assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
#                 "Cannot visit depot twice if any nonzero demand"
#             d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
#             demands[rng, a] -= d
#             used_cap += d
#             used_cap[a == 0] = 0
#             a_prev = a
#         assert (demands == 0).all(), "All demand must be satisfied"

#         # Gather dataset in order of tour
#         loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
#         d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

#         # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
#         return (
#             (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
#             + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
#             + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
#         ), None

#     @staticmethod
#     def make_dataset(*args, **kwargs):
#         return VRPDataset(*args, **kwargs)

#     @staticmethod
#     def make_state(*args, **kwargs):
#         return StateSDVRP.initialize(*args, **kwargs)

#     @staticmethod
#     def beam_search(input, beam_size, expand_size=None,
#                     compress_mask=False, model=None, max_calc_batch_size=4096):
#         assert model is not None, "Provide model"
#         assert not compress_mask, "SDVRP does not support compression of the mask"

#         fixed = model.precompute_fixed(input)

#         def propose_expansions(beam):
#             return model.propose_expansions(
#                 beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
#             )

#         state = SDVRP.make_state(input)

#         return beam_search(state, beam_size, propose_expansions)

