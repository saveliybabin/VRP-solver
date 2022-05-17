import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, squareform


class Beamsearch(object):
    """Class for managing internals of beamsearch procedure.
    References:
        - General: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        - For TSP: https://github.com/chaitjo/graph-convnet-tsp/blob/master/utils/beamsearch.py
    """

    def __init__(self, beam_size, batch_size, num_nodes, car_num,  device='cpu', decode_type='greedy'):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            num_nodes: Number of nodes in TSP tours
            device: GPU/CPU device
            decode_type: Allows sampling from multinomial or greedy decoding
            car_num: number of vehicles
        """
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.device = device
        self.decode_type = decode_type
        self.curr_cap = torch.zeros(batch_size, self.beam_size, dtype=torch.int64).to(self.device)
        self.car_num = car_num
        self.car_cap = torch.ones(batch_size, self.beam_size, dtype=torch.int64).to(self.device)
        
        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size, dtype=torch.long).to(self.device)
        # Set counter of zeros in tour
        self.zero_num = torch.ones(batch_size, dtype=torch.int64).to(self.device)
        # Mask for constructing valid hypothesis
        self.mask = torch.ones(batch_size, beam_size, num_nodes, dtype=torch.float).to(self.device)
        # Mask the starting node of the beam search
        # Assumption: it is possible to make first step for all nodes, e. g. start demand is zero for all 
        self.update_mask(self.start_nodes, torch.zeros(batch_size, num_nodes, dtype=torch.float))  
        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size, dtype=torch.float).to(self.device)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def get_current_state(self):
        """Get the output of the beam at the current timestep.
        """
        current_state = (self.next_nodes[-1].unsqueeze(2)
                         .expand(self.batch_size, self.beam_size, self.num_nodes))
        return current_state

    def get_current_origin(self):
        """Get the backpointers for the current timestep.
        """
        return self.prev_Ks[-1]

    def advance(self, trans_probs, demand, step):
        """Advances the beam based on transition probabilities.
        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
            nodes: locations+demands of nodes
        """
        # Compound the previous scores
        if len(self.prev_Ks) > 0:
            beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            beam_lk[:, 1:] = -1e20 * torch.ones(beam_lk[:, 1:].size(), dtype=torch.float).to(self.device)
        # Multiply by mask
        self.mask[:, :, 0] = 5
        self.mask[self.mask == 0] = 1e20
    #     self.mask[:, :, 0][(self.mask[:, :, 0] != 1e20).nonzero(as_tuple=True)] = 3

        current_demand = torch.zeros(self.batch_size, self.beam_size, dtype=torch.float).to(self.device)

    #     Suggest only possible variants by capacities
        if len(self.prev_Ks) > 0:
            for pos in range(0, self.beam_size):
                ends = pos * torch.ones(self.batch_size, 1).to(self.device) 
                hyp_tours = self.get_hypothesis(ends)
                for idx in range(self.batch_size):
                    current_demand[idx, pos] = demand[idx][(self.array_separator(hyp_tours[idx][:step+1])[-1])].sum()
        #     self.mask[:, :, 0][(self.zero_num >= self.car_num).nonzero(as_tuple=True)] = 1e10

            # Check the nodes that will overdemand vehicles in the next step
            full_future_nodes = (1 - current_demand[:, :, None].expand([self.batch_size, self.beam_size, self.num_nodes]) - demand[:, None].expand([self.batch_size, self.beam_size, self.num_nodes])) < 0
            temp_mask = torch.ones(self.mask.size(), dtype=torch.float).to(self.device) + full_future_nodes.type(torch.int64)*1e20
        else:
            temp_mask = torch.ones(self.mask.size(), dtype=torch.float).to(self.device)
        beam_lk = beam_lk * self.mask * temp_mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # Update scores
        self.scores = bestScores
        # Update backpointers
        prev_k = bestScoresId // self.num_nodes
        self.prev_Ks.append(prev_k)
        # Update outputs
        new_nodes = bestScoresId - prev_k * self.num_nodes
        self.next_nodes.append(new_nodes)
        # Update capacities
        new_demand = demand[:, None].expand([self.batch_size, self.beam_size, self.num_nodes]).gather(2, self.next_nodes[-1][:, :, None]).reshape(self.batch_size, self.beam_size)
        # If demand is zero that means return to depot, so we have to zero current capacity of the vechicle
        self.curr_cap = self.curr_cap * (new_demand != 0).type(torch.int64) + new_demand
#         self.zero_num[(new_nodes == 0)] += 1
        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask = self.mask.gather(1, perm_mask)
        # Mask newly added nodes
        self.update_mask(new_nodes, demand)
    

    def update_mask(self, new_nodes, demand):
        """Sets new_nodes to zero in mask.
        """
        arr = (torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(torch.long).to(self.device))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        update_mask = 1 - torch.eq(arr, new_nodes).type(torch.float).to(self.device)
        self.mask = self.mask * update_mask
#         self.mask[:, :, 0] = 3
#         self.mask[self.mask == 0] = 1e10
#         self.mask[:, :, 0][(self.zero_num >= self.car_num).nonzero(as_tuple=True)] = 1e10


    def sort_best(self):
        """Sort the beam.
        """
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score and index of the best hypothesis in the beam.
        """
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hypothesis(self, k):
        """Walk back to construct the full hypothesis.
        Args:
            k: Position in the beam to construct (usually 0s for most probable hypothesis)
        """
#         assert self.num_nodes == len(self.prev_Ks) + 1 + self.car_num
        
        k = k.type(torch.long).to(self.device)
        hyp = -1 * torch.ones(self.batch_size, self.num_nodes + self.car_num - 1, dtype=torch.long).to(self.device)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k)
        return hyp
    
    def array_separator(self, arr):
        out = []
        out_temp = [0]
        for i in arr[1:]:
            if i == 0:
                if len(out_temp) != 1:
                    out_temp.append(0)
                    out.append(torch.tensor(out_temp,  dtype=torch.int64).to(self.device))
                out_temp = [0]
            else:
                out_temp.append(i)
        if len(out_temp) != 0:
            out_temp.append(0)
            out.append(torch.tensor(out_temp,  dtype=torch.int64).to(self.device))
        return out
    
    def get_cost(self, tour, x):
        dist_matrix = squareform(pdist(x[:, :2].cpu().numpy()))
        dist_matrix_ = torch.tensor(dist_matrix.copy(),  dtype=torch.float).to(self.device)
        dist_matrix_[(np.arange(dist_matrix_.size(0))), (np.arange(dist_matrix_.size(0)))] = 0
        cost = dist_matrix_[tour[:-1], tour[1:]].sum()
        return cost

    def is_valid_tour(self, tour, demand, car_num):

        """Sanity check: tour visits all nodes given and cope with capasities
        """
        graph_size = demand.size()[0]
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = torch.sort(tour)[0]
        # Sorting it should give all zeros at front and then 1...n

        a_check = (torch.arange(0, graph_size) == sorted_pi[-graph_size:]).all()
        b_check = (sorted_pi[:-graph_size] == 0).all()

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        used_cap = []
        for t in self.array_separator(tour):
            used_cap.append(demand[t].sum())

        c_check = (np.array(used_cap) <= 1.0 + 1e-5).all()
        d_check = len(used_cap) <= car_num

    #     print(np.int(a_check), np.int(b_check), np.int(c_check), np.int(d_check))
        return a_check and b_check and c_check and d_check
    
    def get_best_tour_and_score(self, shortest_tours, demand, graph, beam_size):
        # Compute current tour lengths
        shortest_lens = [1e6] * len(shortest_tours)
        for idx in range(len(shortest_tours)):
            shortest_lens[idx] = self.get_cost(shortest_tours[idx].to(self.device), graph[idx].to(self.device))

        # Iterate over all positions in beam (except position 0 --> highest probability)
        for pos in range(1, beam_size):
            ends = pos * torch.ones(batch_size, 1).to(self.device)  # New positions
            hyp_tours = self.get_hypothesis(ends)
            for idx in range(len(hyp_tours)):
                hyp_nodes = hyp_tours[idx].to(self.device)
                hyp_len = self.get_cost(hyp_nodes.to(self.device), graph[idx].to(self.device))
                # Replace tour in shortest_tours if new length is shorter than current best
                if self.is_valid_tour(shortest_tours[idx], demand[idx], car_num):  
                    if hyp_len < shortest_lens[idx] and self.is_valid_tour(hyp_nodes, demand[idx], car_num):
                        shortest_tours[idx] = hyp_tours[idx]
                        shortest_lens[idx] = hyp_len
                else:
                    if self.is_valid_tour(hyp_nodes, demand[idx], car_num):
                        shortest_tours[idx] = hyp_tours[idx]
                        shortest_lens[idx] = hyp_len
        return shortest_tours, shortest_lens
