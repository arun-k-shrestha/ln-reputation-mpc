# Code adapted from the original implementation by Sindura Saraswathi

import datetime
import networkx as nx
import random as rn
from itertools import islice
import pandas as pd
import re
import networkx.algorithms.shortest_paths.weighted as nx2
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function
import math
import configparser
import csv
from ordered_set import OrderedSet
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle
import random
import time
import Yao_MPC


config = configparser.ConfigParser()
config.read('config.ini')

startTime = datetime.datetime.now()
#--------------------------------------------
global use_log, case
# global prob_check, prob_dict
# prob_check = {}
# prob_dict = {}
epoch = int(config['General']['iterations'])

# print(epoch) #10000

#General
cbr = int(config['General']['cbr'])
src_type = config['General']['source_type']
dst_type = config['General']['target_type']
amt_type = config['General']['amount_type']
percent_malicious = float(config['General']['malicious_percent'])
NUM_LEARNING_SENDERS = int(config['General'].get('num_learning_senders', 40))
TRANSACTIONS_PER_SENDER = int(config['General'].get('transactions_per_sender', 25))

#LND
attemptcost = int(config['LND']['attemptcost'])/1000
attemptcostppm = int(config['LND']['attemptcostppm'])
timepref = float(config['LND']['timepref'])
apriori = float(config['LND']['apriori'])
rf = float(config['LND']['riskfactor'])
capfraction = float(config['LND']['capfraction'])
smearing = float(config['LND']['smearing'])

global bimodal_lnd_scale, lnd_scale
bimodal_scales = eval(config['LND']['bimodal_lnd_scale'])
if type(bimodal_scales) != list:
    print('Configuration file error: bimodal_lnd_scale not set as a list')
    raise 
bimodal_lnd_scale = []
for bls in bimodal_scales:
    bimodal_lnd_scale.append(float(bls))
    
lnd_scale = 3e5

# ---------------- Rating system (decayed S/F + Beta smoothing) ----------------
RATING_HALF_LIFE_H = 1.0   # H = 1 → after 1 new event, past evidence is worth 50% H = 2 → after 2 new events, past evidence is worth 50% (then H=1 will be 70.7)
RATING_ALPHA = 1.0         # Laplace prior alpha
RATING_BETA  = 1.0         # Laplace prior beta
MIN_RELIABILITY = 0.05     # below this, the sender avoids the node (optional

#---------------------------------------------------------------------------


def decay_factor(H: float): # H = 1 will yeild 0.5 and H = 2 will yeild 0.707
    H = max(float(H), 1e-9) # 1e-9, a small value, so that H > 0
    return 2.0 ** (-1.0 / H)


def update_reliability(G, sender: int, node: int, success: bool,
                       H: float = RATING_HALF_LIFE_H,
                       alpha: float = RATING_ALPHA,
                       beta: float = RATING_BETA):
    ratings = G.nodes[sender].setdefault("rating", {}) # just in case the node doesn't have rating ledger
    rec = ratings.get(node, {"S": 0.0, "F": 0.0, "p": alpha / (alpha + beta)}) # default is S=0.0, F=0.0 if not history found

    gamma = decay_factor(H)
    S = rec["S"] * gamma
    F = rec["F"] * gamma

    if success:
        S += 1.0
    else:
        F += 1.0

    p_hat = (alpha + S) / (alpha + beta + S + F)
    rec.update({"S": S, "F": F, "p": p_hat})
    ratings[node] = rec
    return p_hat

def get_reliability(G, sender: int, node: int,
                    alpha: float = RATING_ALPHA,
                    beta: float = RATING_BETA):
    rec = G.nodes[sender].get("rating", {}).get(node)
    if not rec:
        return alpha / (alpha + beta)  # (0.5 if alpha=beta=1) -> for new node with no rating
    return float(rec.get("p", alpha / (alpha + beta)))

def make_graph(G):
    df = pd.read_csv('LN_snapshot.csv')
    is_multi = df["short_channel_id"].value_counts() > 1
    df = df[df["short_channel_id"].isin(is_multi[is_multi].index)]
    node_num = {}
    nodes_pubkey = list(OrderedSet(list(df['source']) + list(df['destination'])))
    for i in range(len(nodes_pubkey)):
        G.add_node(i, honest=True, rating={}, score={})
        pubkey = nodes_pubkey[i]
        G.nodes[i]['pubkey'] = pubkey
        node_num[pubkey] = i
    for i in df.index:
        node_src = df['source'][i]
        node_dest = df['destination'][i]
        u = node_num[node_src]
        v = node_num[node_dest]
        G.add_edge(u,v)
        channel_id = df['short_channel_id'][i]
        block_height = int(channel_id.split('x')[0])
        G.edges[u,v]['id'] = channel_id
        G.edges[u,v]['capacity'] = int(df['satoshis'][i])#uncomment
        # G.edges[u,v]['capacity'] = 10**8 #new
        G.edges[u,v]['UpperBound'] = int(df['satoshis'][i])
        # G.edges[u,v]['UpperBound'] = 10**8 #new
        G.edges[u,v]['LowerBound'] = 0
        G.edges[u,v]['Age'] = block_height 
        G.edges[u,v]['BaseFee'] = df['base_fee_millisatoshi'][i]/1000
        G.edges[u,v]['FeeRate'] = df['fee_per_millionth'][i]/1000000
        G.edges[u,v]['Delay'] = df['delay'][i]
        G.edges[u,v]['htlc_min'] = int(re.split(r'(\d+)', df['htlc_minimum_msat'][i])[1])/1000
        G.edges[u,v]['htlc_max'] = int(re.split(r'(\d+)', df['htlc_maximum_msat'][i])[1])/1000
        G.edges[u,v]['LastFailure'] = 100
    return G

      
G = nx.DiGraph()
G = make_graph(G)

# Mark some nodes as malicious
def make_malicious(G, percent):
    malicious_nodes = random.sample(list(G.nodes()), int(len(G.nodes()) * percent))
    for node in malicious_nodes:
        G.nodes[node]["honest"] = False
make_malicious(G, percent_malicious)

y = []
cc = 0
#Sample balance from bimodal or uniform distribution
for i in G.edges:#new
    if 'Balance' not in G.edges[i]:
        cap = G.edges[i]['capacity']
        datasample = config['General']['datasampling']
        if datasample == 'bimodal':
            rng = np.linspace(0, cap, 10000)
            s = cap/10
            P = np.exp(-rng/s) + np.exp((rng - cap)/s)
            P /= np.sum(P)            
            x = int(np.random.choice(rng, p=P))
            # if cc<5:
            #     plt.plot(P)
            #     plt.show()
            #     cc += 1
        else:
            x = int(rn.uniform(0, G.edges[i]['capacity']))
            
        (u,v) = i
        G.edges[(u,v)]['Balance'] = x
        G.edges[(v,u)]['Balance'] = cap - x
        
        y.append(x)
        y.append(cap-x)
        
        if G.edges[v,u]['Balance'] < 0 or G.edges[v,u]['Balance'] > G.edges[i]['capacity']:
            print(i, 'Balance error at', (v,u))
            raise ValueError
            
        if G.edges[u,v]['Balance'] < 0 or G.edges[u,v]['Balance'] > G.edges[i]['capacity']:
            print(i, 'Balance error at', (u,v))
            raise ValueError
            
        if G.edges[(v,u)]['Balance'] + G.edges[(u,v)]['Balance'] != cap:
            print('Balance error at', (v,u))
            raise ValueError


def callable(source, target, amt, result, name):
    success = 0
    failure = 0
    def tracker(path, dist, p_amt, p_dist):
        global amt_dict
        amt_tracker = {}
        dist_tracker = {}
        # prob_tracker = {}
        for i in range(len(path)-1):
            u = path[i+1]
            v = path[i]
            if (u,v) in amt_dict:
                amt_tracker[(u,v)] = amt_dict[(u,v)]
            else:
                amt_tracker[(u,v)] = p_amt[(u,v)]
            if v in dist:
                dist_tracker[v] = dist[v]
            else:
                dist_tracker[v] = p_dist[v]
        dist_tracker[u] = dist[u]
        return amt_tracker, dist_tracker
    
    def shortest_simple_paths(G, source, target, weight):
        global prev_dict, paths, amt_dict, fee_dict, visited
        if source not in G:
            raise nx.NodeNotFound(f"source node {source} not in graph")
    
        if target not in G:
            raise nx.NodeNotFound(f"target node {target} not in graph")
    
        wt = _weight_function(G, weight)
    
        shortest_path_func = nx2._dijkstra
        
        listA = []
        listB = PathBuffer()
        amt_holder = PathBuffer()
        dist_holder = PathBuffer()
        # prob_holder = PathBuffer()
        prev_path = None
        prev_dist = None
        prev_amt = None
        # prev_prob = None
        visited = set()
        while True:
            if not prev_path:
                prev_dict = {}
                # prob_eclair = {} 
                paths = {source:[source]}
                dist = shortest_path_func(G, source=source, 
                                          target=target, 
                                          weight=weight, 
                                          pred=prev_dict, 
                                          paths=paths)
                path = paths[target]
                visited = set()
                amt_tracker, dist_tracker = tracker(path, dist, prev_amt, prev_dist)
                length = dist_tracker[target]
                listB.push(length, path)
                amt_holder.push(length, amt_tracker)
                dist_holder.push(length, dist_tracker)
                # prob_holder.push(length, prob_tracker)
            else:
                # global root,ignore_edges, H
                ignore_nodes = set()
                ignore_edges = set()
                for i in range(1, len(prev_path)):
                    root = prev_path[:i]
                    visited = set(root.copy())
                    root_length = prev_dist[root[-1]]        
                    amt_dict = {}
                    fee_dict = {}
                    prev_dict = {}
                    # prob_eclair = {}
                    if root[-1] != source:
                        temp_amt = prev_amt[(root[-1], root[-2])]
                        amt_dict[root[-1], root[-2]] = temp_amt
                        prev_dict = {root[-1]:[root[-2]]}
                        # prob_eclair[(root[-1], root[-2])] = prev_prob[(root[-1], root[-2])]
                    for path in listA:
                        if path[:i] == root:
                            ignore_edges.add((path[i - 1], path[i]))
                    try:
                        H = nx.subgraph_view(G, 
                        filter_node=lambda n: n not in ignore_nodes,
                        filter_edge=lambda u, v: (u, v) not in ignore_edges)
                        paths = {root[-1]:[root[-1]]}
                        dist = shortest_path_func(
                            H,
                            source=root[-1],
                            target=target,
                            weight=weight,
                            pred = prev_dict,
                            paths = paths
                            
                        )
                        try:
                            path = root[:-1] + paths[target]
                            amt_tracker, dist_tracker = tracker(path, dist, prev_amt, prev_dist)#
                            length = dist[target]
                            listB.push(root_length + length, path)
                            amt_holder.push(root_length + length, amt_tracker)
                            dist_holder.push(root_length + length, dist_tracker)
                            # prob_holder.push(root_length + length, prob_tracker)
                        except:
                            pass
                    except:
                        pass
                    ignore_nodes.add(root[-1])
    
            if listB:
                path = listB.pop()
                yield path
                listA.append(path)
                prev_path = path
                prev_amt = amt_holder.pop()
                prev_dist = dist_holder.pop()
                # prev_prob = prob_holder.pop()
            else:
                break

    class PathBuffer:
        def __init__(self):
            self.paths = set()
            self.sortedpaths = []
            self.counter = count()
    
        def __len__(self):
            return len(self.sortedpaths)
    
        def push(self, cost, path):
            hashable_path = tuple(path)
            if hashable_path not in self.paths:
                heappush(self.sortedpaths, (cost, next(self.counter), path))
                self.paths.add(hashable_path)
    
        def pop(self):
            (cost, num, path) = heappop(self.sortedpaths)
            hashable_path = tuple(path)
            self.paths.remove(hashable_path)
            return path

    def dijkstra_lnd(G, sources, weight, pred=None, paths=None, cutoff=None, target=None):
        try:
            G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
            push = heappush
            pop = heappop
            dist = {}  # dictionary of final distances
            seen = {}
            probability_dist = {}
            p_path = {}
            # fringe is heapq with 3-tuples (distance,c,node)
            # use the count c to avoid comparing nodes (may not be able to)
            c = count()
            fringe = []
            for source in sources:
                seen[source] = 0
                push(fringe, (0, 0, 1, next(c), source))
            while fringe:
                # print(fringe)
                # print(probability_dist)
                # print(dist)
                (prob_dist, d, path_prob,  _, v) = pop(fringe)
                
                if v in dist:
                    continue  # already searched this node.
                dist[v] = d
                probability_dist[v] = prob_dist
                p_path[v] = path_prob
                if v == target:
                    break
                for u, e in G_succ[v].items():
                    cost, prob = weight(v, u, e)
                    if cost is None:
                        continue
                    vu_dist = dist[v] + cost #add only additive weights
                    vu_prob = vu_dist + prob
                    if cutoff is not None:
                        if vu_prob > cutoff:
                            continue
                    if u in probability_dist:
                        u_dist = probability_dist[u]
                        u_prob = p_path[u]
                        if vu_prob < u_dist:
                            raise ValueError("Contradictory paths found:", "negative weights?")
                        elif pred is not None and vu_prob < u_dist and prob > p_path[u]:
                            pred[u].append(v)
                    elif u not in seen or vu_prob < seen[u]:
                        seen[u] = vu_prob
                        push(fringe, (vu_prob, vu_dist, prob, next(c), u))
                        if paths is not None:
                            paths[u] = paths[v] + [u]
                        if pred is not None:
                            pred[u] = [v]
                    elif vu_prob == seen[u]:# or prob <= p_path[u]:
                        if pred is not None:
                            pred[u].append(v)
            # The optional predecessor and path dictionaries can be accessed
            # by the caller via the pred and paths objects passed as arguments.
            return probability_dist
        except:
            raise

    def sub_func(u,v, amount):
        global amt_dict, fee_dict
        fee = round(G.edges[u,v]["BaseFee"] + amount*G.edges[u,v]["FeeRate"], 5)
        fee_dict[(u,v)] = fee
        if u==source:
            fee_dict[(u,v)] = 0
            fee = 0
        amt_dict[(u,v)] = round(amount+fee, 5)
     
    
    def compute_fee(v,u,d):
        global fee_dict, amt_dict, cache_node, visited
        if v == target:
            cache_node = v
            sub_func(u,v,amt)
        else:
            if cache_node != v:
                visited.add(cache_node)
                cache_node = v          
            amount = amt_dict[(v, prev_dict[v][0])] 
            sub_func(u,v,amount)
    
    
    def primitive(c, x):
        # if datasample == 'uniform':
        #     s = 3e5 #fine tune 's' for improved performance
        # else:
        #     s = c/10
        global lnd_scale
        test_scales = config['LND']['test_scales']
        if test_scales == 'True':
            s = c/lnd_scale
        else:
            s = lnd_scale
        if lnd_scale == 3e5:
            s = 3e5
        ecs = math.exp(-c/s)
        exs = math.exp(-x/s)
        excs = math.exp((x-c)/s)
        norm = -2*ecs + 2
        if norm == 0:
            return 0
        return (excs - exs)/norm
    
    
    def integral(cap, lower, upper):
        return primitive(cap, upper) - primitive(cap, lower)
    
    
    def bimodal(cap, a_f, a_s, a):
        prob = integral(cap, a, a_f)
        if prob is math.nan:
            return 0
        reNorm = integral(cap, a_s, a_f)
        
        if reNorm is math.nan or reNorm == 0:
            return 0
        prob /= reNorm
        if prob>1:
            return 1
        if prob<0:
            return 0
        return prob
    
    
    #v - target, u - source, d - G.edges[v,u]
    def lnd_cost(v,u,d):
        global prob_check, prob_dict#new
        global timepref, case

        if get_reliability(G, source, v) < MIN_RELIABILITY:
            return float('inf'), float('inf') # (distance, cost) -> (fees+delay, probability+penalty)
    
        compute_fee(v,u,d)        
        timepref *= 0.9
        defaultattemptcost = attemptcost+attemptcostppm*amt_dict[(u,v)]/1000000
        penalty = defaultattemptcost * ((1/(0.5-timepref/2)) - 1)
        cap = G.edges[u,v]["capacity"]
        if amt_dict[(u,v)] > cap:
            return float('inf'), float('inf')
        if case == 'apriori':
            prob_weight = 2**G.edges[u,v]["LastFailure"]
            den = 1+math.exp(-(amt_dict[(u,v)] - capfraction*cap)/(smearing*cap))
            nodeprob = apriori * (1-(0.5/den))
            prob = nodeprob * (1-(1/prob_weight))
            # prob_check[u,v] = prob
        elif case == 'bimodal':
            prob = bimodal(cap, G.edges[u,v]['UpperBound'], G.edges[u,v]['LowerBound'], amt_dict[(u,v)]) 
            # prob_dict[(v,u)] = prob
        if v == target:
            prob_dict[v,u] = prob
        else:
            pred_node = prev_dict[v][0]
            if u == source:
                if G.edges[u,v]["Balance"]<amt_dict[(u,v)]:
                    prob = 0
                else:
                    prob = 1
            prob *= prob_dict[pred_node, v]
            prob_dict[v,u] = prob
        if prob == 0 or prob < 0.01:
            cost = float('inf')
        else:
            cost = penalty/prob
            p_rep = get_reliability(G, source, v)  # v is the next-hop node being evaluated
            p_rep = max(p_rep, 1e-6)

            LAMBDA_REP = 1.0  # we can tune this

            rep_term = LAMBDA_REP * (-math.log(p_rep)) # or what happens if we try? LAMBDA_REP * (1.0 - p_rep)

            cost = cost + rep_term

        dist = fee_dict[(u,v)] + G.edges[u,v]['Delay']*amt_dict[(u,v)]*rf
        return dist, cost


    #v - target, u - source, d - G.edges[v,u]
    def lnd_cost_test(v,u,d):
        # global prob_check, prob_dict#new
        global timepref, case
        compute_fee(v,u,d)        
        timepref *= 0.9
        defaultattemptcost = attemptcost+attemptcostppm*amt_dict[(u,v)]/1000000
        penalty = defaultattemptcost * ((1/(0.5-timepref/2)) - 1)
        cap = G.edges[u,v]["capacity"]
        
        if amt_dict[(u,v)] > cap:
            return float('inf'), float('inf')
        
        # prob = (G.edges[u,v]['UpperBound'] - amt_dict[(u,v)]+1)/(G.edges[u,v]['UpperBound'] - G.edges[u,v]['LowerBound']+1)
        if G.edges[u,v]["capacity"] != 0:
            prob = 1 - (amt_dict[(u,v)]/cap)
            
        if v == target:
            prob_dict[v,u] = prob
        else:
            pred_node = prev_dict[v][0]
            if u == source:
                if G.edges[u,v]["Balance"]<amt_dict[(u,v)]:
                    prob = 0
                else:
                    prob = 1
            prob *= prob_dict[pred_node, v]
            prob_dict[v,u] = prob
        if prob < 0.01:
            cost = float('inf')
        else:
            cost = penalty/prob
        dist = fee_dict[(u,v)] + G.edges[u,v]['Delay']*amt_dict[(u,v)]*rf
        return dist, cost
    
     
    #simulate payment routing on the path found by the LN clients
    def route(G, path, source, target):
        nonlocal success, failure
        try:
            amt_list = []
            mpc_passed_hops = set()  # tracks hops (u,v) that passed MPC in this attempt
            total_fee = 0
            total_delay = 0
            path_length = len(path)
            for i in range(path_length-1):
                v = path[i]
                u = path[i+1]

                # Fail fast if sender distrusts the next-hop v
                if get_reliability(G, source, v) < MIN_RELIABILITY:
                    return [path, total_fee, total_delay, path_length, 'Failure']
                
                if v == target:
                    amt_list.append(amt)
                fee = G.edges[u,v]["BaseFee"] + amt_list[-1]*G.edges[u,v]["FeeRate"]
                if u==source:
                    fee = 0
                fee = round(fee, 5)
                a = round(amt_list[-1] + fee, 5)
                amt_list.append(a)
                total_fee +=  fee
                total_delay += G.edges[u,v]["Delay"]
            path = path[::-1]
            amt_list = amt_list[::-1]
            amount = amt_list[0]
            for i in range(path_length-1):
                u = path[i]
                v = path[i+1]
                fee = G.edges[u,v]["BaseFee"] + amt_list[i+1]*G.edges[u,v]["FeeRate"]
                if u==source:
                    fee = 0
                fee = round(fee, 5)
                
                bal = G.edges[u, v]["Balance"]
                cap = G.edges[u, v]["capacity"]
                upper = cap + max(1, int(amount))
                # important: we are not couting failure because this is pre-check not system failure.
                if G.nodes[u].get("honest", True): # only checking honest node because the dishonest node will bypass this
                    if not Yao_MPC.Yao_Millionaires_Protocol(amount, bal, upper, 40):
                        # failure += 1 
                        return [path, total_fee, total_delay, path_length, 'Failure']
                mpc_passed_hops.add((u, v))

                if not G.nodes[u].get("honest",True):
                    failure += 1 
                    update_reliability(G, source, v, success=False)
                    return [path, total_fee, total_delay, path_length, 'Failure']

                # even after the MPC check, if it fails, we provide them a bad rating. This will be true for dishonest node
                if amount > G.edges[u, v]["Balance"] or amount <= 0:
                    failure += 1
                    # penalize the hop that "looked ok under MPC" but failed during HTLC
                    # choose to penalize v (next-hop) OR u (forwarder). Here: penalize v.
                    update_reliability(G, source, v, success=False)
                    return [path, total_fee, total_delay, path_length, 'Failure']

                amount = round(amount - fee, 5)

                # Another "during" failure after MPC passed
                if v == target and amount != amt:
                    failure += 1
                    # penalize the last hop actor; here: penalize u (forwarder)
                    update_reliability(G, source, v, success=False)
                    return [path, total_fee, total_delay, path_length, 'Failure']
          
            # release_locked(i-1, path)
            success += 1
            # reward nodes on the successful path (excluding sender)
            for node in path[1:]:
                update_reliability(G, source, node, success=True)
            print(G.nodes[source]["rating"])
            return [path, total_fee, total_delay, path_length, 'Success']
        except Exception as e:
            print(e)
            failure +=1
            return "Routing Failed due to the above error"
    
    
    #----------------------------------------------
    def dijkstra_caller(res_name, func):
        dist = nx2._dijkstra(G, source=target, target=source, weight = func, pred=prev_dict, paths=paths)
        res = paths[source]
        #print("Path found by", res_name, res[::-1])
        result[res_name] = route(G, res, source, target)
        
    def modified_dijkstra_caller(res_name, func):
        dist = dijkstra_lnd(G, sources=[target], target=source, weight = func, pred=prev_dict, paths=paths)
        res = paths[source]
        #print("Path found by", res_name, res[::-1])
        result[res_name] = route(G, res, source, target)
        
        
    def helper(name, func):
        global fee_dict, amt_dict, cache_node, visited
        global prev_dict, paths, prob_dict
        global use_log, case
        global bimodal_lnd_scale, lnd_scale
        
        def clear_globals():
            global fee_dict, amt_dict, cache_node, visited
            global prev_dict, paths, prob_dict
            fee_dict = {}
            amt_dict = {}
            prob_dict = {}
            cache_node = target
            visited = set()
            prev_dict = {}
            paths = {target:[target]}
            
        try:
            clear_globals()
            case = config['LND']['LND1']
            modified_dijkstra_caller('LND1', func)       
        except Exception as e:
            print("Error:", e)
            pass
            
    algo = {'LND':lnd_cost} 
    global fee_dict, amt_dict, cache_node, visited
    global prev_dict, paths, prob_dict
    global bimodal_lnd_scale, lnd_scale
    
    fee_dict = {}
    amt_dict = {}
    prob_dict = {}
    cache_node = target
    visited = set()
    prev_dict = {}
    paths = {target:[target]}
    
    helper(name, algo[name])
    return result, success, failure
        
# startTime = datetime.datetime.now()
start = time.perf_counter()  

if __name__ == '__main__':   

    def node_classifier():
        df = pd.read_csv('LN_snapshot.csv')
        is_multi = df["short_channel_id"].value_counts() > 1
        df = df[df["short_channel_id"].isin(is_multi[is_multi].index)]
        nodes_pubkey = list(OrderedSet(list(df['source']) + list(df['destination'])))
        node_num = {}
        for i in range(len(nodes_pubkey)):
            pubkey = nodes_pubkey[i]
            node_num[pubkey] = i   
        src_count = df['source'].value_counts()
        node_cap = df[['source', 'satoshis']]
        node_cap = node_cap.groupby('source').sum()
        well_node = []
        fair_node = []
        poor_node = []
        for i in node_cap.index:
            chan_cnt = src_count[i]
            cap = node_cap.loc[i,'satoshis']
            if cap >= 10**6 and chan_cnt>5:
                well_node.append(node_num[i])
            elif cap > 10**4 and cap < 10**6 and chan_cnt>5:
                fair_node.append(node_num[i])
            elif chan_cnt<=5:
                poor_node.append(node_num[i])            
        return well_node, fair_node, poor_node
    
    
    def node_selector(node_type):
        if node_type == 'well':
            return rn.choice(well_node)
        elif node_type == 'fair':
            return rn.choice(fair_node)
        elif node_type == 'poor':
            return rn.choice(poor_node)
        else:
            return rn.randint(0,13129)
        
        
    def node_ok(source, target):
        src_max = 0
        tgt_max = 0
        for edges in G.out_edges(source):
            src_max = max(src_max, G.edges[edges]['Balance'])
        for edges in G.in_edges(target):
            tgt_max = max(tgt_max, G.edges[edges]['Balance'])
        upper_bound = int(min(src_max, tgt_max))
        if amt < upper_bound:
            return True
        else:
            return False
        
            
    def node_cap(source, target):
        src_max = 0
        tgt_max = 0
        for edges in G.out_edges(source):
            src_max = max(src_max, G.edges[edges]['Balance'])
        for edges in G.in_edges(target):
            tgt_max = max(tgt_max, G.edges[edges]['Balance'])
        upper_bound = int(min(src_max, tgt_max))
        return upper_bound
    
    print("\n" + "="*80)
    print("LEARNING SENDER EXPERIMENT")
    print("="*80)
    print(f"Number of learning senders: {NUM_LEARNING_SENDERS}")
    print(f"Transactions per sender: {TRANSACTIONS_PER_SENDER}")
    print(f"Total transactions: {NUM_LEARNING_SENDERS * TRANSACTIONS_PER_SENDER}")
    print(f"Malicious nodes: {percent_malicious*100:.1f}%")
    print("="*80 + "\n")

    
    well_node, fair_node, poor_node = node_classifier()

    print(f"Total nodes - Well: {len(well_node)}, Fair: {len(fair_node)}, Poor: {len(poor_node)}")
    
    # Stratified selection of learning senders
    num_hub = min(12, len(well_node))
    num_medium = min(16, len(fair_node))
    num_peripheral = min(12, len(poor_node))

    # Adjust if we can't get exactly NUM_LEARNING_SENDERS
    total_available = num_hub + num_medium + num_peripheral
    if total_available < NUM_LEARNING_SENDERS:
        print(f"Warning: Only {total_available} suitable senders found")
        NUM_LEARNING_SENDERS = total_available
    
    learning_senders = {
        'hub': random.sample(well_node, num_hub) if len(well_node) >= num_hub else well_node,
        'medium': random.sample(fair_node, num_medium) if len(fair_node) >= num_medium else fair_node,
        'peripheral': random.sample(poor_node, num_peripheral) if len(poor_node) >= num_peripheral else poor_node
    }

    all_learning_senders = (learning_senders['hub'] + 
                           learning_senders['medium'] + 
                           learning_senders['peripheral'])
    
    print(f"\nSelected {len(all_learning_senders)} learning senders:")
    print(f"  Hub nodes: {len(learning_senders['hub'])}")
    print(f"  Medium nodes: {len(learning_senders['medium'])}")
    print(f"  Peripheral nodes: {len(learning_senders['peripheral'])}")


    i = 0
         
    work = []              
    result_list = [] 
    prob_dict = {}
    
    algos = config['General']['algos'].split('|')
    amt_end_range = int(config['General']['amt_end_range'])
    
    print("\nBuilding transaction list...")
    
    # For each learning sender, create their transaction sequence
    for sender in all_learning_senders:
        # Determine sender type for tracking
        if sender in learning_senders['hub']:
            sender_type = 'hub'
        elif sender in learning_senders['medium']:
            sender_type = 'medium'
        else:
            sender_type = 'peripheral'
        
        # Each sender performs TRANSACTIONS_PER_SENDER transactions
        for tx_num in range(TRANSACTIONS_PER_SENDER):
            # Select target
            target = node_selector(dst_type)
            
            # Ensure valid source-target pair
            max_retries = 100
            retry_count = 0
            while (target == sender or 
                   target not in G.nodes() or 
                   sender not in G.nodes()):
                target = node_selector(dst_type)
                retry_count += 1
                if retry_count > max_retries:
                    break
            
            if retry_count > max_retries:
                continue
            
            # Amount selection
            if amt_type == 'fixed':
                amt = int(config['General']['amount'])
            elif amt_type == 'random':
                k = (tx_num % amt_end_range) + 1
                amt = rn.randint(10**(k-1), 10**k)
            
            # Check if transaction is feasible
            if not node_ok(sender, target):
                continue
            
            # Progress indicator
            if len(work) % 10000 == 0 and len(work) > 0:
                print(f"  Generated {len(work)} transactions...")
            
            # Build result dictionary with tracking info
            result = {
                'Source': sender,
                'Target': target,
                'Amount': amt,
                'SenderType': sender_type,
                'TxNumber': tx_num,
                'TotalTxSoFar': len(work)
            }
            
            work.append((sender, target, amt, result, 'LND'))
    
    print(f"\nTotal valid transactions generated: {len(work)}")
    print(f"Starting parallel execution with 8 processes...\n")


    pool = mp.Pool(processes=8)
    a = pool.starmap(callable, work)
    pool.close()
    pool.join()

    result_dicts = [r for (r, s, f) in a]
    total_success = sum(s for (r, s, f) in a)
    total_failure = sum(f for (r, s, f) in a)

    print(total_success, total_failure)

    # If you force only LND in work, then:
    # algos = ['LND']
    # k = 1

    k = len(algos)

    if not result_dicts:
        raise RuntimeError("No results returned (result_dicts is empty).")

    ans_list = []
    for start in range(0, len(result_dicts), k):
        temp = {}
        for res in result_dicts[start:start + k]:
            temp.update(res)
        ans_list.append(temp)

    if not ans_list:
        raise RuntimeError("ans_list is empty; nothing to write.")

    fields = list(ans_list[0].keys())

    filename = config['General']['filename']
    # with open(filename, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fields)
    #     writer.writeheader()
    #     for row in ans_list:
    #         writer.writerow(row)

        
#endTime = datetime.datetime.now()

# endtime = time.perf_counter()
# print(endtime - start)

endTime = datetime.datetime.now()
print(endTime - startTime)
    
