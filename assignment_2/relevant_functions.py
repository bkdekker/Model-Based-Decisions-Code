import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from time import time

#from assignment_2.Granovetter_experiment_results import betweenness_seeds, build_networks, draw_thresholds, final_adoption_fraction, greedy_celf, high_degree_seeds, kcore_seeds, random_seeds, threshold_dynamics, time_to_fraction

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.dpi'] = 120
# The threshold mode determines how the threshold values are generated
# Each node samples a threshold value from a specified distribution (defined by mode)
# The phi_mean and phi_std parameters are used to control the mean and standard deviation of the normal distribution
class Granover_methods():
    def draw_thresholds(G, mode='uniform', phi_mean=0.3, phi_std=0.1):
        if mode == 'uniform':
            return {n: np.random.uniform(0, 1) for n in G}
        if mode == 'beta':
            a, b = 2, 5
            return {n: np.random.beta(a, b) for n in G}
        if mode == 'constant':
            return {n: float(phi_mean) for n in G}
        if mode == 'normal':
            vals = np.clip(np.random.normal(phi_mean, phi_std, len(G)), 0, 1)
            return {n: float(vals[i]) for i, n in enumerate(G)}
        raise ValueError(f"Unknown threshold mode: {mode}")

    # This loop iterates over each graph in the graphs dictionary
    # For each graph, it draws threshold values using the draw_thresholds function
    # It then calculates the mean of these threshold values and prints it

    def threshold_dynamics(G, seeds, thresholds, max_steps=100):
        adopted = {n: 0 for n in G}
        for s in seeds:
            adopted[s] = 1 # initially, all seed nodes are adopted

        history = [sum(adopted.values()) / len(G)] #the history will store the fraction of adopted nodes over time, initially it is the fraction of adopted seed nodes

        for _ in range(max_steps): # iterate max_steps times
            new_adopted = adopted.copy()
            changed = False
            for i in G: # iterate over all nodes i in the graph G
                if adopted[i] == 0: # if node i is not adopted yet
                    neigh = list(G.neighbors(i))
                    if not neigh: # if node i has no neighbors, skip it
                        continue
                    frac = sum(adopted[j] for j in neigh) / len(neigh) # calculate the fraction of adopted neighbors of node i
                    if frac >= thresholds[i]: # if the fraction of adopted neighbors is greater than or equal to the threshold of node i
                        new_adopted[i] = 1 # adopt node i
                        changed = True
            #-- end of one iteration

            # update the adopted nodes after checking all nodes
            adopted = new_adopted
            history.append(sum(adopted.values()) / len(G)) #add the total adopted this timestep to the history
            if not changed:
                break # if no node is adopted in this iteration, break the loop (no more adoptions so we can stop)

        return adopted, np.array(history)


    # This function simulates the threshold dynamics on a graph G
    # The function takes as input the graph G, a list of seed nodes (seeds), a dictionary of threshold values (thresholds), and the maximum number of steps (max_steps)
    # The function returns a dictionary of adopted nodes (adopted) and a numpy array of the fraction of adopted nodes over time (history)
    def threshold_dynamics(G, seeds, thresholds, max_steps=100):
        adopted = {n: 0 for n in G}
        for s in seeds:
            adopted[s] = 1 # initially, all seed nodes are adopted

        history = [sum(adopted.values()) / len(G)] #the history will store the fraction of adopted nodes over time, initially it is the fraction of adopted seed nodes

        for _ in range(max_steps): # iterate max_steps times
            new_adopted = adopted.copy()
            changed = False
            for i in G: # iterate over all nodes i in the graph G
                if adopted[i] == 0: # if node i is not adopted yet
                    neigh = list(G.neighbors(i))
                    if not neigh: # if node i has no neighbors, skip it
                        continue
                    frac = sum(adopted[j] for j in neigh) / len(neigh) # calculate the fraction of adopted neighbors of node i
                    if frac >= thresholds[i]: # if the fraction of adopted neighbors is greater than or equal to the threshold of node i
                        new_adopted[i] = 1 # adopt node i
                        changed = True
            #-- end of one iteration

            # update the adopted nodes after checking all nodes
            adopted = new_adopted
            history.append(sum(adopted.values()) / len(G)) #add the total adopted this timestep to the history
            if not changed:
                break # if no node is adopted in this iteration, break the loop (no more adoptions so we can stop)

        return adopted, np.array(history)

    def final_adoption_fraction(adopted):
        return sum(adopted.values()) / len(adopted)

    # function to calculate the time to reach a target fraction of adopted nodes
    def time_to_fraction(history, target=0.5):
        idx = np.where(history >= target)[0]
        return float(idx[0]) if len(idx) else np.nan
    # ## 7. Seeding Strategies — Heuristics

    # %%

    def random_seeds(G, B):
        return random.sample(list(G.nodes()), B)

    def high_degree_seeds(G, B):
        """
        Select the top-B nodes with the highest degree in graph G.
        """
        # Get all (node, degree) pairs
        degree_list = list(G.degree())

        # Sort nodes by degree in descending order
        sorted_by_degree = sorted(degree_list, key=lambda x: x[1], reverse=True)

        # Take the top-B entries
        top_B = sorted_by_degree[:B]

        # Extract only the node IDs
        result = []
        for n, d in top_B:
            result.append(n)

        return result
    #This code below does the same as for degree. It uses a list comprehension and a lambda function.
    # The lambda function takes a tuple x = (n, b) and returns the second element (b), which is the betweenness centrality of node n.]
    # sorted(bc.items(), key=lambda x: x[1], reverse=True) sorts the items of the dictionary bc.items() by the value of the betweenness centrality (x[1]) in descending order. (Highest first)
    # [:B] takes the first B entries from the sorted list, which are the nodes with the highest betweenness centrality.
    def betweenness_seeds(G, B):
        bc = nx.betweenness_centrality(G) #returns a dictionary that maps each node to its betweenness centrality. {n_1: bc(n_1), n_2: bc(n_2), ...}
        return [n for n, _ in sorted(bc.items(), key=lambda x: x[1], reverse=True)[:B]]


    # This function seeds the network with B nodes with the highest core number, the core number is the number of nodes that are connected to all its neighbors
    # For a given graph G, the core number of a node v is the largest integer k such that v belongs to the k-core of G.
    # In other words:
    # 	•	It tells you how deeply embedded a node is within the dense core structure of the network.
    # 	•	A higher core number means the node is part of a more cohesive and connected region of the graph.
    def kcore_seeds(G, B):
        core = nx.core_number(G) #core_number is a dictionary that maps each node to its core number. 
        return [n for n, _ in sorted(core.items(), key=lambda x: x[1], reverse=True)[:B]]
    # This function estimates the spread of the threshold dynamics on graph G
    # It runs the threshold dynamics R times with different random seeds and thresholds
    # and returns the average final adoption fraction
    def estimate_spread(G, seeds, thresholds, R=5, max_steps=100):
        results = []
        for _ in range(R):
            adopted, _ = Granover_methods.threshold_dynamics(G, seeds, thresholds, max_steps=max_steps)
            results.append(Granover_methods.final_adoption_fraction(adopted))
        return float(np.mean(results))

    def run_experiments(N, k_avg, ensembles=5, runs_per_setting=10, threshold_mode='beta',
                        phi_mean=0.3, phi_std=0.1, B=5, max_steps=200, greedy_R=5):
        import numpy as _np
        import pandas as _pd
        from time import time as _time
        results = []
        t0 = _time()
        for e in range(ensembles): # for each ensemble
            graphs = Granover_methods.build_networks(N, k_avg, seed_base=RANDOM_SEED + e * 10) # build the networks
            for name, G in graphs.items(): # for each network
                thresholds = Granover_methods.draw_thresholds(G, mode=threshold_mode, phi_mean=phi_mean, phi_std=phi_std) # draw the thresholds for each node from the distribution specified by threshold_mode

                strategies = { #dictionary of seed selection strategies, the value is a (lamba) function that returns the seed set for the current strategy
                    'Random': lambda: Granover_methods.random_seeds(G, B),
                    'Degree': lambda: Granover_methods.high_degree_seeds(G, B),
                    'K-core': lambda: Granover_methods.kcore_seeds(G, B),
                    'Betweenness': lambda: Granover_methods.betweenness_seeds(G, B),
                    'Greedy': lambda: Granover_methods.greedy_celf(G, B, thresholds, R=greedy_R, max_steps=max_steps, verbose=False)
                }

                for strat, seed_fn in strategies.items(): # for each seed selection strategy
                    seeds = seed_fn() # get the seed set for the current strategy
                    final_sizes, t50_list = [], []
                    for _ in range(runs_per_setting):
                        adopted, history = Granover_methods.threshold_dynamics(G, seeds, thresholds, max_steps=max_steps)
                        final_sizes.append(Granover_methods.final_adoption_fraction(adopted))
                        t50_list.append(Granover_methods.time_to_fraction(history, target=0.5))
                    results.append({
                        'Ensemble': e,
                        'Network': name,
                        'Strategy': strat,
                        'FinalAdoption': float(_np.mean(final_sizes)),
                        'CascadeProb': float(_np.mean(_np.array(final_sizes) >= 0.5)),
                        'Time50': float(_np.nanmean(t50_list)),
                        'Efficiency': float(_np.mean(final_sizes) / B),
                        'Seeds': seeds
                    })
        t1 = _time()
        print(f"Completed {ensembles} ensembles x 3 networks x 5 strategies in {t1 - t0:.1f}s")
        return _pd.DataFrame(results)

