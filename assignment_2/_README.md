#Assignment 2 - Granovetter threshold diffusion on a Facebook Pages Network.

This directory contains the code and outputs for assignemt 2, where a Granovetter threshold model simulates diffusion through the Facebook Public Figures Pages network from assignment 1. The network contains 11.534 nodes and 66.942 edges. The output is used to analyse how different seeding strategies and seeding fractions influence information spread. This is compared between two different threshold distributions.

#Files overview

Code
- assignment_2_script.ipynb
Main notebook that loads the network, computes centrality measures, runs the threshold model, computes different strategies and different seeding fractions, and generates the plots.
- relevant_functions.py
contains all the relevant methods for the Granovetter model taken from the Granovetter_Threshold_Experiments.ipynb notebook. 

Outputs
- results_df.npy - simulation results for the beta-distributed threshold
- results_const.npy - simulation results for the constant threshold
- column_names.txt - column labels for rearranging the dataframe for reproducibility

Images
- adoption_curves_combined.png
Figure 1a and 1b: Shows the adoption curve under Beta vs Constant thresholds
- final_adoption_combined.png
Figure 3a and 3b: Shows the final adoption
- time_to_50_adoption_combined.png
Figure 2a and 2b: Shows the time to 50% adoption

Data
- fb-pages-public-figure.nodes
- fb-pages-public-figure.edges

Packages needed
- networkx, pandas, networkx, matplotlib.pyplot, numpy, random

How to run
- Ensure all files are in same directory. Run all code in assignment_2_script.ipynb. 

