"""Network centrality with quantum walks
=====================================

Networks: which nodes matter? Quantum-walk centrality versus PageRank and degree.

A small collaboration network (illustrative). Continuous-time quantum walk centrality (the long-time
average of the walk's occupation) is compared with PageRank and degree; Spearman rank correlations
show where they agree.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/q.png'
import numpy as np
from scipy.stats import spearmanr
from qlcog.quantum import adjacency, ctqw_probabilities, quantum_walk_centrality, pagerank, degree_centrality

# %%
# A small network and three centralities
# --------------------------------------
edges = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7), (7, 8), (7, 9), (8, 9), (2, 10), (10, 11)]
A = adjacency(edges, 12)
qw, pr, dg = quantum_walk_centrality(A), pagerank(A), degree_centrality(A)
print('node  quantum-walk  PageRank  degree')
for i in np.argsort(-qw):
    print('%4d  %12.3f  %8.3f  %6.3f' % (i, qw[i], pr[i], dg[i]))

# %%
# How well do the rankings agree?
# -------------------------------
print('Spearman rank correlation: quantum walk vs PageRank %.2f, vs degree %.2f'
      % (spearmanr(qw, pr)[0], spearmanr(qw, dg)[0]))
print('walk from node 0 at t = 1, 2, 4:', [ctqw_probabilities(A, t, start=0).round(2).tolist()[:4] for t in (1, 2, 4)])
