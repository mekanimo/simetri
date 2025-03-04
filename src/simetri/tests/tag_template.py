from string import Template
from itertools import product


# italic, bold, new_family, tex_family, no_family, tex_size, num_size, no_size
decision_matrix = [True, True, ]

print(len(list(product([True, False], repeat=6))))