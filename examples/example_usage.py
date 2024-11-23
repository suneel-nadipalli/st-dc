import sys, torch

import numpy as np

sys.path.append("..")

from st_dc import viz

sentences_dict = {
    "bank": [
        "I deposited money in the bank.",
        "The river bank was full of lush vegetation.",
    ],
    "pool": [
        "I had a good swim at the pool.",
        "The questions will be drawn from the pool of available resources.",
    ],
    "figure": [
        "He was an important father figure in her life.",
        "The amount stolen was a very large figure.",
    ],
    "work": [
        "I like this beautiful work by Andy Warhol.",
        "Hundreds of people work in this building.",
    ],
}

focus_word = "pool"

sentences = sentences_dict[focus_word]

viz(focus_word, sentences, dim_technique="pca", num_neighbors=5, plot_type="3D")