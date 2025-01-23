# -*- coding: utf-8 -*-
__author__ = "Chi Xie and Zhao Zhang"
__maintainer__ = "Chi Xie"
import os
from collections import Counter

import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# from pycirclize import Circos
# from Bio.Phylo.BaseTree import Tree
# from Bio import Phylo
# from newick import Node


def plot_hist(data, bins=10, is_norm=False, save_path=None, x=None):
    sns.set_theme(style="whitegrid", font_scale=2.0)
    ax = sns.histplot(data, bins=bins, common_norm=is_norm, kde=False)
    ax.set_xlabel(x)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bars(names, nums, is_sort, save_path=None):
    sns.set(style="whitegrid")

    if is_sort:
        zipped = zip(nums, names)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        result = zip(*sort_zipped)
        nums, names = [list(x) for x in result]

    fontx = {"family": "Times New Roman", "size": 10}
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(16, 4))
    # sns.set_palette("PuBuGn_d")
    sns.barplot(names, nums, palette=sns.cubehelix_palette(80, start=0.5, rot=-0.75))
    fig.autofmt_xdate(rotation=90)
    plt.tick_params(axis="x", labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    plt.tight_layout()
    plt.savefig(save_path)


def generate_wordclouds(sentences, save_dir):
    """Generates word clouds for different parts of speech in a list of sentences.

    Args:
        sentences: A list of sentences.
        save_dir: The directory to save the word cloud images.
    """

    os.makedirs(save_dir, exist_ok=True)
    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")

    # Define the parts of speech to include in the word clouds
    pos_to_include = ["NOUN", "VERB", "ADJ", "ADV"]

    # Process each sentence and collect the relevant words for each part of speech
    words_by_pos = {pos: [] for pos in pos_to_include}
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            if token.pos_ in pos_to_include:
                words_by_pos[token.pos_].append(token.lemma_.lower())

    # Generate a word cloud for each part of speech
    for pos, words in words_by_pos.items():
        if len(words) == 0:
            continue  # skip parts of speech with no words

        # Count the frequency of each word
        word_counts = Counter(words)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="white",
            max_words=200,
            colormap="Set2",
            max_font_size=150,
        ).generate_from_frequencies(word_counts)

        # Save the word cloud image
        filename = f"{pos.lower()}_wordcloud.png"
        filepath = os.path.join(save_dir, filename)
        wordcloud.to_file(filepath)


# def vis_group_tree(data_dict, save_path):

#     # Create 3 randomized trees
#     tree_size_list = [60, 40, 50]
#     trees = [Tree.randomized(string.ascii_uppercase, branch_stdev=0.5) for size in tree_size_list]

#     # Initialize circos sector with 3 randomized tree size
#     sectors = {name: size for name, size in zip(list("ABC"), tree_size_list)}
#     circos = Circos(sectors, space=5)

#     colors = ["tomato", "skyblue", "limegreen"]
#     cmaps = ["bwr", "viridis", "Spectral"]
#     for idx, sector in enumerate(circos.sectors):
#         sector.text(sector.name, r=120, size=12)
#         # Plot randomized tree
#         tree = trees[idx]
#         tree_track = sector.add_track((30, 70))
#         tree_track.axis(fc=colors[idx], alpha=0.2)
#         tree_track.tree(tree, leaf_label_size=3, leaf_label_margin=21)
#         # Plot randomized bar
#         bar_track = sector.add_track((70, 90))
#         x = np.arange(0, int(sector.size)) + 0.5
#         height = np.random.randint(1, 10, int(sector.size))
#         bar_track.bar(x, height, facecolor=colors[idx], ec="grey", lw=0.5, hatch="//")

#     circos.savefig(save_path, dpi=600)

# def clean_newick_key(in_str):
#     bad_chars = [':', ';', ',', '(', ')']
#     for bad_char in bad_chars:
#         in_str = in_str.replace(bad_char, ' ')
#     return in_str

# def build_tree_from_dict(data):
#     root = Node()  # create the root node
#     for key, value in data.items():
#         node = Node(name=clean_newick_key(key)) # name doesn't need to be cleaned
#         if value is not None:
#             child_node = build_tree_from_dict(value)
#             node.add_descendant(child_node)
#         root.add_descendant(node)

#     return root


def replace_chars_in_dict_keys(d):
    """
    Replaces the characters ':', ';', ',', '(', and ')' in the keys of a nested dictionary with '_'.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = replace_chars_in_dict_keys(v)
        new_key = k.translate(str.maketrans(":;,()", "_____"))
        new_dict[new_key] = v
    return new_dict


def build_newick_tree(tree_dict):
    newick_tree = ""
    if isinstance(tree_dict, dict):
        for key, value in tree_dict.items():
            if isinstance(value, dict):
                subtree = build_newick_tree(value)
                if subtree:
                    newick_tree += "(" + subtree + ")" + key + ","
                else:
                    newick_tree += key + ","
            else:
                newick_tree += key + ":" + str(value) + ","
        newick_tree = newick_tree.rstrip(",") + ")"
        return newick_tree
    else:
        return None


# def vis_group_tree(data_dict, save_path):
#     data_dic = replace_chars_in_dict_keys(data_dict)
#     super_group_names = data_dict.keys()

#     # Create 3 randomized trees
#     tree_size_list = [60, 40, 50]
#     trees = [Phylo.read(StringIO(build_newick_tree(data_dict[super_group_name])), "newick") for super_group_name in super_group_names]

#     # Initialize circos sector with 3 randomized tree size
#     sectors = {name: size for name, size in zip(list("ABC"), tree_size_list)}
#     circos = Circos(sectors, space=5)

#     colors = ["tomato", "skyblue", "limegreen"]
#     cmaps = ["bwr", "viridis", "Spectral"]
#     for idx, sector in enumerate(circos.sectors):
#         sector.text(sector.name, r=120, size=12)
#         # Plot randomized tree
#         tree = trees[idx]
#         tree_track = sector.add_track((30, 70))
#         tree_track.axis(fc=colors[idx], alpha=0.2)
#         tree_track.tree(tree, leaf_label_size=3, leaf_label_margin=21)
#         # Plot randomized bar
#         bar_track = sector.add_track((70, 90))
#         x = np.arange(0, int(sector.size)) + 0.5
#         height = np.random.randint(1, 10, int(sector.size))
#         bar_track.bar(x, height, facecolor=colors[idx], ec="grey", lw=0.5, hatch="//")

#     circos.savefig(save_path, dpi=600)
