import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import pickle
from constants import *


def train_test_split(data):
    np.random.seed(42)
    # Set user id and skill name as index to the data frame
    data = data.set_index(['user_id', 'skill_name'])
    print("Dataframe:", data)
    print("*" * 25)
    # Use permutation to change the order in the data frame
    # Order of records in which nodes are presented should not affect the final output of the network
    idx = np.random.permutation(data.index.unique())
    # Split train and test portions
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_val = data.loc[test_idx].reset_index()
    print("Train data shape", data_train.shape)
    print("*" * 25)
    print("Validation data shape", data_val.shape)
    print("*" * 25)
    return data_train, data_val


def preprocess_data(data):
    """
    Pre-process data and pad to the maximum length.
    """
    features = ['skill_id', 'correct']
    # Group the data based on user id where each user group is a list of features (skill id and correct answer)
    seqs = data.groupby(['user_id']).apply(lambda x: x[features].values.tolist())
    # Get the length of sequences and limit their length based on block size defined in constants
    length = min(max(seqs.str.len()), block_size)
    # Pad short sequences with -100 to ensure sequences' lengths consistency
    seqs = seqs.apply(lambda s: s[:length] + (length - min(len(s), length)) * [[-1000] * len(features)])
    print("Max sequence length:", max(seqs))
    print("*" * 25)
    return seqs


def create_skill_graph(df):
    """
    Construct the skill graph
    :param df:
    :return:
    """
    # Load the graph and its dictionary if they exist
    if os.path.exists('graph/skill_graph.pickle'):
        print("Using existing skill graph...")
        print("*" * 25)
        return pickle.load(open('graph/skill_graph.pickle', 'rb')), pickle.load(open('graph/skill_dict.pickle', 'rb'))
    # Otherwise construct the graph
    print("Constructing skill graph...")
    print("*" * 25)
    # Ignore rows without skill name
    df = df[~df['skill_name'].isna()]
    # Group skill name based on user ide in a list
    grouped = df.groupby('user_id')['skill_name'].agg(list)
    # Get unique skill names in a distinct list
    uniques = list(df['skill_name'].unique())
    # Creat a dictionary of skill name with initial zeros to the dictionary values
    skill_cooccurs = {skill_name: np.zeros(df['skill_name'].nunique()) for skill_name in uniques}

    for seq in tqdm(grouped.values):
        cooccur = np.zeros(df['skill_name'].nunique())
        for s in reversed(seq):
            # Increment skill name based on its appearance
            cooccur[uniques.index(s)] += 1
            # Update the dictionary values based on cooccur
            skill_cooccurs[s] = skill_cooccurs[s] + cooccur
    # Make the skill co-occurrence weighted for edge weight in order to represent the weighted relationships
    skill_cooccurs = {k: (v / sum(v)).round(1) for k, v in skill_cooccurs.items()}
    dod = {}
    for i, (skill_name, edges) in enumerate(skill_cooccurs.items()):
        dod[i] = {}
        for j, e in enumerate(edges):
            if e > 0:
                # if the edge weight is larger than zero, then the skill has a neighboring skill with index j
                dod[i][j] = {'weight': e}
    # Construct the graph using networkx from the dictionary of dictionaries
    skill_graph = nx.from_dict_of_dicts(dod)
    skill_dict = dict(zip(uniques, range(len(uniques))))
    pickle.dump(skill_graph, open('graph/skill_graph.pickle', 'wb'))
    pickle.dump(skill_dict, open('graph/skill_dict.pickle', 'wb'))
    print(skill_graph)
    print("*" * 25)
    print(skill_dict)
    print("*" * 25)
    return skill_graph, skill_dict


def preprocess(data):
    def train_test_split(data, skill_list=None):
        np.random.seed(42)
        data = data.set_index(['user_id', 'skill_name'])
        idx = np.random.permutation(data.index.unique())
        train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
        data_train = data.loc[train_idx].reset_index()
        data_val = data.loc[test_idx].reset_index()
        return data_train, data_val

    if 'skill_name' not in data.columns:
        data.rename(columns={'skill_id': 'skill_name'}, inplace=True)
    if 'original' in data.columns:
        data = data[data['original'] == 1]
    # Filter out the data frame by ignoring missing or special skills
    data = data[~data['skill_name'].isna() & (data['skill_name'] != 'Special Null Skill')]
    multi_col = 'template_id' if 'template_id' in data.columns else 'Problem Name'
    # Split train and validation data
    data_train, data_val = train_test_split(data)
    print("Train-test split finished...")
    print("*" * 25)
    # Invoke create skill function to build the graph for the data
    skill_graph, skill_dict = create_skill_graph(data_train)
    print("Imputing skills...")
    print("*" * 25)
    # Replace skills with ids to map replacements with dictionary keys
    repl = skill_dict[data_train['skill_name'].value_counts().index[0]]
    for skill_name in set(data_val['skill_name'].unique()) - set(skill_dict):
        skill_dict[skill_name] = repl
    print("Replacing skills...")
    print("*" * 25)
    # Return the processed train and validation data
    data_train['skill_id'] = data_train['skill_name'].apply(lambda s: skill_dict[s])
    data_val['skill_id'] = data_val['skill_name'].apply(lambda s: skill_dict[s])

    return data_train, data_val, skill_graph
