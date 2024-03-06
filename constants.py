# Read KT dataset using Pandas.
import pandas as pd

# Set dataset path
data = pd.read_csv("data/as.csv", encoding='latin', low_memory=False)

# GNN model hypermarkets
num_epochs = 20
batch_size = 16
block_size = 1024

# Embedding dimensions for vector space representation
skill_embd_dim = 128

# Train split percentage
train_split = 0.8
