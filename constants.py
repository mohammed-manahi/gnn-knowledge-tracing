# Read KT dataset using Pandas.
import pandas as pd

data = pd.read_csv("data/as.csv", encoding='latin', low_memory=False)

# Global hyperparameters for KT model.
num_epochs = 20
batch_size = 16
block_size = 1024

# Graph network hyperparameters
skill_embd_dim = 128

# Train split percentage
train_split = 0.8
