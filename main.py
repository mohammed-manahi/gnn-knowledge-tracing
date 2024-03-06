from sklearn.metrics import *
import torch.optim as optim
from torch_geometric.utils.convert import to_networkx, from_networkx
import itertools
from tqdm import tqdm

import constants
from gcn_model import *
from graphsage_model import *
from gat_model import *
from transformer import *
from preprocess import *
from constants import *
from visualizer import visualize_metrics

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Use these lists for plotting metrics
epoch_list = []
auc_list = []
acc_list = []
rmse_list = []


def construct_batches(raw_data, epoch=0, val=False):
    """
    Construct batches based on tabular KT data with user_id, skill_id, and
    correctness. Pads to the minimum of the maximum sequence length and the
    block size of the transformer.
    """
    np.random.seed(epoch)
    user_ids = raw_data['user_id'].unique()

    # Loop until one epoch of training.
    for _ in range(len(user_ids) // batch_size):
        # Randomize data samples based on unique user ids divided by the batch size defined in the constants
        user_idx = raw_data['user_id'].sample(batch_size).unique() if not val else user_ids[
                                                                                   _ * (batch_size // 2): (_ + 1) * (
                                                                                           batch_size // 2)]
        # Filter data by user id and sort them based on order and user ids
        filtered_data = raw_data[raw_data['user_id'].isin(user_idx)].sort_values(['user_id', 'order_id'])
        # Invoke process data function to pre-process filtered data
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        # The purpose of the model to predict the skill required to solve the problem
        # Therefore X here refers to the skill sequences for predicting the next skill
        # X excepts the last sequence where the prediction should occur based on learning representations
        X = torch.tensor(batch[:, :-1, ..., :], requires_grad=True, dtype=torch.float32).cuda()
        print("Input", X)
        print("*" * 25)
        # Y is the target prediction label
        y = torch.tensor(batch[:, 1:, ..., [0, 1]], requires_grad=True, dtype=torch.float32).cuda()
        print("Target", y)
        print("*" * 25)
        for i in range(X.shape[1] // block_size + 1):
            if X[:, i * block_size: (i + 1) * block_size].shape[1] > 0:
                yield [X[:, i * block_size: (i + 1) * block_size], y[:, i * block_size: (i + 1) * block_size]]


def evaluate(model, batches, baseline=False):
    """
    Evaluates the GraphKT or baseline model on a given set of batches and returns
    the outputted correctness probability predictions.

    Arguments:
      - model: causal transformer (Transformer)
      - batches: evaluation batches (generator from construct_batches function)
      - baseline: whether this is a baseline model (bool)

    Returns:
      - ypred: predicted probabilities (np.ndarray)
      - ytrue: ground truth correctness (np.ndarray)
    """
    ypred, ytrue = [], []
    for X, y in batches:
        # Iterate through skill sequences for each batch
        # Filter out padded values '-1000'
        mask = y[..., -1] != -1000
        # Get initial embeddings from the graph
        all_skill_embd = skill_net(torch.arange(110).cuda(), skill_graph.edge_index.cuda(),
                                   skill_graph.weight.cuda().float())
        print("Shape of all_skill_embd:", all_skill_embd.shape)
        print("*" * 25)
        skill_embd = all_skill_embd[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
        print("Shape of skill_embd after filtering:", skill_embd.shape)
        print("*" * 25)
        # One hot-encoder for skill representations
        ohe = torch.eye(110).cuda()[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
        # Account for baseline and remove node embedding if so.
        if baseline:
            feat = [X, ohe]
        else:
            feat = [X, skill_embd, ohe]
        print("Shape of concatenated features:", torch.cat(feat, dim=-1).shape)
        print("*" * 25)
        # Concatenate features in a tensor for input
        X = torch.cat(feat, dim=-1)
        # Invoke the forward pass for the GNN model
        corrects = model.forward(X, y[..., 0])[mask]
        y = y[..., -1].unsqueeze(-1)[mask]
        # Squeeze and put into list.
        ypred.append(corrects.ravel().detach().cpu().numpy())
        ytrue.append(y.ravel().detach().cpu().numpy())
    # Concatenate target prediction
    ypred = np.concatenate(ypred)
    # Concatenate target true values
    ytrue = np.concatenate(ytrue)
    return ypred, ytrue


def init_models(data, skill_embd_dim=skill_embd_dim, graph_net=GAT, baseline=False):
    """
    Pre-process data, generate the skill graph, and initialize the model
    and optimizer.
    """
    # 1. Pre-process data by removing sequences of length 1 and undefined skills.
    # 2. Split data into training and validation set.
    # 3. Generate and load skill graph based on
    data_train, data_val, skill_graph = preprocess(data)
    n_skills = skill_graph.number_of_nodes()

    # Transformer configuration for KT model.
    config = GPTConfig(vocab_size=n_skills, block_size=block_size,
                       n_layer=2, n_head=8, n_embd=128,
                       input_size=2 + n_skills + skill_embd_dim * (1 - baseline),
                       bkt=False)
    model = GPT(config).cuda()

    # Convert skill_graph to torch_geometric graph and instantiate graph network.
    skill_graph = from_networkx(skill_graph)
    skill_net = graph_net(n_skills, skill_embd_dim).cuda()
    print("Total Parameters:", sum(p.numel() for p in model.parameters()) +
          sum(p.numel() for p in skill_net.parameters()))
    print("*" * 25)

    # Optimize all parameters end-to-end.
    optimizer = optim.AdamW(itertools.chain(model.parameters(), skill_net.parameters()), lr=1e-4)
    return data_train, data_val, skill_graph, model, skill_net, optimizer


def train(model, skill_net, data_train, data_val, num_epochs, baseline=False):
    """
    Train the KT transformer and GNN end-to-end by optimizing the KT binary
    cross-entropy objective.

    Arguments:
      - model (transformer for KT)
      - skill_net (GNN for skill embeddings)
      - data_train (training data)
      - data_val (validation data)
      - num_epochs (number of training epochs)
    """
    for epoch in range(num_epochs):
        # Train model for num_epochs epochs.
        model.train()
        skill_net.train()
        batches_train = construct_batches(data_train, epoch=epoch)
        pbar = tqdm(batches_train)
        losses = []

        for X, y in pbar:
            optimizer.zero_grad()

            # Get node embeddings for all skills from skill_net (GNN).
            all_skill_embd = skill_net(torch.arange(110).cuda(),
                                       skill_graph.edge_index.cuda(),
                                       skill_graph.weight.cuda().float())

            # Select node embeddings corresponding to skill tagged with data.
            skill_embd = all_skill_embd[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
            ohe = torch.eye(110).cuda()[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]

            # Concatenate data with skill embedding and one-hot encoding of skill.
            if baseline:
                feat = [X, ohe]
            else:
                feat = [X, skill_embd, ohe]
            print("Shape of concatenated features (x):", torch.cat(feat, dim=-1).shape)
            print("*" * 25)
            output = model(torch.cat(feat, dim=-1), skill_idx=y[..., 0].detach()).ravel()

            # Compute loss and mask padded values.
            mask = (y[..., -1] != -1000).ravel()
            loss = F.binary_cross_entropy(output[mask], y[..., -1:].ravel()[mask])

            # Backpropagate and take a gradient step.
            loss.backward()
            optimizer.step()
            print("Training Loss:", loss.item())
            print("*" * 25)
            # Report the training loss.
            losses.append(loss.item())
            pbar.set_description(f"Training Loss: {np.mean(losses)}")

        if epoch % 1 == 0:
            # Evaluate model using validation set.
            batches_val = construct_batches(data_val, val=True)
            model.eval()
            skill_net.eval()

            # Construct predictions based on current model and compute error(s).
            ypred, ytrue = evaluate(model, batches_val, baseline=baseline)
            auc = roc_auc_score(ytrue, ypred)
            acc = (ytrue == ypred.round()).mean()
            rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))

            # Append metrics
            epoch_list.append(epoch)
            auc_list.append(auc)
            acc_list.append(acc)
            rmse_list.append(rmse)

            # Report error metrics on validation set and save checkpoint.
            print(
                f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {auc}] - [VALIDATION ACC: {acc}] - [VALIDATION RMSE: {rmse}]")
            print("*" * 25)
            torch.save(model.state_dict(), f"model/model-{skill_net.tag}-{epoch}-{auc}-{acc}-{rmse}.pth")
            if not baseline:
                torch.save(skill_net.state_dict(), f"model/skill_net-{skill_net.tag}-{epoch}-{auc}-{acc}-{rmse}.pth")


def init_graphkt(graph_net):
    data_train, data_test = train_test_split(data)
    # Initialize and train model with GCN, GraphSAGE and GAT.
    data_train, data_val, skill_graph, model, skill_net, optimizer = init_models(data_train, graph_net=graph_net)
    print(f"Training {skill_net.tag} model!")
    print("*" * 25)
    return data_train, data_val, skill_graph, model, skill_net, optimizer


if __name__ == "__main__":
    print("Current model configuration:\nNumber of epochs: {0}, Batch Size: {1}, Maximum Sequence Block Size: {2}, "
          "Embedding Dimension: {3}".format({constants.num_epochs}, {constants.batch_size}, {constants.block_size},
                                            {constants.skill_embd_dim}))
    print("*" * 25)
    data_train, data_val, skill_graph, model, skill_net, optimizer = init_graphkt(GCN)
    train(model, skill_net, data_train, data_val, num_epochs)
    visualize_metrics(skill_net, epoch_list, acc_list, auc_list, rmse_list)
    # predict_skills(model, skill_net)
