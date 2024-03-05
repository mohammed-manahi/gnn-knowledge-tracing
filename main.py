from sklearn.metrics import *
import torch.optim as optim
from torch_geometric.utils.convert import to_networkx, from_networkx
import itertools
from tqdm import tqdm
from gcn_model import *
from graphsage_model import *
from gat_model import *
from transformer import *
from preprocess import *
from constants import *
from visualizer import visualize_metrics
from test import predict_skills

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        user_idx = raw_data['user_id'].sample(batch_size).unique() if not val else user_ids[
                                                                                   _ * (batch_size // 2): (_ + 1) * (
                                                                                           batch_size // 2)]
        filtered_data = raw_data[raw_data['user_id'].isin(user_idx)].sort_values(['user_id', 'order_id'])
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        # Next token prediction.
        X = torch.tensor(batch[:, :-1, ..., :], requires_grad=True, dtype=torch.float32).cuda()
        y = torch.tensor(batch[:, 1:, ..., [0, 1]], requires_grad=True, dtype=torch.float32).cuda()
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
        # Same code as for training forward pass
        mask = y[..., -1] != -1000
        all_skill_embd = skill_net(torch.arange(110).cuda(), skill_graph.edge_index.cuda(),
                                   skill_graph.weight.cuda().float())
        skill_embd = all_skill_embd[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
        ohe = torch.eye(110).cuda()[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
        # Account for baseline and remove node embedding if so.
        if baseline:
            feat = [X, ohe]
        else:
            feat = [X, skill_embd, ohe]
        X = torch.cat(feat, dim=-1)
        corrects = model.forward(X, y[..., 0])[mask]
        y = y[..., -1].unsqueeze(-1)[mask]
        # Squeeze and put into list.
        ypred.append(corrects.ravel().detach().cpu().numpy())
        ytrue.append(y.ravel().detach().cpu().numpy())
    ypred = np.concatenate(ypred)
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
            output = model(torch.cat(feat, dim=-1), skill_idx=y[..., 0].detach()).ravel()

            # Compute loss and mask padded values.
            mask = (y[..., -1] != -1000).ravel()
            loss = F.binary_cross_entropy(output[mask], y[..., -1:].ravel()[mask])

            # Backpropagate and take a gradient step.
            loss.backward()
            optimizer.step()

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
            torch.save(model.state_dict(), f"model/model-{skill_net.tag}-{epoch}-{auc}-{acc}-{rmse}.pth")
            if not baseline:
                torch.save(skill_net.state_dict(), f"model/skill_net-{skill_net.tag}-{epoch}-{auc}-{acc}-{rmse}.pth")


def init_graphkt(graph_net):
    data_train, data_test = train_test_split(data)
    # Initialize and train model with GCN, GraphSAGE and GAT.
    data_train, data_val, skill_graph, model, skill_net, optimizer = init_models(data_train, graph_net=graph_net)
    print(f"Training {skill_net.tag} model!")
    return data_train, data_val, skill_graph, model, skill_net, optimizer


if __name__ == "__main__":
    data_train, data_val, skill_graph, model, skill_net, optimizer = init_graphkt(GCN)
    train(model, skill_net, data_train, data_val, num_epochs)
    visualize_metrics(skill_net, epoch_list, acc_list, auc_list, rmse_list)
    #predict_skills(model, skill_net)
