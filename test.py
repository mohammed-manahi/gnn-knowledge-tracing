import torch
import constants
from transformer import GPT
from gcn_model import GCN
from preprocess import preprocess

# Load trained model and skill network
# model = GPT.from_pretrained("model/model-GCN-19-0.8145085740265459-0.7750665581200906-0.39473408460617065.pth")
#
# data_train, data_val, skill_graph = preprocess(constants.data)
# n_skills = skill_graph.number_of_nodes()
# skill_net = GCN(n_skills, constants.skill_embd_dim)
#
# def predict_skills(model, skill_net):
#   while True:
#     print("Enter a sequence of skills (separated by spaces) or 'quit' to exit:")
#     skill_sequence = input("> ").lower()
#     if skill_sequence == "quit":
#       break
#     skill_tensor = torch.tensor([skill_ids]).cuda()
#     all_skill_embd = skill_net(torch.arange(110).cuda(), skill_graph.edge_index.cuda(), skill_graph.weight.cuda().float())
#     skill_embd = all_skill_embd[skill_tensor.long()]
#     prediction = model(input_data).argmax(-1).item()
#     predicted_skill = skill_dict[prediction]
#     print(f"Predicted Next Skill: {predicted_skill}")

