from dataset import load_cub_datasets
from config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from trainer import SequentialTrainer
from cub_models import Sequential, Joint, Standard
from torch import nn
import torch 
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_attributes_weights(train_loader, n_concepts):
    # Initialize a dictionary to store the count of each attribute's values
    concept_counts = defaultdict(lambda: defaultdict(int))

    for _, _, attributes in train_loader:
        attributes = torch.stack(attributes, dim=1)
        for idx, attribute in enumerate(attributes.t()):
            unique, counts = attribute.unique(return_counts=True)
            for u, c in zip(unique, counts):
                concept_counts[idx][u.item()] += c.item()

    # Calculate weights for each attribute's values
    weights_dict = {}
    for attr_idx, counts in concept_counts.items():
        total_counts = sum(counts.values())
        weights = {val: total_counts / count for val, count in counts.items()}
        # Normalize weights so that min weight is 1.0
        min_weight = min(weights.values())
        weights = {val: weight / min_weight for val, weight in weights.items()}
        weights_dict[attr_idx] = torch.tensor(list(weights.values()), dtype=torch.float32).to(device)

    return weights_dict

n_concepts = 112
batch_size = 64
train_loader = load_cub_datasets('train', n_concepts=n_concepts, batch_size=batch_size)
validation_loader = load_cub_datasets('val', n_concepts=n_concepts, batch_size=batch_size)
test_loader = load_cub_datasets('test', n_concepts=n_concepts, batch_size=batch_size)


model = Sequential(n_concepts=n_concepts, num_classes=N_CLASSES)
# model.load_g('../../models/Experiment 4 cub/g_model.pth')
attributes_weights = calculate_attributes_weights(train_loader, n_concepts)
# print(attributes_weights)

sequential_trainer = SequentialTrainer(model, attributes_weights)
sequential_trainer.train_g(train_loader, validation_loader, n_epochs=20)
sequential_trainer.test_g(test_loader)

for image, class_labe, attributes in train_loader:
    print(class_labe, len(attributes))
    # model = Sequential(n_concepts=2, num_classes=N_CLASSES)
    # model = Joint(n_concepts=2, num_classes=N_CLASSES)
    model = Standard(num_classes=N_CLASSES)
    output = model(image)
    print(output)
    break

# postive_att = 0
# negative_att = 0
# total = 0
# for image, class_label, attributes in train_loader:
#     for atts in attributes:
#         for att in atts:
#             # print(att)
#             if att == 1:
#                 postive_att += 1
#             else:
#                 negative_att += 1
#             total += 1
# print(postive_att, negative_att, total)
# print(postive_att/total*100, negative_att/total*100)