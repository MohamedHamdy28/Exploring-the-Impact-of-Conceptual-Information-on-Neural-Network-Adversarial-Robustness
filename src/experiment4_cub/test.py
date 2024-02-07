from dataset import load_cub_datasets
from config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from trainer import SequentialTrainer
from models import Sequential, Joint, Standard
from torch import nn

n_concepts = 50
train_loader = load_cub_datasets('train', n_concepts=n_concepts, batch_size=32)
validation_loader = load_cub_datasets('val', n_concepts=n_concepts, batch_size=32)
test_loader = load_cub_datasets('test', n_concepts=n_concepts, batch_size=32)

# for image, class_labe, attributes in train_loader:
#     print(class_labe, len(attributes))
#     # model = Sequential(n_concepts=2, num_classes=N_CLASSES)
#     # model = Joint(n_concepts=2, num_classes=N_CLASSES)
#     model = Standard(num_classes=N_CLASSES)
#     output = model(image)
#     print(output)
#     break

model = Sequential(n_concepts=2, num_classes=N_CLASSES)
sequential_trainer = SequentialTrainer(model)
sequential_trainer.train_g(train_loader, n_epochs=10)
sequential_trainer.test_g(test_loader)