import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class g(nn.Module):
    """The network g consists of 2 convolutional
    layers with 32 channels each, along with a maxpool
    layer in between followed by a fully connected
    layer."""

    def __init__(self, n_concepts):
        super(g, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10+n_concepts*2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        non_overlapping, overlapping = x[:,:10], x[:,10:]
        return non_overlapping, overlapping
    

class f(nn.Module):
    def __init__(self, input_size) -> None:
        super(f, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Sequential(nn.Module):
    def __init__(self, n_concepts):
        super(Sequential, self).__init__()
        self.g_model = g(n_concepts).to(device)
        self.f_model = f(10+n_concepts*2).to(device)

        # Defining the training parameters for the concepts model g
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=1e-4)
        self.g_criterion = nn.CrossEntropyLoss()

        # Defining the training parameters for the prediction model f
        self.learned_g = False
        self.f_optimizer = torch.optim.Adam(self.f_model.parameters(), lr=1e-4)
        self.f_criterion = nn.CrossEntropyLoss()
        self.name = 'sequential'

    def split_concepts(self, concepts):
        return concepts[:,:10], concepts[:,10:]

    def loss_overlapping(self, overlapping, y_true):
        total_loss = 0
        for i in range(0, overlapping.shape[1], 2):
            loss = self.g_criterion(overlapping[:, i:i+2], y_true[:, int(i/2)].long())
            total_loss += loss

        return total_loss

    def train_g(self, train_loader, epochs):
        self.g_model.train()
        for epoch in range(epochs):
            for images, labels, concepts in train_loader:
                images, labels, concepts = images.to(device), labels.to(device), concepts.to(device)
                y_true_non_overlapping, y_true_overlapping = self.split_concepts(concepts)

                self.g_optimizer.zero_grad()
                non_overlapping, overlapping = self.g_model(images)

                # Calculate the loss for non_overlaaping concepts
                loss_non_overlapping = self.g_criterion(non_overlapping, y_true_non_overlapping.argmax(dim=1))

                # Calculate the loss for overlapping concepts
                loss_overlapping = self.loss_overlapping(overlapping, y_true_overlapping)
                
                # Calculate the total loss
                loss = loss_non_overlapping + loss_overlapping
                loss.backward()
                self.g_optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.3f}")
        self.learned_g = True

    def calc_acc_non_overlapping(self, test_loader):
        self.g_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, _, concepts in test_loader:
                images, concepts = images.to(device), concepts.to(device)
                y_true_non_overlapping, _ = self.split_concepts(concepts)
                non_overlapping, _ = self.g_model(images)
                y_pred_non_overlapping = non_overlapping.argmax(dim=1)
                correct += (y_pred_non_overlapping == y_true_non_overlapping.argmax(dim=1)).sum().item()
                total += y_true_non_overlapping.size(0)

        return correct / total if total > 0 else 0

    def calc_acc_overlapping(self, test_loader):
        self.g_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, _, concepts in test_loader:
                images, concepts = images.to(device), concepts.to(device)
                _, y_true_overlapping = self.split_concepts(concepts)
                _, overlapping = self.g_model(images)

                for i in range(0, overlapping.shape[1], 2):
                    y_pred_overlapping = overlapping[:, i:i+2].argmax(dim=1)
                    correct += (y_pred_overlapping == y_true_overlapping[:, int(i/2)]).sum().item()
                    total += y_true_overlapping.size(0)

        return correct / total if total > 0 else 0

    def calc_acc_g(self, test_loader):
        self.g_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, _, concepts in test_loader:
                images, concepts = images.to(device), concepts.to(device)
                y_true_non_overlapping, y_true_overlapping = self.split_concepts(concepts)
                non_overlapping, overlapping = self.g_model(images)

                y_pred_non_overlapping = non_overlapping.argmax(dim=1)
                correct += (y_pred_non_overlapping == y_true_non_overlapping.argmax(dim=1)).sum().item()
                total += y_true_non_overlapping.size(0)

                for i in range(0, overlapping.shape[1], 2):
                    y_pred_overlapping = overlapping[:, i:i+2].argmax(dim=1)
                    correct += (y_pred_overlapping == y_true_overlapping[:, int(i/2)]).sum().item()
                    total += y_true_overlapping.size(0)

        return correct / total if total > 0 else 0
    
    def train_f(self, train_loader, epochs):
        if not self.learned_g:
            raise "You have to train g before training f in sequential training"
        self.f_model.train()
        for epoch in range(epochs):
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                self.f_optimizer.zero_grad()

                non_overlapping, overlapping = self.g_model(images)
                # Concatenate the concepts
                overlapping_concat = torch.cat([non_overlapping, overlapping], dim=1)
                y_pred = self.f_model(overlapping_concat)
                loss = self.f_criterion(y_pred, labels.argmax(dim=1))
                loss.backward()
                self.f_optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.3f}")

    def calc_acc_prediction(self, test_loader, delta=None):
        accuracies = []
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            if delta is not None:
                y_pred = self.forward(images + delta[:images.shape[0]]).argmax(dim=1)
            else:
                y_pred = self.forward(images).argmax(dim=1)
            acc = (y_pred == labels.argmax(dim=1)).float().mean().item()
            accuracies.append(acc)
        return sum(accuracies)/len(accuracies)

    def train_model(self, train_loader, epochs=20):
        self.train_g(train_loader, epochs)
        self.train_f(train_loader, epochs)

    def forward(self, x):
        non_overlapping, overlapping = self.g_model(x)
        # Concatenate the concepts
        overlapping = torch.cat([non_overlapping, overlapping], dim=1)
        return self.f_model(overlapping)

    def save_g(self):
        torch.save(self.g_model.state_dict(), "g_model1.pth")

    def load_g(self, path):
        self.g_model.load_state_dict(torch.load(path))
        self.learned_g = True

    def save_f(self):
        torch.save(self.f_model.state_dict(), "f_model1.pth")

    def load_f(self, path):
        self.f_model.load_state_dict(torch.load(path))


class Joint(nn.Module):
    def __init__(self, n_concepts):
        super(Joint, self).__init__()
        self.g_model = g(n_concepts).to(device)
        self.f_model = f(10+n_concepts*2).to(device)

        self.name = 'joint'

    def forward(self, x):
        non_overlapping, overlapping = self.g_model(x)
        overlapping_concat = torch.cat([non_overlapping, overlapping], dim=1)
        y_pred = self.f_model(overlapping_concat)
        return y_pred, non_overlapping, overlapping


    def loss_overlapping(self, overlapping, y_true):
        total_loss = 0
        concept_idx = 0
        # print(overlapping.shape)
        for i in range(0, overlapping.shape[1], 2):
            loss = nn.CrossEntropyLoss()(overlapping[:, i:i+2], y_true[:, concept_idx].long())
            concept_idx += 1
            total_loss += loss

        return total_loss



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10)

        self.name = 'CNN'

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
