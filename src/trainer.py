import torch
import torch.nn as nn

class Sequential_Trainer:
    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self, training_data, n_epochs):
        self.model.train_model(training_data, n_epochs)

    def test(self, test_loader):
        return self.model.calc_acc_prediction(test_loader)
    
class Joint_Trainer:
    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.pred_loss_weight, self.concepts_loss_weight = 0.5, 1

    def split_concepts(self, concepts):
        return concepts[:,:10], concepts[:,10:]
    
    def calc_loss(self, model, y_pred, labels, non_overlapping, overlapping, y_true_non_overlapping, y_true_overlapping):
        
        # Calculate the loss for non_overlaaping concepts
        loss_non_overlapping = self.criterion(non_overlapping, y_true_non_overlapping.argmax(dim=1))
        # Calculate the loss for overlapping concepts
        # print(overlapping.shape, y_true_overlapping.shape)
        loss_overlapping = model.loss_overlapping(overlapping, y_true_overlapping)
        # Calculate the loss for concepts
        concept_loss = loss_non_overlapping + loss_overlapping

        # Calculate the loss for prediction
        pred_loss = self.criterion(y_pred, labels.argmax(dim=1))

        loss = self.pred_loss_weight * pred_loss + self.concepts_loss_weight * concept_loss
        return loss
    
    def train(self, train_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            for images, labels, concepts in train_loader:
                images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
                self.optimizer.zero_grad()
                y_true_non_overlapping, y_true_overlapping = self.split_concepts(concepts)
                y_pred, non_overlapping, overlapping = self.model(images)

                loss = self.calc_loss(self.model, y_pred, labels, non_overlapping, overlapping, y_true_non_overlapping, y_true_overlapping)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.3f}")

    def test(self, test_loader):
        self.model.eval()
        accuracies = []
        for images, labels, concepts in test_loader:
            images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
            y_pred, _, _ = self.model(images)
            acc = (y_pred.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
            accuracies.append(acc)
        return sum(accuracies)/len(accuracies)
    
class CNN_Trainer:
    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            for images, labels, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(images)
                loss = self.criterion(y_pred, labels.argmax(dim=1))
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.3f}")

    def test(self, test_loader):
        self.model.eval()
        accuracies = []
        for images, labels, _ in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            y_pred = self.model(images)
            acc = (y_pred.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
            accuracies.append(acc)
        return sum(accuracies)/len(accuracies)