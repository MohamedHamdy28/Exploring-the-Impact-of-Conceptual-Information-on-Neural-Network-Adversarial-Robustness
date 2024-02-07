import torch.nn as nn
import torch.optim as optim
import torch
from config import LR_DECAY_SIZE

class SequentialTrainer:
    def __init__(self, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        
    def concepts_loss(self, predicted, y_true):
        total_loss = 0
        for i in range(0, predicted.shape[1], 2):
            loss = self.criterion(predicted[:, i:i+2], y_true[:, int(i/2)].long())
            total_loss += loss

        return total_loss

    def train_g(self, train_loader, n_epochs):
        self.model.g_model.train()
        best_loss = 100000
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels, attributes = data
                # attributes = torch.tensor(attributes, dtype=torch.float32)
                attributes = torch.stack(attributes, dim=1)
                # print(attributes.shape)
                # print(attributes, attributes.shape)
                inputs, labels, attributes = inputs.to(self.device), labels.to(self.device), attributes.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model.g_model(inputs)
                outputs = torch.stack(outputs, dim=1).squeeze(2)
                # print(outputs.shape, attributes.shape)
                # print(outputs, attributes)
                loss = self.concepts_loss(outputs, attributes)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            test_loss = self.test_g(train_loader)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(self.model.g_model.state_dict(), '../../models/Experiment 4 cub/g_model.pth')

            self.scheduler.step()
        print('Finished Training')

    def test_g(self, test_loader):
        self.model.g_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels, attributes = data
                attributes = torch.stack(attributes, dim=1)
                images, labels, attributes = images.to(self.device), labels.to(self.device), attributes.to(self.device)
                outputs = self.model.g_model(images)
                outputs = torch.stack(outputs, dim=1).squeeze(2)
                loss = self.concepts_loss(outputs, attributes)
                total += labels.size(0)
                correct += (loss == 0).sum().item()
        print('Accuracy of the g model on the test images: %d %%' % (100 * correct / total))
        return loss

    def train_f(self, train_loader, n_epochs):
        self.model.f_model.train()
        self.model.g_model.eval()
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                images, labels, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                inputs = self.model.g_model(images)
                outputs = self.model.f_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            self.scheduler.step()
        print('Finished Training')
    
    def test_f(self, test_loader):
        self.model.f_model.eval()
        self.model.g_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                inputs = self.model.g_model(images)
                outputs = self.model.f_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the f model on the test images: %d %%' % (100 * correct / total))

    def train(self, train_loader, n_epochs):
        self.train_g(train_loader, n_epochs)
        self.train_f(train_loader, n_epochs)
