
import torch

class Attacker:
    def __init__(self, batch_size) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.example = torch.zeros((batch_size, 1, 28, 28), requires_grad=True).to(self.device)


    def pgd_linf_targ(self, model, data_loader, epsilon, alpha, num_iter, y_targ):
        """
        Training on the whole test set
        """
        delta = torch.zeros_like(self.example, requires_grad=True)
        for t in range(num_iter):
            print(f"Running iteration {t}")
            for X, y, concepts in data_loader:
                X, y, concepts = X.to(self.device), y.to(self.device), concepts.to(self.device)
                if model.name == "joint":
                    yp, _, _ = model(X + delta[:X.shape[0]])
                else:
                    yp = model(X + delta[:X.shape[0]])
                loss = 2*yp[:, y_targ].sum() - yp.sum()
                loss.backward()

                delta = delta + alpha * delta.grad.detach().sign()
                delta = delta.clamp(-epsilon, epsilon).detach().requires_grad_(True)

                # Clear gradients after updating delta
                if delta.grad is not None:
                    delta.grad.zero_()
        return delta.detach()
    
    def test_addv(self, model, test_loader, delta):
        model.eval()
        accuracies = []
        for images, labels, concepts in test_loader:
            images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
            if model.name == "joint":
                y_pred, _, _ = model(images + delta[:images.shape[0]])
            else:
                y_pred = model(images + delta[:images.shape[0]])
            acc = (y_pred.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
            accuracies.append(acc)
        return sum(accuracies)/len(accuracies)