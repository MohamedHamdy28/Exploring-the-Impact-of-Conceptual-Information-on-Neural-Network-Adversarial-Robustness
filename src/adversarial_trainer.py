import torch
import torch.nn as nn
import torch.optim as optim

class AdversarialTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def epoch_adversarial(self, loader, model, attack, opt=None, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for X,y,_ in loader:
            X,y = X.to(self.device), y.to(self.device)
            delta = attack(model, X, y)
            if model.name == "joint":
                yp, _, _ = model(X+delta)
            else:
                yp = model(X+delta)
            loss = nn.CrossEntropyLoss()(yp,y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_err += (yp.max(dim=1)[1] != y.argmax(dim=1)).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)
    
    def pgd_linf_targ(self, model, X,y, epsilon=0.4, alpha=1e-2, num_iter=20, y_targ=2):
        """
        Training on the whole test set
        """
        X, y = X.to(self.device), y.to(self.device)
        delta = torch.zeros_like(X, requires_grad=True)
        for t in range(num_iter):
            if model.name == "joint":
                yp, _, _ = model(X + delta)
            else:        
                yp = model(X + delta)
            loss = 2*yp[:, y_targ].sum() - yp.sum()
            loss.backward()

            delta = delta + alpha * delta.grad.detach().sign()
            delta = delta.clamp(-epsilon, epsilon).detach().requires_grad_(True)

            # Clear gradients after updating delta
            if delta.grad is not None:
                delta.grad.zero_()
        return delta.detach()
    

    def train_model(self, model, num_iter, train_loader, test_loader):
        opt = optim.Adam(model.parameters(), lr=1e-4)
        for t in range(num_iter):
            print(f"Running iteration {t}")
            train_err, train_loss = self.epoch_adversarial(train_loader, model, self.pgd_linf_targ, opt)
            adv_err, adv_loss = self.epoch_adversarial(test_loader, model, self.pgd_linf_targ)
            if t == 4:
                for param_group in opt.param_groups:
                    param_group["lr"] = 1e-5
            print(*("{:.6f}".format(i) for i in (train_err, adv_err)), sep="\t")
        return model
