# Importing Libraries
import torch
import numpy as np

# DEFAULTS
TRAIN_PARAMS = {
    "epochs": 10,
    "loss_function": torch.nn.BCEWithLogitsLoss,
    "learning_rate": 1e-3,
    "post_activation": torch.nn.Sigmoid(),
}


# Helper Class
class BBTCAV:

    def __init__(self, model, train_params, device):
        self.device = device
        self.train_params = TRAIN_PARAMS
        self.initModel(model, train_params)

    def train(self, trainloader, testloader=None):
        self.model.train()
        criterion = self.train_params["loss_function"]()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_params["learning_rate"]
        )
        for epoch in range(1, self.train_params["epochs"] + 1):
            print(f"Working on epoch: {epoch}/{self.train_params['epochs']}")
            running_loss = 0.0
            correct = 0
            total = 0
            for imgs, labels in trainloader:
                optimizer.zero_grad()
                imgs, labels = imgs.to(self.device), labels.to(self.device).unsqueeze(1).float()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if(self.post_activation):
                    predicted = (torch.sigmoid(outputs) > 0.5) * 1.
                else:
                    predicted = (outputs > 0.5) * 1.
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_loss = running_loss / total
            train_accuracy = correct / total
            if epoch % 1 == 0:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
                if testloader != None:
                    self.test(testloader)
                    self.model.train()
        print("\nModel training completed")

    def test(self, testloader):
        test_loss = 0.0
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).unsqueeze(1).float()
                outputs = self.model(inputs)
                criterion = self.train_params["loss_function"]()
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                if(self.post_activation):
                    predicted = (torch.sigmoid(outputs) > 0.5) * 1.
                else:
                    predicted = (outputs > 0.5) * 1.
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss /= total
        test_accuracy = correct / total
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_accuracy:.4f}\n")

    def getScores(self, imgs: torch.tensor):
        self.model.eval()
        with torch.no_grad():
            scores = self.model(imgs)
            if self.post_activation:
                scores = self.post_activation(scores)
        return scores

    def initModel(self, model: torch.nn.Module, train_params: dict = None):
        self.model = model
        self.model.to(self.device)
        for key, value in train_params.items():
            self.train_params[key] = value
        self.post_activation = train_params.get("post_activation", False)
        if (self.post_activation == False) and (
            self.train_params["loss_function"] == torch.nn.BCEWithLogitsLoss
        ):
            self.post_activation = torch.nn.Sigmoid()

    def calcAttribution(self, preds, scores):
        # Simply applying dy/dx on concept vs prediction.
        # grads = torch.gradient(preds, spacing = (scores,)) # torch implementation is different
        grads = np.gradient(preds, scores, edge_order=2)
        return grads


# Testing
if __name__ == "__main__":
    pass
