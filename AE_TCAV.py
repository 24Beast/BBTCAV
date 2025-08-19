# Importing Libraries
import torch
import numpy as np

# DEFAULTS
TRAIN_PARAMS = {
    "epochs": 15,
    "recon_loss_function": torch.nn.MSELoss,
    "cls_loss_function": torch.nn.BCEWithLogitsLoss,
    "learning_rate": 1e-3,
    "alpha": 0.1,
}


# Helper Class
class AETCAV:

    def __init__(self, model, train_params, device):
        self.device = device
        self.train_params = TRAIN_PARAMS
        self.initModel(model, train_params)

    def train(self, trainloader, testloader=None):
        print("Training Auxiliary Model")
        self.model.train()
        recon_criterion = self.train_params["recon_loss_function"]()
        cls_criterion = self.train_params["cls_loss_function"]()
        alpha = self.train_params["alpha"]
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_params["learning_rate"]
        )
        for epoch in range(1, self.train_params["epochs"] + 1):
            print(f"Working on epoch: {epoch}/{self.train_params['epochs']}")
            total_recon, total_cls = 0.0, 0.0
            num = 0
            for imgs, labels in trainloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)
                recon, preds, _ = self.model(imgs)
                loss_recon = recon_criterion(recon, imgs)
                labels = labels.float()
                loss_cls = cls_criterion(preds, labels)
                loss = (1 - alpha) * loss_recon + alpha * loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_recon += loss_recon.item()
                total_cls += loss_cls.item()
                num += 1

            total_recon = total_recon / num
            total_cls = total_cls / num

            if epoch % 1 == 0:
                print(
                    f"Train Recon Loss: {total_recon:.4f} | Train Class loss: {total_cls:.4f}"
                )
                if testloader != None:
                    self.test(testloader)
                    self.model.train()
        print("\nModel training completed")

    def test(self, testloader):
        self.model.eval()
        recon_criterion = self.train_params["recon_loss_function"]()
        cls_criterion = self.train_params["cls_loss_function"]()
        total_recon, total_cls = 0.0, 0.0
        num = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)

                recon, preds, _ = self.model(imgs)
                loss_recon = recon_criterion(recon, imgs)
                labels = labels.float()
                loss_cls = cls_criterion(preds, labels)

                total_recon += loss_recon.item()
                total_cls += loss_cls.item()
                num += 1
        total_recon = total_recon / num
        total_cls = total_cls / num

        print(
            f"Test Recon Loss: {total_recon:.4f} | Test Class Loss: {total_cls:.4f}\n"
        )

    def initModel(self, model: torch.nn.Module, train_params: dict = None):
        self.model = model
        self.model.to(self.device)
        for key, value in train_params.items():
            self.train_params[key] = value

    def getAttribution(self, pred_model, imgs, concept_num, class_num, eps=0.01):
        recon, c_preds, z = self.model(imgs)
        c_vector = self.model.classifier.classifier.weight[concept_num]
        c_vector = c_vector / torch.norm(c_vector)  # converting to unit vector
        z_new = z + (eps * c_vector)
        imgs_new = self.model.decoder(z_new)
        preds = self.getPreds(pred_model, recon, class_num)
        new_preds = self.getPreds(pred_model, imgs_new, class_num)
        grads = (new_preds - preds) / eps
        return grads

    def getPreds(self, pred_model, imgs, class_num=None):
        preds = pred_model(imgs)
        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
        if class_num == None:
            return preds
        return preds[:, class_num]


# Testing
if __name__ == "__main__":
    import random
    from torchvision import transforms
    from utils.data import CelebAConcept
    from torch.utils.data import DataLoader
    from utils.models import SimpleCNN, AutoEncoderWithClassifier

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    MODEL_PATH = "models/celebA_CNN.pth"
    DATA_DIR = "../Datasets/CelebA/"
    TRAIN_PARAMS_NEW = {
        "epochs": 25,
        "recon_loss_function": torch.nn.MSELoss,
        "cls_loss_function": torch.nn.BCEWithLogitsLoss,
        "learning_rate": 1e-3,
        "alpha": 0.01,
    }
    B_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VALS_PATH = "./vals/"
    LATENT_DIMS = 1024
    #    CONCEPTS = ["Age", "Gender", "Skin", "Bald"]

    print("Loading Data")
    concept_attr = "Gender"
    target_attr = "Attractive"
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    train_concept_data = CelebAConcept(
        data_dir=DATA_DIR,
        split="train",
        target=target_attr,
        concept=concept_attr,
        mode="concept",
        transform=transform,
        concept_num=2500,
    )
    train_concept_loader = DataLoader(train_concept_data, batch_size=B_SIZE)
    val_concept_data = CelebAConcept(
        data_dir=DATA_DIR,
        split="valid",
        target=target_attr,
        concept=concept_attr,
        mode="concept",
        transform=transform,
        concept_num=1000,
    )
    val_concept_loader = DataLoader(val_concept_data, batch_size=B_SIZE)

    print("Preparing for Training.")
    model = AutoEncoderWithClassifier(latent_dim=LATENT_DIMS, num_classes=1, H=64, W=64)
    checkpoint = torch.load(MODEL_PATH)
    main_model = SimpleCNN().to(DEVICE)
    main_model.load_state_dict(checkpoint["model_state_dict"])
    interpreter = AETCAV(model, TRAIN_PARAMS_NEW, DEVICE)
    interpreter.train(train_concept_loader, val_concept_loader)

    print("Getting Attribution")
    grads = torch.zeros(len(val_concept_data))
    start = 0
    with torch.no_grad():
        main_model.eval()
        for i, (imgs, concepts) in enumerate(val_concept_loader, 1):
            l = len(imgs)
            imgs = imgs.to(DEVICE)
            curr_grads = interpreter.getAttribution(main_model, imgs, 0, 0, eps=1)
            grads[start : start + l] = curr_grads
            start += l
    print(f"{concept_attr=},{len(grads)=},{grads.mean()=},{grads.std()/(len(grads)**0.5)=}")
