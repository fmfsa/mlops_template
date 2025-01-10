import torch
import typer
from hydra import initialize, compose
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, RocCurveDisplay

import wandb

from mlops_template.preprocess_data import corrupt_mnist
from mlops_template.model import MyAwesomeModel
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
DEVICE = torch.device(
    "cpu"
)  # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()


@app.command()
def train() -> None:
    """
    Train a model on MNIST.

    Args:
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
    """
    with initialize(config_path="../../configs"):
        # hydra.main() decorator was not used since it was conflicting with typer decorator
        config = compose(config_name="config.yaml")

    hparams = config.train
    lr, batch_size, epochs = hparams["lr"], hparams["batch_size"], hparams["epochs"]

    wandb.init(
        project="my_awesome_project",
        config=OmegaConf.to_container(hparams, resolve=True, throw_on_missing=True),
    )
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})

    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    torch.save(model.state_dict(), "models/model.pth")
    artifact = wandb.Artifact(
        name="my_awesome_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("models/model.pth")
    wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    app()
