import torch
import typer
from mlops_template.preprocess_data import corrupt_mnist
from mlops_template.model import MyAwesomeModel
from loguru import logger

logger.add("logs/evaluate.log", level="DEBUG", rotation="100 MB")

DEVICE = torch.device(
    "cpu"
)  # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """
    Evaluate a trained model.

    Args:
        model_checkpoint (str): Path to the model checkpoint.
    """
    logger.info("Evaluating like my life depended on it")
    logger.info(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)

    model.load_state_dict(
        torch.load(model_checkpoint, map_location="cpu")
    )  # model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    logger.info(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()
