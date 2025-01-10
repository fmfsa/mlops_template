import torch
from omegaconf import OmegaConf
from torch import nn
import hydra
from loguru import logger

logger.add("logs/model.log", level="DEBUG", rotation="100 MB")


class MyAwesomeModel(nn.Module):
    """This model is a simple convolutional neural network for image classification."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(config) -> None:
    hparams = config.model

    logger.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    torch.manual_seed(hparams["seed"])

    model = MyAwesomeModel()
    logger.info(f"Model architecture: {model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
