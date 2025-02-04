from torchvision import models
from torch import nn

# Model mappings
model_mapping = {
    "densenet121": (
        models.densenet121,
        {"weights": models.DenseNet121_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet161": (
        models.densenet161,
        {"weights": models.DenseNet161_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet169": (
        models.densenet169,
        {"weights": models.DenseNet169_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet201": (
        models.densenet201,
        {"weights": models.DenseNet201_Weights.DEFAULT, "family": "densenet"},
    ),
    "resnet50": (
        models.resnet50,
        {"weights": models.ResNet50_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "resnet101": (
        models.resnet101,
        {"weights": models.ResNet101_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "resnet152": (
        models.resnet152,
        {"weights": models.ResNet152_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
    "vit-b-16": (
        models.vit_b_16,
        {"weights": models.ViT_B_16_Weights.DEFAULT, "family": "vit"},
    ),
    "vit-b-32": (
        models.vit_b_32,
        {"weights": models.ViT_B_32_Weights.DEFAULT, "family": "vit"},
    ),
    "convnext-b": (
        models.convnext_base,
        {"weights": models.ConvNeXt_Base_Weights.DEFAULT, "family": "convnext"},
    ),
    "swin-t": (
        models.swin_t,
        {"weights": models.Swin_T_Weights.DEFAULT, "family": "swin"},
    ),
    # Add more models as needed with their respective configurations.
}

class Model(nn.Module):
    """Model definition."""

    def __init__(self, model_name: str, num_classes: int):
        """
        Initialize Model instance.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes.
        """
        super(Model, self).__init__()

        # Get model class and configuration
        model_class, model_config = model_mapping[model_name]
        self.model = model_class(weights=model_config["weights"])

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the in_features from the model family
        in_features = self._get_in_features(model_config["family"])

        # Adjust the classifier according to the model family
        if model_config["family"] == "densenet":
            self.model.classifier = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "resnet":
            self.model.fc = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "vit":
            self.model.heads = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "convnext":
            self.model.classifier[2] = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "swin":
            self.model.head = self._create_classifier(in_features, num_classes)


    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _get_in_features(self, family: str) -> int:
        """Return the number of input features for the classifier."""
        if family == "densenet":
            return self.model.classifier.in_features
        elif family == "resnet":
            return self.model.fc.in_features
        elif family == "vit":
            return self.model.heads.head.in_features
        elif family == "convnext":
            return self.model.classifier[2].in_features
        elif family == "swin":
            return self.model.head.in_features
        else:
            raise ValueError(f"Unknown model family: {family}")

    def _create_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        """Create the classifier module."""
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1), # 0.5
            nn.Linear(in_features // 2, num_classes),
        )


class ModelFactory:
    """
    Factory for creating different models based on their names.
    """

    def __init__(self, model_name: str, num_classes: int):
        """
        Initialize ModelFactory instance.

        Args:
            model_name (str): The name of the model.
            num_classes (int): The number of output classes.
        """
        self.model_name = model_name
        self.num_classes = num_classes

    def __call__(self):
        """
        Create a model instance based on the provided name.

        Returns:
            Model: An instance of the selected model.
        """
        if self.model_name not in model_mapping:
            valid_options = ", ".join(model_mapping.keys())
            raise ValueError(
                f"Invalid model name: '{self.model_name}'. Available options: {valid_options}"
            )

        return Model(self.model_name, self.num_classes)

    def get_model(self):
        """Returns the created model directly."""
        return self()


if __name__ == "__main__":
    model_factory = ModelFactory("swin-t", 3)
    model = model_factory.get_model()

    print(model)
