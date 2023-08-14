from .model import *

__all__ = ["create_models"]

MODELS = {
    "mnistcnn": FedAvgNetMNIST,
    "cifarcnn": FedAvgNetCIFAR,
    "tinycnn": FedAvgNetTiny,
    "vgg11": vgg.vgg11,
    "res18": resnet.resnet20,
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cinic10": 10,
    "tinyimagenet": 200,

}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)

    return model
