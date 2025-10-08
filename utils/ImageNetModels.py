import enum
import torchvision
from torchvision import transforms


def rescale(x):
    return x / 255.0


resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
DEFAULT_TRANSFORM = [
    transforms.CenterCrop(224),
    transforms.functional.pil_to_tensor,
    rescale,
    normalize,
]

MODEL_NAMES = [
    "vit_b_16",
    "swin_t",
    "resnet18",
    "vgg16",
    "mobilenet_v3_large",
    "mobilenet_v2",
    "squeezenet1_1",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "vit_b_32",
    "swin_s",
    "squeezenet1_0",
    "maxvit_t",
]

TRANSFORM_DICT = {
    model: transforms.Compose([resize, *DEFAULT_TRANSFORM]) for model in MODEL_NAMES
}

TRANSFORM_DICT["swin_t"] = transforms.Compose(
    [
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC),
        *DEFAULT_TRANSFORM,
    ]
)

TRANSFORM_DICT["mobilenet_v3_large"] = transforms.Compose(
    [
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC),
        *DEFAULT_TRANSFORM,
    ]
)

TRANSFORM_DICT["swin_s"] = transforms.Compose(
    [
        transforms.Resize(246, interpolation=transforms.InterpolationMode.BICUBIC),
        *DEFAULT_TRANSFORM,
    ]
)

TRANSFORM_DICT["maxvit_t"] = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        *DEFAULT_TRANSFORM,
    ]
)


def getWeights(model_name: str) -> enum.EnumMeta:
    weight_name = model_name + "_weights"
    for item in dir(torchvision.models):
        if item.lower() == weight_name.lower():
            return getattr(getattr(torchvision.models, item), "IMAGENET1K_V1")
    raise ValueError(f"Model Name: {model_name} might be invalid, weights not found.")


def loadImageNetModels(model_name: str):
    if not (model_name in MODEL_NAMES):
        raise ValueError(f"Invalid Value given for Model name: {model_name}")
    weights = getWeights(model_name)
    model = getattr(torchvision.models, model_name)(weights=weights)
    transforms = TRANSFORM_DICT[model_name]
    return model, transforms
