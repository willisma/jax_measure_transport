"""File containing util functions for the dataloading."""

# built-in libs
import hashlib

# external libs
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms


class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def anything_to_seed(*args):
    serialized_args = []
    for arg in args:
        if isinstance(arg, int):
            type_code = 'int'
            value_repr = str(arg)
        elif isinstance(arg, float):
            type_code = 'float'
            value_repr = repr(arg)
        elif isinstance(arg, bool):
            type_code = 'bool'
            value_repr = str(arg)
        elif isinstance(arg, str):
            type_code = 'str'
            value_repr = repr(arg)
        else:
            raise TypeError(f"Unsupported type: {type(arg).__name__}")
        serialized_arg = f"{type_code}:{value_repr}"
        serialized_args.append(serialized_arg)

    serialized_str = '|'.join(serialized_args)
    serialized_bytes = serialized_str.encode('utf-8')
    hash_bytes = hashlib.sha256(serialized_bytes).digest()
    seed_int = int.from_bytes(hash_bytes, 'big')
    return seed_int % (1 << 64)

# Aligning with OpenAI ADM
def center_crop_arr(pil_image: PIL.Image, image_size: int):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

def build_transform(image_size: int):
    crop_fn = lambda x: center_crop_arr(x, image_size)
    transform = transforms.Compose([
        crop_fn,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )  # normalizes to [-1, 1]
    return transform
