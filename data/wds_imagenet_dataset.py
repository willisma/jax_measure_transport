"""File containing the WebDataset dataloading pipeline."""

# built-in libs

# external libs
from absl import logging
import jax
import ml_collections
import PIL
import torch
import webdataset as wds

# deps
from data import utils


# Main entry point for imagenet dataset
def build_imagenet_dataset(
    is_train: bool, data_dir: str, image_size: int, file_path: str | None = None, latent_dataset: bool = False, shuffle_buffer: int = 1024
) -> torch.util.data.IterableDataset:
    """Build the WebDataset for ImageNet. Code practice largely follows https://github.com/webdataset/webdataset/blob/main/examples/train-resnet50-multiray-wds.ipynb"""
    if latent_dataset:
        # latent structure is
        # - data_dir/
        #   - dataset.json  <- metadata
        #   - folder1/
        #     - image1.png
        #     ...
        raise NotImplementedError("Latent dataset not supported for WebDataset yet")
    else:
        # wds image structure is
        # - data_dir/
        #   - imagenet1k-train-{index}.tar/
        #     - image1.png
        #     - image1.json
        #     ...
        
        transform = utils.build_transform(image_size)
        def wds_preprocess(sample):
            image, json = sample
            try: 
                # check this path
                label = json['annotations'][0]['category_id']
            except Exception:
                label = 1000
            
            return transform(image), label
        
        def multiproc_splitter(urls):
            proc_id, proc_num = jax.process_index(), jax.process_count()
            return urls[proc_id::proc_num]

        # shard to multiprocesses
        dataset = wds.WebDataset(data_dir, resampled=True, shardshuffle=True, nodesplitter=multiproc_splitter)
        # data decoding
        dataset = dataset.shuffle(shuffle_buffer).decode("pil").to_tuple("png", "json")
        # data preprocessing
        dataset = dataset.map(wds_preprocess)

    logging.info(dataset)
    return dataset


def build_imagenet_loader(
    config: ml_collections.ConfigDict,
    dataset: torch.utils.data.IterableDataset
):
    """Build loader for WebDataset."""
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    
    dataset = dataset.batched(batch_size)
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    loader = loader.unbatched().shuffle(1000).batched(batch_size)

    return loader