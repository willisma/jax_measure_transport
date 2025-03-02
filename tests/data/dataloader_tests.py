"""File containing unittests for Local & WDS dataloading."""

# built-in libs
import unittest

# external libs
import jax
import jax.numpy as jnp
import ml_collections

# deps
from data import utils
from data import local_imagenet_dataset as lds
from data import wds_imagenet_dataset as wds


class TestWDS(unittest.TestCase):

    def setUp(self):
        jax.process_index()
        jax.process_count()
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': 16,
                    'num_workers': 8,
                }
            }
        )
        self.dataset = wds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/data/imagenet_wds",
            image_size=256,
        )
        
    
    def test_loader(self):
        loader = wds.build_imagenet_loader(
            self.config, self.dataset
        )
        batch = next(iter(loader))
        batch = utils.parse_batch(batch)
        self.assertEqual(batch['images'].shape, (jax.local_device_count(), 16 // jax.local_device_count(), 256, 256, 3))
        self.assertEqual(batch['labels'].shape, (jax.local_device_count(), 16 // jax.local_device_count()))

        pixel_max = jnp.max(batch['images'])
        pixel_min = jnp.min(batch['images'])

        # default normalization is to [-1, 1]
        self.assertTrue(pixel_max <= 1.0)
        self.assertTrue(pixel_min >= -1.0)

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )


class TestLatentDS(unittest.TestCase):

    def setUp(self):
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': 16,
                    'num_workers': 8,
                }
            }
        )

    def test_loader_in256(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/imagnet/prepared/imagenet_256_sd.zip",
            image_size=256,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        print(batch)

if __name__ == "__main__":

    unittest.main()
