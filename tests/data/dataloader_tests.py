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
                    'seed': 0,
                    'seed_pt': 0,
                }
            }
        )

    def test_loader_in64(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/imagenet/prepared/imagenet_64",
            image_size=64,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        batch = utils.parse_batch(batch)
        self.assertEqual(batch['images'].shape, (jax.local_device_count(), 16 // jax.local_device_count(), 64, 64, 3))
        self.assertEqual(batch['labels'].shape, (jax.local_device_count(), 16 // jax.local_device_count()))
        
        pixel_max = jnp.max(batch['images'])
        pixel_min = jnp.min(batch['images'])

        # default normalization is to [-1, 1]
        print(pixel_max, pixel_min)
        self.assertTrue(pixel_max == 255)
        self.assertTrue(pixel_min == 0)

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )

    def test_loader_in256(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/imagenet/prepared/imagenet_256",
            image_size=256,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        batch = utils.parse_batch(batch)
        self.assertEqual(batch['images'].shape, (jax.local_device_count(), 16 // jax.local_device_count(), 32, 32, 8))
        self.assertEqual(batch['labels'].shape, (jax.local_device_count(), 16 // jax.local_device_count()))

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )
    
    def test_loader_in512(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/imagenet/prepared/imagenet_512",
            image_size=512,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        batch = utils.parse_batch(batch)
        self.assertEqual(batch['images'].shape, (jax.local_device_count(), 16 // jax.local_device_count(), 64, 64, 8))
        self.assertEqual(batch['labels'].shape, (jax.local_device_count(), 16 // jax.local_device_count()))

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )

if __name__ == "__main__":

    unittest.main()
