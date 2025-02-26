"""File containing unittests for DDP WDS dataloading."""

# built-in libs
import unittest
import warnings

# external libs
import jax
import jax.numpy as jnp
import ml_collections

# deps
from data import utils
from data import wds_imagenet_dataset as wds


# suppress resource warning
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)
    return do_test


class TestWDSDDP(unittest.TestCase):
    """Test naive DDP implementation via `split_by_node`."""

    def setUp(self):
        jax.process_index()
        jax.process_count()
        self.batch_size = 64
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': self.batch_size,  # <- simulate 2048 // 32
                    'num_workers': 1,
                }
            }
        )
        self.world_size = 4
        self.dataset_len = 1_281_167
        
    @ignore_warnings
    def test_multi_proc_loader(self):

        total_num_samples = 0
        num_incomplete_batch = 0
        for rank in range(self.world_size):
            dataset = wds.build_imagenet_dataset(
                is_train=True,
                data_dir="/mnt/disks/data/imagenet_wds",
                image_size=256,
                debug=True,
                world_size=self.world_size,
                rank=rank
            )

            loader = wds.build_imagenet_loader(
                self.config, dataset, use_torch=True
            )
            
            print(f"Start loading on process {rank}...")
            for i, batch in enumerate(loader):
                sample, label = batch
                if sample.shape[0] != self.batch_size:
                    num_incomplete_batch += 1
                    continue
                total_num_samples += sample.shape[0]
            
            print(f"Total loaded samples: ", total_num_samples)
            print(f"Dropped batches: ", num_incomplete_batch)


if __name__ == "__main__":

    unittest.main()
