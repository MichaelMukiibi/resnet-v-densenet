from datasets import load_dataset
import numpy as np

class HFDatasetIterator:
    def __init__(self, hf_dataset_split, batch_size, shuffle=False):
        self.ds = hf_dataset_split.to_iterable_dataset()
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def as_numpy_iterator(self):
        ds = self.ds
        if self.shuffle:
            ds = ds.shuffle(seed=42, buffer_size=10000)
        
        for batch in ds.iter(batch_size=self.batch_size, drop_last_batch=True):
            images = np.array(batch['img'], dtype=np.float32) / 255.0
            
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
            images = (images - mean) / std
            
            yield {
                'image': images,
                'label': np.array(batch['label'], dtype=np.int32)
            }

def get_cifar10_datasets(batch_size: int = 64):
    ds = load_dataset("cifar10", trust_remote_code=True)
    
    train_ds = HFDatasetIterator(ds['train'], batch_size, shuffle=True)
    test_ds = HFDatasetIterator(ds['test'], batch_size, shuffle=False)
    
    return train_ds, test_ds