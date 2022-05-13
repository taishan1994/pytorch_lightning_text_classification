import inspect
import importlib

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .data_common import Collater


class DataInterface(pl.LightningDataModule):

    def __init__(self, num_workers=2,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()
        self.collate = Collater(token2id=kwargs['token2id'],
                                class_num=kwargs['class_num'])

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(mod="train")
            self.valset = self.instancialize(mod="val")

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(mod="test")


        if stage == 'predict' or stage is None:
            self.predictset = self.instancialize(mod="test")
        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.

        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.collate)

    def predict_dataloader(self):
        return DataLoader(self.predictset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.collate)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        # 这里在main.py里面定义了dataset参数，simple_data
        # 然后会转换为SimpleData类，也就是找到了我们加载数据的Dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])

        try:
            self.data_module = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        print(args1)
        return self.data_module(**args1)

