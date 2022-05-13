import os

import torch
import torch.utils.data as data


class SimpleData(data.Dataset):
    def __init__(self,
                 data_dir,
                 mod="",
                 ):
        # Set all input args as attributes
        # 将输入作为该类的属性
        self.__dict__.update(locals())
        self.data_dir = data_dir
        self.load_data()

    def __len__(self):
        return len(self.examples)

    def load_data(self):
        if self.mod == "train":
            with open(os.path.join(self.data_dir, 'train.txt'), 'r', encoding='utf-8') as fp:
                data = fp.read().strip()
        elif self.mod == "val":
            with open(os.path.join(self.data_dir, 'dev.txt'), 'r', encoding='utf-8') as fp:
                data = fp.read().strip()
        elif self.mod == "test":
            with open(os.path.join(self.data_dir, 'test.txt'), 'r', encoding='utf-8') as fp:
                data = fp.read().strip()
        else:
            raise Exception("请输入正确的mode:[train/val/test]")

        self.examples = []
        for d in data.split('\n'):
            d = d.split('\t')
            self.examples.append((d[0], int(d[1]), len(d[0])))
        return self.examples

    def __getitem__(self, idx):
        return self.examples[idx][0], self.examples[idx][1], self.examples[idx][2]


if __name__ == '__main__':
    simpleData = SimpleData('ref/cnews/', mod="train")
    print(simpleData[0])
    print(simpleData.__dict__)
