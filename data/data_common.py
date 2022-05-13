import torch
import numpy as np



class Collater:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, examples):
        input_ids, label_ids, lengths = [], [], []
        for i, example in enumerate(examples):
            token_ids = [self.kwargs['token2id'].get(i, 1) for i in example[0]]
            input_ids.append(token_ids)
            label_ids.append(self.to_one_hot(example[1]))
            lengths.append(example[2])
        input_ids, label_ids, lengths = sort_batch(sequence_padding(input_ids), label_ids, lengths)
        return torch.tensor(input_ids).long(), torch.tensor(label_ids), lengths

    def to_one_hot(self, idx):
        out = np.zeros(self.kwargs['class_num'], dtype=float)
        out[idx] = 1
        return out


def sort_batch(data, label, length):
    """用于将一个batch里面的数据按照长度排列"""
    # 先将数据转化为numpy()，再得到排序的index
    inx = np.argsort(length)[::-1].copy()

    data = np.array(data)[inx]
    label = np.array(label)[inx]
    length = np.array(length)[inx]
    # length转化为了list格式，不再使用torch.Tensor格式
    length = list(length)
    return (data, label, length)


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


if __name__ == '__main__':
    token2id = {'你': 0}
    collator = Collater(token2id=token2id)
    print(collator.kwargs)
