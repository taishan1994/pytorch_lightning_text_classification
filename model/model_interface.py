import importlib
import inspect
import torch
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from torch.nn import functional as F


class ModelInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kwargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, x, length):
        return self.model(x, length)

    def training_step(self, batch, batch_idx):
        train_x, train_y, length = batch
        out, out_length = self.forward(train_x, length)
        loss = self.loss_function(out, train_y)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        train_x, train_y, length = batch
        out, out_length = self.forward(train_x, length)
        loss = self.loss_function(out, train_y)
        label_digit = train_y.argmax(axis=1)
        out_digit = out.argmax(axis=1)
        correct_num = sum(label_digit.view(-1) == out_digit.view(-1)).cpu().item()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num / len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)
        return (correct_num, loss, len(out_digit))

    """这里会将每一个step的结果存到list中
    在step里面指定了on_step=False,on_epoch=True就不需要这个了
    def test_epoch_end(self, outputs):

        crooret = sum([i[0] for i in outputs])
        total = sum([i[2] for i in outputs])
        loss = sum([i[1] for i in outputs]) / len(outputs)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('correct_num', crooret, on_step=False, on_epoch=True, prog_bar=True)
        self.log('total_num', total, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', crooret / total,
                 on_step=False, on_epoch=True, prog_bar=True)
    """

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, length = batch
        out, out_length = self(x, length)
        out = out.argmax(1).detach().cpu()
        y = y.argmax(1).detach().cpu()
        return (out, y)

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == "ce":
            self.loss_function = F.cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
