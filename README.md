# pytorch_lightning_text_classification
基于pytorch_lightning的中文文本分类样例。

# 目录说明

```
data：存放数据及数据加载模块
----ref：存放数据
----data_common.py：存放公用的数据处理模块
----data_interface.py：pytorch_lightning构建数据加载器
----simple_data.py：pytorch构建dataset
model：存放模型的模块
----model_common.py：公用的模块
----model_interface.py：pytorch_lightning构建训练器
----simple_model.py：pytorch构建的普通的模型
main.py：运行主函数，包含训练、验证、测试和预测
utils.py：运行主函数的辅助模块
lightning_logs：存放训练好的模型和日志
----version_{num}：存放版本为num的模型和日志
--------checkpoint：存放模型
--------events.out.features.xxx：tensorboadx可视化所需日志
--------hyparams.yaml：存储定义好的一些参数
```

# 使用说明

以下是我在基于https://link.zhihu.com/?target=https%3A//github.com/miracleyoo/pytorch-lightning-template 进行修改为文本分类中使用的一些理解。

- 这里没有指定使用的gpu，使用的是cpu训练的，怎么使用gpu可参考最后贴的那个知乎链接。

- data下的simple_data.py和里面的SimpleData类是下划线和驼峰一致的，在main.py中指定dataset为simple_data，在data_interface.py中会调用instancialize()获取构建的数据的dataset，也就是SimpleData。
- model下的simple_model和里面的SimpleModel类是下划线和驼峰一致的，在main.py中指定model_name为simple_model，在model_interface.py中会调用load_model()获取构建的数据的model，也就是SimpleModel。
- 在simple_dataset.py里面构建的是普通的pytorch格式的dataset，如果需要在dataloader中传入collate_fn，可以在data_common.py里面查看样例。构建为dataloader是在data_interface.py里面。主要是setup()，会根据trainer.fit()、trainer.test()、trainer.predict()来判断是哪一步，然后分别生成不同的dataset。
- 在model_interface.py中，我们需要关注的是在step或者epoch进行的操作。比如validation_step()，计算每一批次的指标，需要注意的是使用self.log()的时候指定on_step=False, on_epoch=True，那么最终计算的是每一个epoch的指标，这里你可以自己定义一个test_epoch_end()然后去比对看是不是这样。

- main.py里面定义了do_train，do_test，do_predict。分别修改后就可以运行使用了。不过最好是先do_train，得到保存的模型后修改main函数里面需要加载的模型的位置，然后再进行测试和预测。

# 参考

- [Pytorch Lightning 完全攻略 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/353985363)
- https://link.zhihu.com/?target=https%3A//github.com/miracleyoo/pytorch-lightning-template

