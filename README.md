估计很多开发员使用tesseract做中文识别，但是结果不是一般的差，譬如下面的图片

![alt text](https://github.com/JinpengLI/deep_ocr/blob/master/test_data.png "需要识别文本")


```
$ tesseract -l chi_sim test_data.png out_test_data
```

```
看到恨多公司在招腭大改癫和机器字习胸人 v 我有3个建议 (T) 忧T ' 2个上t较靠遭
胸人就譬了 v不是越多越好 (2) 这T '2个人要能给大蒙上踝'倩邂知L目 (3) 不要招
不宣代四胸人:虹大改癫和机器字习胸v不裹目宣 (或者宣过) 大量代四v基本上就
只会忽悠了
```

其实现在做文字识别不是很难，特别基于深度学习，这里是这个项目的reco_chars.py脚本，基于caffe的识别效果，是不是好很多？而且代码比tesseract短很多。

```
$ python reco_chars.py
```

```
看很多公苘在招聘天数据和机器学习人我有个建议找个较靠谱的人就够了不是越多越好这个人要给大家上课传递知识不要招不写代码的人做天数据机器学习的不亲写或者写过天且代码基本上就只会忽悠了
```


项目需要下载已训练好的模型， reco_chars.py源代码中的 base_dir = "/root/data/chongdata_caffe_cn_sim_digits_64_64"。

```
https://pan.baidu.com/s/1qYld5N2
```

大家可以基于caffe训练自己的字体，系统基于这个文章开发单个字的识别：

```
Deep Convolutional Network for Handwritten Chinese Character Recognition

http://cs231n.stanford.edu/reports/zyh_project.pdf
```
