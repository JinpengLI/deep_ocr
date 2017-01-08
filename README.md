
deep ocr
--------


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
看很多公苘在招聘天数据和机器学习人我有个建议找个较靠谱
的人就够了不是越多越好这个人要给大家上课传递知识不要招
不写代码的人做天数据机器学习的不亲写或者写过天且代码基本上就
只会忽悠了
```

大家可以基于caffe训练自己的字体，系统基于这个文章开发单个字的识别：

```
Deep Convolutional Network for Handwritten Chinese Character Recognition

http://cs231n.stanford.edu/reports/zyh_project.pdf
```

通过 Docker 安装
------------------------

先安装docker，以下教程在Ubuntu 14.04 通过测试

```
https://www.docker.com/
```

下载deep_ocr_workspace.zip (https://pan.baidu.com/s/1nvz2wrB 和 https://pan.baidu.com/s/1qYPKH3Y )

两个文件的md5sum值，用于校验文件是否成功下载。

```
$ md5sum deep_ocr_workspace.zip
ffeda7ea6604e7b8835c05a33fa0459e  deep_ocr_workspace.zip
$ md5sum deep_ocr_workspace.z01
ea66796c2bbdb2bec9b7ee28eb44012d  deep_ocr_workspace.z01
```

解压到本地硬盘，譬如到以下地方 (~/deep_ocr_workspace)

```
cat deep_ocr_workspace.z* > unsplit_deep_ocr_workspace.zip
unzip unsplit_deep_ocr_workspace.zip -d ~/
```

这个zip包含deep_ocr所有需要数据文件（由于太大了，所以放百度云了）。所有数据到解压到 `~/deep_ocr_workspace`，你也可以把需要处理的数据放到这个文件夹。

基于cpu
=======

```
docker pull jinpengli/deep_ocr_cpu_docker:latest
```

启动 docker container

```
docker run -ti --volume=${HOME}/deep_ocr_workspace:/workspace jinpengli/deep_ocr_cpu_docker:latest /bin/bash
cd /opt/deep_ocr
git pull origin master
```

volume用于mount到container里面，这样可以获取上面的识别结果。

```
python /opt/deep_ocr/reco_chars.py
```

然后可以继续你们的开发。。。。加油。。。

身份证识别
========

暂时不是很稳定，需要加一些语义模型。等等吧。。。。

识别图片

![识别图片](https://github.com/JinpengLI/deep_ocr/raw/master/data/id_card_img.jpg)


执行命令

```
export WORKSPACE=/workspace
deep_ocr_id_card_reco --img $DEEP_OCR_ROOT/data/id_card_img.jpg             --debug_path /tmp/debug             --cls_sim ${WORKSPACE}/data/chongdata_caffe_cn_sim_digits_64_64             --cls_ua ${WORKSPACE}/data/chongdata_train_ualpha_digits_64_64
```

识别结果：

```
...
ocr res:
============================================================
name
韦小宝
============================================================
address
北京市东城区累山前街4号
紫禁城敬事房
============================================================
month
12
============================================================
minzu
汉
============================================================
year
1654
============================================================
sex
男
============================================================
id
1X21441114X221243X
============================================================
day
20

```
