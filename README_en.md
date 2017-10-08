
deep ocr
--------

This installation has been usually test on Ubuntu 16.04.


```
git clone https://github.com/JinpengLI/deep_ocr.git ~/deep_ocr
virtualenv ~/deep_ocr_env
source ~/deep_ocr_env/bin/activate
pip install -r ~/deep_ocr/requirements.txt
cd ~/deep_ocr && python setup.py install
```

For a simple recognition test, you can launch the below command.

```
source ~/deep_ocr_env/bin/activate && cd ~/deep_ocr && ./bin/deep_ocr_reco data/holiday_notification.jpg -v -d
```
