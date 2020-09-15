# 🚶‍♂️ Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI

The idea of this project is to use MobileNet SSD with Caffe implementation as the person detection algorithm. This program uses OpenCV API for image processing and utilizing the existing DNN module.
</br>

_💻 To run the program on command line:_

```sh
python3 social-distance.py --video [path to input] --prototxt [path to MobileNetSSD config] --weights [path to MobileNetSSD weights]
```

</br>

**🎯 Accuracy for social distance monitoring:**
Dataset | TP | TN | FP | FN | %
------- | -- | -- | -- | -- | --
Oxford Town Centre | 11 | 19 | 14 | 4 | 62.5
PETS2009 | 14 | 38 | 19 | 5 | 68
VIRAT | 9 | 4 | 0 | 10 | 56.5

**🎯 Accuracy for safety violation alert based on segmented ROI:**
Dataset | TP | TN | FP | FN | %
------- | -- | -- | -- | -- | --
Oxford Town Centre | 55 | 58 | 0 | 5 | 95.8

---

## Kindly check out below URL:

### ☕ MobileNet SSD(Caffe)

_🔗 Object detection model:_ [MobileNetSSD Caffe](https://github.com/chuanqi305/MobileNet-SSD)

_CC:_ [chuanqi305](https://github.com/chuanqi305/), [FreeApe](https://github.com/FreeApe/VGG-or-MobileNet-SSD)
