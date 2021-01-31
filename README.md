<div align="center">
  <img src="images/Github.png" width="250" height="250">
</div>

# 🚶‍♂️ Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI ![FYP](https://img.shields.io/badge/Build-v1.0_pass-brightgreen) [![LICENSE](https://img.shields.io/badge/license-MIT-blue)](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP/blob/master/LICENSE)

Person detection algorithm used is MobileNet SSD with Caffe implementation and the model pre-trained on MS-COCO by [chuanqi305@github](https://github.com/chuanqi305/MobileNet-SSD 'MobileNet SSD'). Both programs uses OpenCV API for image processing and utilizing the DNN module (tested on CPU). The programs later tested on several datasets to prove the concepts.
</br>

_💻 To run the program on command line:_

**Non-Threading program**

```sh
$ python3 social-distance.py --video [input video] --prototxt [MobileNetSSD cfg] --weights [MobileNetSSD weights]
```

```sh
$ python3 safety-violation.py --video [input video] --prototxt [MobileNetSSD cfg] --weights [MobileNetSSD weights]
```

</br>

**Threading program**

```sh
$ python3 social-distance(threading).py --video [input video] --prototxt [MobileNetSSD cfg] --weights [MobileNetSSD weights]weights]
```

```sh
$ python3 safety-violation(threading).py --video [input video] --prototxt [MobileNetSSD cfg] --weights [MobileNetSSD weights]weights]
```

</br>

### 🎬 Output example:

**_Social distance monitoring:_**
| ![outputimage](/images/output.gif) |
| ---------------------------------- |

**_Safety violation alert based on segmented ROI:_**
| ![outputimage](/images/output2.gif) |
| ----------------------------------- |

**🎯 Accuracy for social distance monitoring:**

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 11  | 19  | 14  | 4   | 62.5 |
| PETS2009           | 14  | 38  | 19  | 5   | 68   |
| VIRAT              | 9   | 4   | 0   | 10  | 56.5 |

**🎯 Accuracy for safety violation alert based on segmented ROI:**

| Dataset | TP  | TN  | FP  | FN  | %    |
| ------- | --- | --- | --- | --- | ---- |
| CamNeT  | 55  | 58  | 0   | 5   | 95.8 |

---

## Kindly check out below links:

☕ [![MobileNetSSD Caffe](https://img.shields.io/badge/MobileNet_SSD_Caffe-Github-lightgrey)](https://github.com/chuanqi305/MobileNet-SSD)

_Credit:_ [![Github](https://img.shields.io/badge/chuanqi305-Github-lightgrey)](https://github.com/chuanqi305/) [![Github](https://img.shields.io/badge/FreeApe-Github-lightgrey)](https://github.com/FreeApe/VGG-or-MobileNet-SSD)

**📊 Dataset**

MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets </br>
[![Oxford TownCentre](https://img.shields.io/badge/Oxford_Town_Centre-URL-yellowgreen)](https://megapixels.cc/oxford_town_centre/)
</br>

A Camera Network Tracking (CamNeT) Dataset and Performance Baseline </br>
[![CamNet](https://img.shields.io/badge/CamNeT-URL-yellowgreen)](https://vcg.ece.ucr.edu/datasets)

**📑 Publication**

Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI </br>
[![SCOPUS](https://img.shields.io/badge/Scopus-DOI-orange)](https://www.scopus.com/record/display.uri?eid=2-s2.0-85093867522&doi=10.1109%2fICCSCE50387.2020.9204934&origin=inward&txGid=9b9b8e6a62cd80f3ea7f3bb9929641e7)
</br>

### LICENSE

_This project is under MIT license, please look at [LICENSE](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP/blob/master/LICENSE)._
