<div align="center">
  <img src="images/Github.png" width="250" height="250">
</div>

# 🚶‍♂️ Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI ![FYP](https://img.shields.io/badge/Build-v1.0_pass-brightgreen) [![LICENSE](https://img.shields.io/badge/license-MIT-blue)](https://github.com/afiqharith/Social-Distancing-and-Safety-Violation-Alert-ROI-MobileNetSSD-FYP/blob/master/LICENSE)

The idea of this project is to use MobileNet SSD with Caffe implementation as the person detection algorithm. This program uses OpenCV API for image processing and utilizing the existing DNN module.
</br>

_💻 To run the program on command line:_

**Non-Threading program**

```sh
$ python3 social-distance.py --video [path to input] --prototxt [path to MobileNetSSD config] --weights [path to MobileNetSSD weights]
```

```sh
$ python3 safety-violation.py --video [path to input] --prototxt [path to MobileNetSSD config] --weights [path to MobileNetSSD weights]
```

</br>

**Threading program**

```sh
$ python3 social-distance(threading).py --video [path to input] --prototxt [path to MobileNetSSD config] --weights [path to MobileNetSSD weights]
```

```sh
$ python3 safety-violation(threading).py --video [path to input] --prototxt [path to MobileNetSSD config] --weights [path to MobileNetSSD weights]
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
[![IEEE](https://img.shields.io/badge/IEEE_Xplore-DOI-blue)](https://doi.org/10.1109/ICCSCE50387.2020.9204934)
</br>

### LICENSE

_This project is under MIT license, please look at [LICENSE](https://github.com/afiqharith/Social-Distancing-and-Safety-Violation-Alert-ROI-MobileNetSSD-FYP/blob/master/LICENSE)._
