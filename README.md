# Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI

![FYP](https://img.shields.io/badge/Build-v1.0_pass-brightgreen) [![LICENSE](https://img.shields.io/badge/license-MIT-blue)](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP/blob/master/LICENSE) [![SCOPUS](https://img.shields.io/badge/Index-SCOPUS-orange)](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85093867522&doi=10.1109%2fICCSCE50387.2020.9204934&partnerID=40&md5=375a3cd12ad9bd52e66b1a4201fefe89)

Person detection algorithm used is [MobileNet SSD](https://github.com/chuanqi305/MobileNet-SSD 'MobileNet SSD Caffe') with Caffe implementation and the model pre-trained on MS-COCO. Both programs uses OpenCV API for image processing and utilizing the DNN module (tested on CPU). The programs later tested on several datasets to prove the concepts.
</br>

_Install the dependencies on command line:_

```sh
$ pip3 install -r requirements.txt
```

_To run the program on command line:_

```sh
$ python3 social-distanceApp.py
```

_Or_

```sh
$ python3 safety-violationApp.py
```

_Edit program configuration on **config.py**:_

- To use device's camera as program input, change `CAMERA = False` to `CAMERA = True`.
- To apply threading parallelism, change `THREAD = False` to `THREAD = True`.
  </br>

### Output example:

|    _Social distance monitoring_    | _Safety violation alert based on segmented ROI_ |
| :--------------------------------: | :---------------------------------------------: |
| ![outputimage](/images/output.gif) |       ![outputimage](/images/output2.gif)       |

**Accuracy for social distance monitoring:**

| Dataset            | TP  | TN  | FP  | FN  | %    |
| ------------------ | --- | --- | --- | --- | ---- |
| Oxford Town Centre | 11  | 19  | 14  | 4   | 62.5 |
| PETS2009           | 14  | 38  | 19  | 5   | 68   |
| VIRAT              | 9   | 4   | 0   | 10  | 56.5 |

**Accuracy for safety violation alert based on segmented ROI:**

| Dataset | TP  | TN  | FP  | FN  | %    |
| ------- | --- | --- | --- | --- | ---- |
| CamNeT  | 55  | 58  | 0   | 5   | 95.8 |

## Kindly check out below links for references:

**☕ Mobilenet SSD Caffe** </br>
[![Github](https://img.shields.io/badge/chuanqi305-Github-lightgrey)](https://github.com/chuanqi305/) [![Github](https://img.shields.io/badge/FreeApe-Github-lightgrey)](https://github.com/FreeApe/VGG-or-MobileNet-SSD)

**📊 Dataset** </br>
MegaPixels: Origins, Ethics, and Privacy Implications of Publicly Available Face Recognition Image Datasets </br>
[![Oxford TownCentre](https://img.shields.io/badge/Oxford_Town_Centre-URL-yellowgreen)](https://exposing.ai/oxford_town_centre/)
</br>
A Camera Network Tracking (CamNeT) Dataset and Performance Baseline </br>
[![CamNet](https://img.shields.io/badge/CamNeT-URL-yellowgreen)](https://vcg.ece.ucr.edu/datasets)

**📑 Publication** </br>
Person Detection for Social Distancing and Safety Violation Alert based on Segmented ROI </br>
[![SCOPUS](https://img.shields.io/badge/DOI-SCOPUS-orange)](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85093867522&doi=10.1109%2fICCSCE50387.2020.9204934&partnerID=40&md5=375a3cd12ad9bd52e66b1a4201fefe89)
</br>

### LICENSE

_This project is under MIT license, please refer [LICENSE](https://github.com/afiqharith/SocialDistancing-SafetyViolationROI-MobileNetSSD-FYP/blob/master/LICENSE)._
