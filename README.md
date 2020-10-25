# IIB_project

**Yungyu Song, Churchill College**
**ys467**

## Project Summary

Gas sensors are currently ubiquitous with their use in countless aspects of our daily life. The market is currently dominated by MOx based chemiresistive gas sensors due to their cost and simplicity. The main drawbacks of MOx is the need for a heating element for operation. Therefore, new sensing materials such as rGO/MOx composite is currently being studied.
However, several issues persist. For this project, I identified two key issues: presence of baseline drift and poor device selectivity. Baseline drift is caused by sensor being unable to reset after gas detection and selectivity problem arises from sensor being affected by not only its target analyte: both issues cause sensor inaccuracy.
The goal is to therefore tackle these problems in order to achieve optimised performance.

Specifically for poor device selectivity, the problem arises as the sensor is not only responsive to its target analyte, but to also other gases, especially humidity.
One of the possible solution is to use a multi-sensor system where responses of different sensors are compared for an accurate measurement.
For this project, I used a single sensor, and applied a mathematical tool called PCA (principal component analysis), where the sensor parameters of interest are chosen to be as follows.

- Maximum sensitivity (peak response value)
- Response time (time taken between 10% to 90% of the peak)
- Area ratio (ratio between areas under the curve during response and recovery)

Classifiers are then applied on PCA results, which allow decision regions to be formed, each corresponding to a type of analyte.
Several classifiers are applied to determine optimal classifiers and the degree of acceptable randomness of data for classifiers to function.

## Acknowledgements
This project is supported by Dr. Tawfique Hasan & Mr. Osarenkhoe Ogbeide, University of Cambridge.
