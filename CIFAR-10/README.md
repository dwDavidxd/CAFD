## How to generate adversarial training data?
Run "example_cam.py".

We use the "Class Activation Feature based Attack" (CAFD) to generate adversaial samples. We also provide some similar methods and variants to generate adversarial samples.


## How to train the "Adversarial noise Removing Network"?
Run "train_ARN.py"

See "./config/adver.yaml" for network configurations and data selection. 

The training data includes natural data and two types of adversarial data.

## How to test the "Adversarial noise Removing Network"?
Run "test_ARN.py"

See "./config/adver.yaml" for network configurations and data selection.

Input the natural or adversairal data into the ARN and obtain the processed data. Then, input the processed data into the target model.

"[advertorch](https://github.com/BorealisAI/advertorch)" toolbox to help generate adversairal samples. This code provides ![](http://latex.codecogs.com/svg.latex?L_{\infty}) PGD, ![](http://latex.codecogs.com/svg.latex?L_{2}) CW, [DDN](https://arxiv.org/abs/1811.09600) and [STA](https://openreview.net/forum?id=HyydRMZC-) attacks to generate different adversarial samples.

The generated samples can be saved with ".png" and ".npy" format. The storage directory defaults to "adv_example".
