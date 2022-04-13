## How to train a target model?
Run "train_target_model.py"
This code provides three model architectures (including VggNet, ResNet and Wide-ResNet). The trained model will be saved in the "checkpoint" folder.


## How to generate adversarial training data?

-Training data
Run "example_cam.py".

We use the "Class Activation Feature based Attack" (CAFD) to generate adversaial samples. The generated samples will be saved in the "data/training" folder.

-Test data
Run "example_other.py" or "example_autoattack.py".

We use the "[advertorch](https://github.com/BorealisAI/advertorch)" toolbox to help generate adversairal samples. The first code provides ![](http://latex.codecogs.com/svg.latex?L_{\infty}) PGD, ![](http://latex.codecogs.com/svg.latex?L_{2}) CW, [DDN](https://arxiv.org/abs/1811.09600), [STA](https://openreview.net/forum?id=HyydRMZC-), etc., to generate different adversarial samples. The second code provides [Autoattack](https://arxiv.org/abs/2003.01690).
The generated samples will be saved in the "data/test" folder.


## How to train the "Class activation Feature based Defense"?
Run "train_or_test_denoiser.py" with "mode=0".

The model parameters of the used target model comes from "checkpoint" folder. The trained defense model will be saved in "checkpoint_denoise" folder.


## How to test the defense?
Run "train_or_test_denoiser.py" with "mode=1".

The input data comes from "data/test" folder, and the denoised data is saved in "results/defense" folder. Then run "test.py" to compute the accuracy rate.
