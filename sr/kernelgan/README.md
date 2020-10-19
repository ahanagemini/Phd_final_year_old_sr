# Blind Super-Resolution Kernel Estimation using an Internal-GAN
# "KernelGAN"
### Sefi Bell-Kligler, Assaf Shocher, Michal Irani 
*(Official implementation)*

Paper: https://arxiv.org/abs/1909.06581

Project page: http://www.wisdom.weizmann.ac.il/~vision/kernelgan/  


## Usage:

``` python3 -m pip install -r requirements.txt ```

Then use the function train in train.py from the parent package.
This will produce the kernel you need.


### Data:
Download the DIV2KRK dataset: [dropbox](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip)

Reproduction code for your own Blind-SR dataset: [github](https://github.com/assafshocher/BlindSR_dataset_generator)
