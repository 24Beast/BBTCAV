# BBTCAV
Calculating concept activation for Black Box Models

### Tests

#### Run AutoEncoder based BBTCAV on CelebA

In the main directory, run the following snippet:

`python -m tests.CelebA_AETCAV_test`

#### Run TCAV on CelebA

In the main directory, run the following snippet:

`python -m tests.CelebA_TCAV`

### Requirements
torch: 2.7.0
torchvision: 0.22.0
captum: 0.7.0

###NOTE

If you wish to use GPU for TCAV, do the following steps otherwise you may get "RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu), a potential fix that worked for me was:"

1. Go to the captum/concept/_utils/classifier.py file.
2. Go to line ~196 which says predict = self.lm.classes()[torch.argmax(predict, dim=1)].
3. Change it to predict = self.lm.classes()[torch.argmax(predict.cpu(), dim=1)]


The linear model stays on CPU causing the error unless the code above is modified.


