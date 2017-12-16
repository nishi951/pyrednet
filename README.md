# pyrednet
Extension of prednet for CS229 Project.

TODO:
- get loader to work with full training dataset w/o crapping out on memory
[x] see if weights need to be initialized in neural net
[x] try visualizing examples to see what they look like
[x] see why our train loop has 35 and theirs has 125
- save weights



### PAPER STRUCTURE
[ ] abstract
[ ] introduction
[ ] related work
    [x] video frame prediction
    [x] predictive coding
[ ] methods/model
    [ ] architecture
    [ ] propagation
    [ ] parameters
    [x] pyrednet
[ ] dataset and features
    [x] source
    [x] preprocessing
[ ] experiments/results
    [ ] basic iteration
        [ ] example predictions
        [ ] loss plots
    [ ] peephole vs not peephole
    [ ]  **predict center pixels with linear regression** baseline model
[ ] discussions
    [ ] discussion of PyTorch vs Keras in training
    [ ] discussions of limitations in model, particular training difficulties like zeros
[ ] conclusions
[ ] future work

