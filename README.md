# Sourse code for our ICML 2023 paper: A Hybrid Quantum-Classical Approach based on the Hadamard Transform for the Convolutional Layer
Paper link: https://proceedings.mlr.press/v202/pan23d.html

The folder “QCHT” contains the implementation of the Hybrid Quantum-Classical Hadamard Transform (QCHT) on the IBM-Q cloud platform.
https://quantum-computing.ibm.com/
This shows that the QCHT can return the exactly same results as the classical HT.

The folder "layers" contains the implementation of the proposed Hadamard Transform Perceptron Layer. This is the classical approach for the neural network since in this work we have no quantum trainable parameters. The quantum part is only in the transform implementation.
You need to import the function "WHTConv2D":

    from layers.WHT import WHTConv2D
For example, if the input tensor is 3x16x32x32 and the output is 3x16x32x32, 
single-path HT-perceptron layer: 

    WHTConv2D(32, 32, 16, 16, 1, residual=True)
3-path HT-perceptron layer: 

    WHTConv2D(32, 32, 16, 16, 3, residual=False)
The parameter "pod" in the function "WHTConv2D" stands for the number of paths.

More examples can be founded in the folder CIFAR10 and ImageNet1K.
