These experiments are based on PyTorch's official ImageNet training:

    https://github.com/pytorch/examples/tree/main/imagenet

WHTResNet50x3 is the 3-path HT-ResNet50, and the input size is 224x224.

WHTResNet50x3_256 is the 3-path HTResNet50, but the input size is 256x256.

To train the network:
        
    python main.py -a wht_resnet50 -b 128 --lr 0.05

To test the network:

    python test.py -a wht_resnet50 -b 128 -b10 32

The test code contains a 10-fold test, and the 10-fold test batch size is 32. We reduce this size to avoid the memory issue. 
