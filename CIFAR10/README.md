resnet20 is the baseline experiment.

resnet20-HTConvx3 is the 3-path HT-Resnet-20.

resnet20-WHTConvx3 is the 3-path HT-Resnet-20, but we implement it using the WHT. However, we obtained the same test accuracy as HT, which shows the permutation in the HT does not affect this work. 

resnet20-WHTConvx1 is the 1-path HT-Resnet-20, but we implement it using the WHT.  

resnet20-WHTConvx5 is the 1-path HT-Resnet-20, but we implement it using the WHT.  

Please run "train.py" for training and "test.py" for testing:
    
    python train.py
    python test.py
