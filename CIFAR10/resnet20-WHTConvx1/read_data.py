from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def read_cifar10(batchsize,data_dir):

    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  
                                    transforms.RandomCrop(32, padding=4),  
                                    transforms.RandomHorizontalFlip(p=0.5), 
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    data_train = datasets.CIFAR10(root=data_dir,
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    data_test = datasets.CIFAR10(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 download=True
                                 )

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    return data_loader_train,data_loader_test

