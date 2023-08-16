from _1_lib import *
from _2_config import *

    # xử lý ảnh cho từng train,test,val bằng transforms.compose
    # call gọi nó ra

class ImageTransform():
    def __init__(self,resize,mean,std):
        self.data_transform={
            'train': transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize,scale=(0.5,1)),
                    transforms.RandomHorizontalFlip(0.5),   # tỷ lệ xoay ngang ảnh 0.5
                    transforms.ToTensor(),# ma tran 3 chieu
                    transforms.Normalize(mean,std)#đưa về chuẩn ImageNet mục đích train nhanh
                ]
            ),
            'val': transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ]
            ),
            'test': transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ]
            )
        }
    def __call__(self,img,phase='train'):
        return self.data_transform[phase](img)    
