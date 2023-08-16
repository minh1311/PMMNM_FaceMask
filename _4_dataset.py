from _1_lib import *

# Quy trinh:
#     1. contructer
#     2.len
#     3.path
#     4.image
#     5.transform
#     6.label
#     7. return image, label
#     Mục đích: transform ảnh, gán label
#               lấy độ dài
class MyDataSet():
    def __init__(self, file_list, phase, transform):
        self.file_list=file_list
        self.phase=phase
        self.transform=transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        img_path= self.file_list[index]
        img=Image.open(img_path)#mở ảnh bằng PILLOW, OPENCV

        img_transformed  = self.transform(img,self.phase)#chuyển đổi ảnh

        if self.phase == 'train':#gán nhãn
            label = img_path[13:21]
        elif self.phase == 'val':
            label = img_path[11:19]

        if  label=='WithMask':
            label = 0
        else:
            label = 1

        return img_transformed,label
