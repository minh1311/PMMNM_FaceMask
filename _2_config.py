from _1_lib import *
#fix khi train test
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# lấy size cho ImageNet
size=224
mean = (0.485, 0.456, 0.406) 
std = (0.229, 0.224, 0.225)

#4 ảnh trong 1 batch
batch_size= 4 #4 ảnh update 1 gradient 1 lần
#train 1 epochs
num_epochs= 2

save_path='./weight_transfer_classifier6.pth'