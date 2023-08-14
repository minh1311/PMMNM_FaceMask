from _1_lib import *
from _2_config import *
from _3_Image_Transform import *
from _4_dataset import *
from _5_ultis import *

def model():
    #path list
    train_list=make_datapath_list('train')
    val_list= make_datapath_list('val')

    #dataset 
    train_dataset = MyDataSet(train_list,transform=ImageTransform(size,mean,std),phase='train')
    val_dataset = MyDataSet(val_list,transform=ImageTransform(size,mean,std),phase='val')

    #dataloader: để dễ truy cập trong model
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle='False')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size,shuffle='False')
    dataloader_dict={'train':train_dataloader,'val':val_dataloader}

    #network
    use_pretrained=True
    net = models.vgg16(pretrained= use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096,out_features=2)

    #loss
    criterior = nn.CrossEntropyLoss()

    #optimize
    params = params_to_update(net)
    # print(params)
    # print(np.array(params).shape)
    optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)

    # #training
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)