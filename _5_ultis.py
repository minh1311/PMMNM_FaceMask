from _1_lib import *
from _2_config import *

# tạo một list path ảnh
# tạo một train
# tạo một params update
    

def make_datapath_list(phase):#quản lí đường dẫn của ảnh
    rootpath = './data/'#đường dẫn gốc
    target_path=osp.join(rootpath + phase+ "/**/*.png")#phase+ "/**/*.png": đường dẫn trực tiếp của ảnh

    
    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

def params_to_update(net):
    params_to_update = []

    update_param_name = ['classifier.6.weight','classifier.6.bias']

    for name,param in net.named_parameters():
        if name in update_param_name:
            param.requires_grad=True
            params_to_update.append(param)
        else:
            param.requires_grad=False
    return params_to_update

def train_model(net, dataloader_dict, criterior,optimizer,num_epochs):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ',device)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs))

        # move network to device(GPU/CPU)
        net.to(device)

        torch.backends.cudnn.benchmark = True
        for phase in ['train','val']:
            if phase=='train':
                net.train()
            else:
                net.eval()

            epoch_loss= 0.0
            epoch_corrects = 0
            if epoch==0 and phase=='train':
                continue
            for inputs,labels in tqdm(dataloader_dict[phase]):
                #move inputs, labels to GPU 
                inputs= inputs.to(device)
                labels= labels.to(device)
                #với mỗi node/layer sẽ có weight khác nhau. zero_grad() reset weight sau mỗi update về
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs=net(inputs)
                    loss=criterior(outputs,labels)#Tinh loss bằng crossentropy
                    _, preds = torch.max(outputs,1)#tính max bằng softmax

                    if phase=='train':
                        loss.backward()#đạo hàm loss
                        optimizer.step()#update weights

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)


            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:4f}".format(phase,epoch_loss,epoch_accuracy))

    torch.save(net.state_dict(),save_path)


def load_model(net,model_path):
    load_weights = torch.load(model_path, map_location={'cuda:0':'cpu'})
    net.load_state_dict(load_weights)
    return net

