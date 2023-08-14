from _1_lib import *
from _2_config import *
from _3_Image_Transform import ImageTransform
from _5_ultis import *

class_index = ['WithMask','WithoutMask']

class Predictor():
    def __init__(self,class_index):
        self.clas_index = class_index

    def predict_max(self, output): # [0.9, 1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.clas_index[max_id]
        return predicted_label

predictor = Predictor(class_index)

def predict(img):
    #prepare network
    use_predicted = True
    net = models.vgg16(pretrained= use_predicted)
    net.classifier[6]=nn.Linear(in_features = 4096,out_features = 2)
    net.eval()
    save_weight='.\weight_transfer_classifier6.pth'
    #prepare model
    model = load_model(net,save_weight)

    #prepare model
    transform = ImageTransform(size,mean,std)
    img = transform(img,phase='test')
    img = img.unsqueeze_(0)

    #predict
    output = model(img)
    response = predictor.predict_max(output)

    return response