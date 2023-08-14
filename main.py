
from _7_predict import *
from _6_model import *
from _1_lib import *



if __name__== "__main__":
    # model()
   
    # for i in range(151,201):
    #     path='./data/train/Dog/d'+str(i)+'.png'
    #     img= Image.open(path)
    #     label = predict(img)
    #     print(label)

    path= './data/Test/WithMask/3.png'
    
    img = Image.open(path)
    label = predict(img)

    img = cv2.imread(path)
    img=cv2.resize(img,(200,200))
    if label=='WithMask':
        cv2.putText(img,'WithMask',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),4)
        cv2.imshow("WithMask",img)
    else:
        cv2.putText(img,'WithoutMask',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),4)
        cv2.imshow("WithoutMask",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()