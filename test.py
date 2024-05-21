import cv2 as cv
import numpy as np

def image_io_demo():
    image = cv.imread("MCN.jpg")#读取的通道顺序是BGR
    h,w,c = image.shape
    print(h,w,c)

    cv.namedWindow("input",cv.WINDOW_FREERATIO)#创建名称为input，自由比例的图像显示窗口（可随意拖动）
    cv.imshow("input",image)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow("gray",gray)
    print(gray.shape)
    cv.waitKey(0)#一直在显示，直到用户敲击键盘上任意键
    cv.destroyAllWindows()

def video_io_demo():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        cv.imshow("frame",frame)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        cv.imshow("gray",gray)
        c = cv.waitKey(10)
        if c == 27:
            break
    cap.release()

def basic_ops_demo():
    image = cv.imread("MCN.jpg")#读取的通道顺序是BGR
    h,w,c = image.shape
    print(h,w,c)    

    mv = cv.split(image)#图像分通道
    cv.imshow("blue",mv[0])#mv[0]是B通道，1是G通道，2是R通道
    cv.waitKey(0)

    blob = cv.resize(image,(300,300))#调整图像大小
    cv.imshow("blob",blob)
    cv.waitKey(0)
    print(blob.shape)#HWC

    #NCHW，扩充通道数
    image_blob = blob.transpose(2,0,1)
    print(image_blob.shape)
    #两种方法均可以实现扩充通道数的作用，其中用np.expand_dims()更加直观和便捷
    image_blob = np.expand_dims(image_blob,0)
    image_blob = image_blob.reshape(1, *image_blob.shape)
    print(image_blob.shape)

    #找最大和最小值
    a = np.array([1,2,3,4,5,6])
    index_max = np.argmax(a)
    index_min = np.argmin(a)
    print(a[index_max])
    for row in range(h):
        for col in range(w):
            b,g,r = image[row,col]
            print(b,g,r)


if __name__ == "__main__":
    basic_ops_demo()