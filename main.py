'''
Kubilay Bozak
Samet Kara
Hüsna Şişli
'''
import math
import cv2
import skimage.exposure as exposure
import matplotlib.pyplot as plt
from skimage import draw
import numpy as np
import skimage



def polygon2mask(image_shape, polygon):
    mask = skimage.draw.polygon2mask(image_shape, polygon)
    return mask


def mypolygon_perimeter(X, Y, image_shape):
    rr, cc = skimage.draw.polygon_perimeter(X, Y, image_shape)
    return rr, cc


# read images
imgs = cv2.imreadmulti("health.tif", flags=cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)[1]

for i, img in enumerate(imgs):
    filename = f"face_1_frame-{i}.png"
    print(f"Processing frame {i} into file {filename}")
    # normalize image to 8-bit range
    img_norm = exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)).astype(np.uint8)
    print(img_norm)

    cv2.imwrite(filename, img_norm)

    # display normalized image
    cv2.imshow('normalized', img_norm)



def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return (x, y)


nBands = 10
mSize = 150
mzSize = 5
filts = np.ones((nBands, 1))
#print("filts değerli :")
#print(filts)
dirs = np.zeros((2, nBands))
#print("dirs değerleri :")
#print(dirs)
theta = np.arange(0, nBands)
theta = theta * (math.pi / nBands)
#print("theta değerleri :")
#print(theta)
rho = np.ones((1, nBands))
#print("rho değerleri :")
#print(rho)
[X, Y] = pol2cart(theta, rho)
# image_shape = (mSize, mzSize)
# XX = X[0]
# YY = Y[0]
# polygon = np.array([XX, YY]).T ## çalışıyor
#print("[X,Y] değerleri")
#print([X, Y])
#print(" Sadece X değerleri :")
#print(X)
#print(" Sadece y değerleri :")
#print(Y)
#print()
#print()
ang = math.atan(mzSize / mSize)
#print("ang değerleri")
#print(ang)
dist = math.ceil((math.sqrt(math.pow((mzSize / 2), 2) + math.pow((mSize / 2), 2))))
#print("dist değerleri")
#print(dist)
kt = 0;
for_start = math.pi / nBands
# print(for_start)
for_last = math.pi
# print(for_last)
toplam_eleman = (for_last / for_start)
FilteringResutls = []
for k in range(0, int(toplam_eleman)):
    kk = for_start * (k + 1)
    ang1 = (kk - ang)
    #print("ang1 : " + str(ang1))
    ang2 = (kk + ang)
    #print("ang1 : " + str(ang2))
    theta = [ang1, ang2, ang1, ang2, ang1]
    #print(" for içinde bulunan theta")
    #print(theta)
    rho = np.array([1, 1, -1, -1, 1]) * dist
    #print("rho")
    #print(rho)
    [X, Y] = pol2cart(theta, rho)
    #print(" For içinde bulunan X 'in Değişmeden öncesi ")
    #print(X)
    X22 = X + math.ceil(mSize / 2);
    #print("Değişen X ")
    #print(X22)
    #print(" For içinde bulunan Y 'nin Değişmeden öncesi ")
    #print(Y)
    Y22 = Y + math.ceil(mSize / 2);
    #print("Değişen Y ")
    #print(Y22)
    polygon22 = np.array([X22, Y22]).T
    #print("polygon22 deperi")
    #print(polygon22)
    image_shape = (mSize, mSize)
    mask = polygon2mask(image_shape, polygon22)
    count = np.count_nonzero(mask)
    #print("mask edilmiş hali")
    #print(mask)
    #print("count değeri")
    #print(count)
    mask_int = mask.astype(int)
    count = np.count_nonzero(mask_int)
    #print("count değeri")
    #print(count)
    #print("1,0 hali")
    #print(mask_int)
    a = mask_int / count;
    #print(a)
    FilteringResutls.append(a)
    #plt.imshow(mask)
    #plt.show()

import sys
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from skimage import io, img_as_float
ConvolResults = []

for value in range(0,len(FilteringResutls)):
    img_gaussian_noise = img_as_float(io.imread('face_1_frame-0.png', as_gray=True))
    img_salt_pepper_noise = img_as_float(io.imread('face_1_frame-0.png', as_gray=True))
    img = img_salt_pepper_noise
    kernel = FilteringResutls[value]

    conv_using_cv2 = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ConvolResults.append(conv_using_cv2)

    conv_using_scipy = convolve2d(img, kernel, mode='same')

    conv_using_scipy2 = convolve(img, kernel, mode='constant', cval=0.0)

    filename_1 = f"nothasta bireye Uygulanan Filtre-{value}.png"
    filename_2 = f"conv_using_scipy-{value}."
    filename_3 = f"conv_using_scipy2-{value}"
    #cv2.imshow("Original", img)
    #cv2.imshow(filename_1, conv_using_cv2)
    #cv2.imshow(filename_2, conv_using_scipy)
    #cv2.imshow(filename_3, conv_using_scipy2)
    #plt.imsave(filename_1, conv_using_cv2 , cmap='gray')


#cv2.waitKey(0)
selVals,colAssignment  = np.array(ConvolResults).max(0),np.array(ConvolResults).argmax(0)
#maxInColumns = np.amax(ConvolResults, axis=0)
np.set_printoptions(threshold=sys.maxsize)
#cv2.imshow("nothasta_Max_Responce.png", selVals)
plt.imsave("maxResponce.png", selVals, cmap='gray')


'''for i in range(0,512):
    for k in range(0,512):
        if(0<=colAssignment[i][k]<=2):
            colAssignment[i][k]=1
        if (3 <= colAssignment[i][k]<=5):
            colAssignment[i][k] =4
        if (6 <= colAssignment[i][k]<9):
            colAssignment[i][k] =7
'''
print('matris2', colAssignment)
#plt.imsave("test.png",colAssignment)
plt.imsave("nothasta_yönelim.png",colAssignment)
c = plt.imshow(colAssignment)
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.colorbar(c,boundaries=bounds)
plt.title('Yönelim Haritası', fontweight ="bold")
plt.show()
#cv2.imshow("sssss", maxInColumns)


for i in range(0,10,1):
    new_name="saglikli"+str(i)+".png"
    new_colAssignment=np.where(colAssignment == i, colAssignment, -1)
    plt.imsave(new_name, new_colAssignment)


im = cv2.imread("saglikli0.png", 0)  # dosyayi oku
plt.subplot(2, 2, 1)
plt.imshow(im, cmap="gray")

# inverse ikili esik degeri
_, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV)
plt.subplot(2, 2, 2)
plt.imshow(im, cmap="gray")

im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones([1, 20]))  # gurultuden kurtul
_, comp = cv2.connectedComponents(im)  # connected components analizi
plt.subplot(2, 2, 3)
plt.imshow(im, cmap="gray")
'''
temp = 0
for i in range(0,512):
    temp = 0
    for k in range(0,512):
        if(comp[i][k]==1):
            temp=comp[i][k]+temp
            if(temp==100):
                for j in range(0,512):
                    comp[i][j]=0
            temp=0
'''
plt.subplot(2, 2, 4, label="renk1")
c = plt.imshow(comp, cmap="nipy_spectral",label="renk2")
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.colorbar(c,boundaries=bounds)
plt.title('Connected Component Analysis', fontweight ="bold")
print(comp)

plt.show()

cv2.waitKey(0)

