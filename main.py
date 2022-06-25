'''
Kubilay Bozak  kubilaybozak@gmail.com
Samet Kara     samet.krx@gmail.com
Hüsna Şişli    husnasisli@gmail.com
'''

import math as math
import skimage.exposure as exposure
import matplotlib.pyplot as plt
from skimage import draw
import skimage
import sys
import cv2
import cv2 as cv
import numpy as np
import numpy
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from skimage import io, img_as_float
import time
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)






#Görselin adı
#nBands = 10
#mSize = 40
#mzSize = 5








def small(nBands,mSize,mzSize,imgName):

    nBands = nBands
    mSize = mSize
    mzSize = mzSize
    filts = np.ones((nBands, 1))

    def polygon2mask(image_shape, polygon):
        mask = skimage.draw.polygon2mask(image_shape, polygon)
        return mask

    def mypolygon_perimeter(X, Y, image_shape):
        rr, cc = skimage.draw.polygon_perimeter(X, Y, image_shape)
        return rr, cc

    # read images
    imgs = cv2.imreadmulti(imgName, flags=cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)[1]

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



    # print("filts değerli :")
    # print(filts)
    dirs = np.zeros((2, nBands))
    # print("dirs değerleri :")
    # print(dirs)
    theta = np.arange(0, nBands)
    theta = theta * (math.pi / nBands)
    # print("theta değerleri :")
    # print(theta)
    rho = np.ones((1, nBands))
    # print("rho değerleri :")
    # print(rho)
    [X, Y] = pol2cart(theta, rho)
    # image_shape = (mSize, mzSize)
    # XX = X[0]
    # YY = Y[0]
    # polygon = np.array([XX, YY]).T ## çalışıyor
    # print("[X,Y] değerleri")
    # print([X, Y])
    # print(" Sadece X değerleri :")
    # print(X)
    # print(" Sadece y değerleri :")
    # print(Y)
    # print()
    # print()
    ang = math.atan(mzSize / mSize)
    # print("ang değerleri")
    # print(ang)
    dist = math.ceil((math.sqrt(math.pow((mzSize / 2), 2) + math.pow((mSize / 2), 2))))
    # print("dist değerleri")
    # print(dist)
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
        # print("ang1 : " + str(ang1))
        ang2 = (kk + ang)
        # print("ang1 : " + str(ang2))
        theta = [ang1, ang2, ang1, ang2, ang1]
        # print(" for içinde bulunan theta")
        # print(theta)
        rho = np.array([1, 1, -1, -1, 1]) * dist
        # print("rho")
        # print(rho)
        [X, Y] = pol2cart(theta, rho)
        # print(" For içinde bulunan X 'in Değişmeden öncesi ")
        # print(X)
        X22 = X + math.ceil(mSize / 2);
        # print("Değişen X ")
        # print(X22)
        # print(" For içinde bulunan Y 'nin Değişmeden öncesi ")
        # print(Y)
        Y22 = Y + math.ceil(mSize / 2);
        # print("Değişen Y ")
        # print(Y22)
        polygon22 = np.array([X22, Y22]).T
        # print("polygon22 deperi")
        # print(polygon22)
        image_shape = (mSize, mSize)
        mask = polygon2mask(image_shape, polygon22)
        count = np.count_nonzero(mask)
        # print("mask edilmiş hali")
        # print(mask)
        # print("count değeri")
        # print(count)
        mask_int = mask.astype(int)
        count = np.count_nonzero(mask_int)
        # print("count değeri")
        # print(count)
        # print("1,0 hali")
        # print(mask_int)
        a = mask_int / count;
        # print(a)
        FilteringResutls.append(a)
        # plt.imshow(mask)
        # plt.show()

    ConvolResults = []

    for value in range(0, len(FilteringResutls)):
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
        # cv2.imshow("Original", img)
        # cv2.imshow(filename_1, conv_using_cv2)
        # cv2.imshow(filename_2, conv_using_scipy)
        # cv2.imshow(filename_3, conv_using_scipy2)
        # plt.imsave(filename_1, conv_using_cv2 , cmap='gray')

    # cv2.waitKey(0)
    selVals, colAssignment = np.array(ConvolResults).max(0), np.array(ConvolResults).argmax(0)
    # maxInColumns = np.amax(ConvolResults, axis=0)
    np.set_printoptions(threshold=sys.maxsize)
    # cv2.imshow("nothasta_Max_Responce.png", selVals)
    plt.imsave("maxResponce.png", selVals, cmap='gray')

    """
    for i in range(0,512):
        for k in range(0,512):
            if(0<=colAssignment[i][k]<=2):
                colAssignment[i][k]=1
            if (3 <= colAssignment[i][k]<=5):
                colAssignment[i][k] =4
            if (6 <= colAssignment[i][k]<9):
                colAssignment[i][k] =7
    """
    print('matris2', colAssignment)
    # plt.imsave("test.png",colAssignment)
    plt.imsave("nothasta_yönelim.png", colAssignment)
    c = plt.imshow(colAssignment)
    bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.colorbar(c, boundaries=bounds)
    plt.title('Yönelim Haritası', fontweight="bold")
    plt.show()
    # cv2.imshow("sssss", maxInColumns)
#small(10,10,5,"1.tif")

def Normal(nBands,mSize,mzSize):

    nBands = nBands
    mSize = mSize
    mzSize = mzSize
    filts = np.ones((nBands, 1))

    def polygon2mask(image_shape, polygon):
        mask = skimage.draw.polygon2mask(image_shape, polygon)
        return mask

    def mypolygon_perimeter(X, Y, image_shape):
        rr, cc = skimage.draw.polygon_perimeter(X, Y, image_shape)
        return rr, cc

    # read images
    imgs = cv2.imreadmulti("health.tif", flags=cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)[1]

    for i, img in enumerate(imgs):
        filename = f'face_1_frame-{i}' + str(mSize) + '.png'
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

    filts = np.ones((nBands, 1))
    # print("filts değerli :")
    # print(filts)
    dirs = np.zeros((2, nBands))
    # print("dirs değerleri :")
    # print(dirs)
    theta = np.arange(0, nBands)
    theta = theta * (math.pi / nBands)
    # print("theta değerleri :")
    # print(theta)
    rho = np.ones((1, nBands))
    # print("rho değerleri :")
    # print(rho)
    [X, Y] = pol2cart(theta, rho)
    # image_shape = (mSize, mzSize)
    # XX = X[0]
    # YY = Y[0]
    # polygon = np.array([XX, YY]).T ## çalışıyor
    # print("[X,Y] değerleri")
    # print([X, Y])
    # print(" Sadece X değerleri :")
    # print(X)
    # print(" Sadece y değerleri :")
    # print(Y)
    # print()
    # print()
    ang = math.atan(mzSize / mSize)
    # print("ang değerleri")
    # print(ang)
    dist = math.ceil((math.sqrt(math.pow((mzSize / 2), 2) + math.pow((mSize / 2), 2))))
    # print("dist değerleri")
    # print(dist)
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
        # print("ang1 : " + str(ang1))
        ang2 = (kk + ang)
        # print("ang1 : " + str(ang2))
        theta = [ang1, ang2, ang1, ang2, ang1]
        # print(" for içinde bulunan theta")
        # print(theta)
        rho = np.array([1, 1, -1, -1, 1]) * dist
        # print("rho")
        # print(rho)
        [X, Y] = pol2cart(theta, rho)
        # print(" For içinde bulunan X 'in Değişmeden öncesi ")
        # print(X)
        X22 = X + math.ceil(mSize / 2);
        # print("Değişen X ")
        # print(X22)
        # print(" For içinde bulunan Y 'nin Değişmeden öncesi ")
        # print(Y)
        Y22 = Y + math.ceil(mSize / 2);
        # print("Değişen Y ")
        # print(Y22)
        polygon22 = np.array([X22, Y22]).T
        # print("polygon22 deperi")
        # print(polygon22)
        image_shape = (mSize, mSize)
        mask = polygon2mask(image_shape, polygon22)
        count = np.count_nonzero(mask)
        # print("mask edilmiş hali")
        # print(mask)
        # print("count değeri")
        # print(count)
        mask_int = mask.astype(int)
        count = np.count_nonzero(mask_int)
        # print("count değeri")
        # print(count)
        # print("1,0 hali")
        # print(mask_int)
        a = mask_int / count;
        # print(a)
        FilteringResutls.append(a)
        # plt.imshow(mask)
        # plt.show()

    ConvolResults = []

    for value in range(0, len(FilteringResutls)):
        img_gaussian_noise = img_as_float(io.imread('face_1_frame-0' + str(mSize) + '.png', as_gray=True))
        img_salt_pepper_noise = img_as_float(io.imread('face_1_frame-0' + str(mSize) + '.png', as_gray=True))
        img = img_salt_pepper_noise
        kernel = FilteringResutls[value]

        conv_using_cv2 = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        ConvolResults.append(conv_using_cv2)

        conv_using_scipy = convolve2d(img, kernel, mode='same')

        conv_using_scipy2 = convolve(img, kernel, mode='constant', cval=0.0)

        filename_1 = f"nothasta bireye Uygulanan Filtre-{value}-{str(mSize)}.png"
        filename_2 = f"conv_using_scipy-{value}."
        filename_3 = f"conv_using_scipy2-{value}"
        # cv2.imshow("Original", img)
        # cv2.imshow(filename_1, conv_using_cv2)
        # cv2.imshow(filename_2, conv_using_scipy)
        # cv2.imshow(filename_3, conv_using_scipy2)
        # plt.imsave(filename_1, conv_using_cv2 , cmap='gray')

    # cv2.waitKey(0)
    selVals, colAssignment = np.array(ConvolResults).max(0), np.array(ConvolResults).argmax(0)
    # maxInColumns = np.amax(ConvolResults, axis=0)
    np.set_printoptions(threshold=sys.maxsize)
    # cv2.imshow("nothasta_Max_Responce.png", selVals)
    plt.imsave("maxResponce" + str(mSize) + ".png", selVals, cmap='gray')

    """
    for i in range(0,512):
        for k in range(0,512):
            if(0<=colAssignment[i][k]<=2):
                colAssignment[i][k]=1
            if (3 <= colAssignment[i][k]<=5):
                colAssignment[i][k] =4
            if (6 <= colAssignment[i][k]<9):
                colAssignment[i][k] =7
    """
    print('matris2', colAssignment)
    # plt.imsave("test.png",colAssignment)
    plt.imsave("nothasta_yönelim" + str(mSize) + ".png", colAssignment)
    c = plt.imshow(colAssignment)
    bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.colorbar(c, boundaries=bounds)
    plt.title('Yönelim Haritası', fontweight="bold")
    plt.show()
    # cv2.imshow("sssss", maxInColumns)

    print("colAssignment boyut." + str(colAssignment.ndim))



    for i in range(0, 10, 1):
        new_name = "saglikli" + str(i) + str(mSize) + ".png"
        new_colAssignment = np.where(colAssignment == i, colAssignment, -1)
        plt.imsave(new_name, new_colAssignment)

    """for l in range(0,10,1):
        if(l==0 or l==1 or l==2 or l==7 or l==8 or l==9):
            im = cv2.imread("saglikli"+str(l)+".png", 0)  # dosyayi oku
            plt.subplot(2, 2, 1)
            plt.imshow(im, cmap="gray")

            # inverse ikili esik degeri
            _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV)
            plt.subplot(2, 2, 2)
            plt.imshow(im, cmap="gray")

            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones([30, 15]))  # gurultuden kurtul
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

            plt.imsave("CCA"+str(l)+".png", comp)
            cv2.waitKey(0)
        else:
            im = cv2.imread("saglikli" + str(l) + ".png", 0)  # dosyayi oku
            plt.subplot(2, 2, 1)
            plt.imshow(im, cmap="gray")

            # inverse ikili esik degeri
            _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV)
            plt.subplot(2, 2, 2)
            plt.imshow(im, cmap="gray")

            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones([15, 30]))  # gurultuden kurtul
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
            c = plt.imshow(comp, cmap="nipy_spectral", label="renk2")
            bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.colorbar(c, boundaries=bounds)
            plt.title('Connected Component Analysis', fontweight="bold")
            print(comp)

            plt.imsave("CCA" + str(l) + ".png", comp)
            cv2.waitKey(0)"""

    for x in range(0, 10, 1):
        img = cv2.imread('saglikli' + str(x) + str(mSize) + '.png', 0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        retval, labels = cv2.connectedComponents(img)

        ##################################################
        ts = time.time()
        num = labels.max()

        N = 600
        for i in range(1, num + 1):
            pts = np.where(labels == i)
            if len(pts[0]) < N:
                labels[pts] = 0
        for y in range(1, num + 1):
            pts = np.where(labels == y)
            if len(pts[0]) > 2500:
                labels[pts] = 0

        print("Time passed: {:.3f} ms".format(1000 * (time.time() - ts)))
        # Time passed: 4.607 ms

        ##################################################

        # Map component labels to hue val
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        # cv2.imshow('labeled'+x+'.png', labeled_img)
        cv2.imwrite('labeled' + str(x) + str(mSize) + '.png', labeled_img)
        cv2.waitKey(0)

    img1 = cv2.imread('labeled0' + str(mSize) + '.png')
    img2 = cv2.imread('labeled1' + str(mSize) + '.png')
    dst = cv2.addWeighted(img1, 1, img2, 1, 0)
    img3 = cv2.imread('labeled2' + str(mSize) + '.png')
    dst1 = cv2.addWeighted(dst, 1, img3, 1, 0)
    img4 = cv2.imread('labeled3' + str(mSize) + '.png')
    dst2 = cv2.addWeighted(dst1, 1, img4, 1, 0)
    img5 = cv2.imread('labeled4' + str(mSize) + '.png')
    dst3 = cv2.addWeighted(dst2, 1, img5, 1, 0)
    img6 = cv2.imread('labeled5' + str(mSize) + '.png')
    dst4 = cv2.addWeighted(dst3, 1, img6, 1, 0)
    img7 = cv2.imread('labeled6' + str(mSize) + '.png')
    dst5 = cv2.addWeighted(dst4, 1, img7, 1, 0)
    img8 = cv2.imread('labeled7' + str(mSize) + '.png')
    dst6 = cv2.addWeighted(dst5, 1, img8, 1, 0)
    img9 = cv2.imread('labeled8' + str(mSize) + '.png')
    dst7 = cv2.addWeighted(dst6, 1, img9, 1, 0)
    img10 = cv2.imread('labeled9' + str(mSize) + '.png')
    dst8 = cv2.addWeighted(dst7, 1, img10, 1, 0)

    # cv2.imshow('Blended Image',dst8)
    plt.imsave('labeled10' + str(mSize) + '.png', dst8)

    YonelimYuzdeler = []
    for x in range(0, 11, 1):
        img = Image.open("labeled" + str(x) + str(mSize) + '.png')
        for i in range(0, img.size[0] - 1):
            for j in range(0, img.size[1] - 1):
                pixelColorVals = img.getpixel((i, j));
                redPixel = 255 - pixelColorVals[0];  # Negate red pixel
                greenPixel = 255 - pixelColorVals[1];  # Negate green pixel
                bluePixel = 255 - pixelColorVals[2];  # Negate blue pixel
                img.putpixel((i, j), (redPixel, greenPixel, bluePixel));
        img.save('new' + str(mSize) + '.png')

        img = cv2.imread('new' + str(mSize) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 244, 255, cv2.THRESH_BINARY_INV)
        kernal = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(thresh, kernal, iterations=2)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = len(contours) - 1
        YonelimYuzdeler.append(int(objects))
        text = "Yonelim:" + str(objects)
        cv2.putText(dilation, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 0, 159), 1)
        # cv2.imshow('Dilation', dilation)
        print(text)
        cv2.waitKey(0)
    objects
    print("Görüntüdeki Collagen Bundle Sayısı: ", objects)
    c = 36
    for y in range(0, 10, 1):
        temp = int(float(YonelimYuzdeler[y] / objects) * 100)
        print("Yonelim", y, "'daki Collagen Bundles sayısı toplam Bundles sayısının %", temp,
              "'i kadardır ve Y eksenine göre", c, "derecelik açıya aittir.")
        c += 36
    cv2.destroyAllWindows()

    kernel = np.ones((10, 10), 'uint8')
    # dst8
    dilate_img = cv2.dilate(dst8, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilate_img)
    cv2.imwrite('Dilated Image' + str(mSize) + '.png', dilate_img)
    cv2.waitKey(0)

    img = img_as_float(io.imread('Dilated Image' + str(mSize) + '.png', as_gray=True))
    for i in range(0, 512):
        for k in range(0, 512):
            if (img[i][k] > 0):
                img[i][k] = 1
    cv2.imshow("BlackandWhite", img)
    plt.imsave('Dilated Image With Defauld_BlackandWhite' + str(mSize) + '.png', img)
    cv2.waitKey(0)
    img2 = img_as_float(io.imread('face_1_frame-0' + str(mSize) + '.png', as_gray=True))
    result = img2 * img
    plt.imsave('Dilated Image With Defauld' + str(mSize) + '.png', result)
    cv2.imshow("result", result)
    cv2.waitKey(0)

    img = cv2.imread('Dilated Image With Defauld' + str(mSize) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 60  # 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 70  # 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow("lines_edges", lines_edges)
    plt.imsave('lines_edges' + str(mSize) + '.png', lines_edges)
    cv2.waitKey(0)
#Normal(10,40,5)


def DilateExampe(mSize,b,c):
    dst8=cv2.imread("labeled1040.png")
    kernel = np.ones((10, 10), 'uint8')
    print("dilate öncesi boyut." + str(dst8.ndim))
    dilate_img = cv2.dilate(dst8, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilate_img)
    cv2.imwrite('Dilated Image' + str(mSize) + '.png', dilate_img)  #.görüntüyü okuyoruz
    test33=np.array(dilate_img)  #boyutunu görebilmek için bir arraya çevirdim.
    print("dilate sonrası boyut." +str(test33.ndim))
    cv2.waitKey(0)
    img = cv2.imread('Dilated Image' + str(mSize) + '.png',)
    cv2.imshow('img', img)
    test = img[:,:,0]

    cv2.imshow("BlackandWhite", test)
    #plt.imsave('Dilated Image With Defauld_BlackandWhite' + str(mSize) + '.png', img)
    cv2.waitKey(0)
    img2 = img_as_float(io.imread('nothasta_yönelim.png', as_gray=True))
    result = test*img2
    #plt.imsave('Dilated Image With Defauld' + str(mSize) + '.png', result)
    cv2.imshow("result", result)
    cv2.waitKey(0)

    img = cv2.imread('Dilated Image With Defauld' + str(mSize) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 60  # 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 70  # 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow("lines_edges", lines_edges)
    plt.imsave('lines_edges' + str(mSize) + '.png', lines_edges)
    cv2.waitKey(0)
#DilateExampe(40,1,2)

### V2 bazı özellikler silindi.

def Normal2(nBands,mSize,mzSize,imgName):

    nBands = nBands
    mSize = mSize
    mzSize = mzSize
    filts = np.ones((nBands, 1))

    def polygon2mask(image_shape, polygon):
        mask = skimage.draw.polygon2mask(image_shape, polygon)
        return mask

    def mypolygon_perimeter(X, Y, image_shape):
        rr, cc = skimage.draw.polygon_perimeter(X, Y, image_shape)
        return rr, cc

    # read images
    imgs = cv2.imreadmulti(imgName, flags=cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)[1]

    for i, img in enumerate(imgs):
        filename = f'face_1_frame-{i}' + str(mSize) + '.png'
        print(f"Processing frame {i} into file {filename}")
        # normalize image to 8-bit range
        img_norm = exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)).astype(np.uint8)
        print(img_norm)

        cv2.imwrite(filename, img_norm)

        # display normalized image
        ###cv2.imshow('normalized', img_norm)

    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return (x, y)

    filts = np.ones((nBands, 1))
    # print("filts değerli :")
    # print(filts)
    dirs = np.zeros((2, nBands))
    # print("dirs değerleri :")
    # print(dirs)
    theta = np.arange(0, nBands)
    theta = theta * (math.pi / nBands)
    # print("theta değerleri :")
    # print(theta)
    rho = np.ones((1, nBands))
    # print("rho değerleri :")
    # print(rho)
    [X, Y] = pol2cart(theta, rho)
    # image_shape = (mSize, mzSize)
    # XX = X[0]
    # YY = Y[0]
    # polygon = np.array([XX, YY]).T ## çalışıyor
    # print("[X,Y] değerleri")
    # print([X, Y])
    # print(" Sadece X değerleri :")
    # print(X)
    # print(" Sadece y değerleri :")
    # print(Y)
    # print()
    # print()
    ang = math.atan(mzSize / mSize)
    # print("ang değerleri")
    # print(ang)
    dist = math.ceil((math.sqrt(math.pow((mzSize / 2), 2) + math.pow((mSize / 2), 2))))
    # print("dist değerleri")
    # print(dist)
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
        # print("ang1 : " + str(ang1))
        ang2 = (kk + ang)
        # print("ang1 : " + str(ang2))
        theta = [ang1, ang2, ang1, ang2, ang1]
        # print(" for içinde bulunan theta")
        # print(theta)
        rho = np.array([1, 1, -1, -1, 1]) * dist
        # print("rho")
        # print(rho)
        [X, Y] = pol2cart(theta, rho)
        # print(" For içinde bulunan X 'in Değişmeden öncesi ")
        # print(X)
        X22 = X + math.ceil(mSize / 2);
        # print("Değişen X ")
        # print(X22)
        # print(" For içinde bulunan Y 'nin Değişmeden öncesi ")
        # print(Y)
        Y22 = Y + math.ceil(mSize / 2);
        # print("Değişen Y ")
        # print(Y22)
        polygon22 = np.array([X22, Y22]).T
        # print("polygon22 deperi")
        # print(polygon22)
        image_shape = (mSize, mSize)
        mask = polygon2mask(image_shape, polygon22)
        count = np.count_nonzero(mask)
        # print("mask edilmiş hali")
        # print(mask)
        # print("count değeri")
        # print(count)
        mask_int = mask.astype(int)
        count = np.count_nonzero(mask_int)
        # print("count değeri")
        # print(count)
        # print("1,0 hali")
        # print(mask_int)
        a = mask_int / count;
        # print(a)
        FilteringResutls.append(a)
        # plt.imshow(mask)
        # plt.show()

    ConvolResults = []

    for value in range(0, len(FilteringResutls)):
        img_gaussian_noise = img_as_float(io.imread('face_1_frame-0' + str(mSize) + '.png', as_gray=True))
        img_salt_pepper_noise = img_as_float(io.imread('face_1_frame-0' + str(mSize) + '.png', as_gray=True))
        img = img_salt_pepper_noise
        kernel = FilteringResutls[value]

        conv_using_cv2 = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        ConvolResults.append(conv_using_cv2)

        conv_using_scipy = convolve2d(img, kernel, mode='same')

        conv_using_scipy2 = convolve(img, kernel, mode='constant', cval=0.0)

        filename_1 = f"nothasta bireye Uygulanan Filtre-{value}-{str(mSize)}.png"
        filename_2 = f"conv_using_scipy-{value}."
        filename_3 = f"conv_using_scipy2-{value}"
        # cv2.imshow("Original", img)
        # cv2.imshow(filename_1, conv_using_cv2)
        # cv2.imshow(filename_2, conv_using_scipy)
        # cv2.imshow(filename_3, conv_using_scipy2)
        # plt.imsave(filename_1, conv_using_cv2 , cmap='gray')

    # cv2.waitKey(0)
    selVals, colAssignment = np.array(ConvolResults).max(0), np.array(ConvolResults).argmax(0)
    # maxInColumns = np.amax(ConvolResults, axis=0)
    np.set_printoptions(threshold=sys.maxsize)
    # cv2.imshow("nothasta_Max_Responce.png", selVals)
    plt.imsave("maxResponce" + str(mSize) + ".png", selVals, cmap='gray')

    """
    for i in range(0,512):
        for k in range(0,512):
            if(0<=colAssignment[i][k]<=2):
                colAssignment[i][k]=1
            if (3 <= colAssignment[i][k]<=5):
                colAssignment[i][k] =4
            if (6 <= colAssignment[i][k]<9):
                colAssignment[i][k] =7
    """
    print('matris2', colAssignment)
    # plt.imsave("test.png",colAssignment)
    plt.imsave("nothasta_yönelim" + str(mSize) + ".png", colAssignment)
    c = plt.imshow(colAssignment)
    bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.colorbar(c, boundaries=bounds)
    plt.title('Yönelim Haritası', fontweight="bold")
    ###plt.show()
    # cv2.imshow("sssss", maxInColumns)





    for i in range(0, 10, 1):
        new_name = "saglikli" + str(i) + str(mSize) + ".png"
        new_colAssignment = np.where(colAssignment == i, colAssignment, -1)
        plt.imsave(new_name, new_colAssignment)

    """for l in range(0,10,1):
        if(l==0 or l==1 or l==2 or l==7 or l==8 or l==9):
            im = cv2.imread("saglikli"+str(l)+".png", 0)  # dosyayi oku
            plt.subplot(2, 2, 1)
            plt.imshow(im, cmap="gray")

            # inverse ikili esik degeri
            _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV)
            plt.subplot(2, 2, 2)
            plt.imshow(im, cmap="gray")

            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones([30, 15]))  # gurultuden kurtul
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

            plt.imsave("CCA"+str(l)+".png", comp)
            cv2.waitKey(0)
        else:
            im = cv2.imread("saglikli" + str(l) + ".png", 0)  # dosyayi oku
            plt.subplot(2, 2, 1)
            plt.imshow(im, cmap="gray")

            # inverse ikili esik degeri
            _, im = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY_INV)
            plt.subplot(2, 2, 2)
            plt.imshow(im, cmap="gray")

            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones([15, 30]))  # gurultuden kurtul
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
            c = plt.imshow(comp, cmap="nipy_spectral", label="renk2")
            bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            plt.colorbar(c, boundaries=bounds)
            plt.title('Connected Component Analysis', fontweight="bold")
            print(comp)

            plt.imsave("CCA" + str(l) + ".png", comp)
            cv2.waitKey(0)"""

    for x in range(0, 10, 1):
        img = cv2.imread('saglikli' + str(x) + str(mSize) + '.png', 0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        retval, labels = cv2.connectedComponents(img)

        ##################################################
        ts = time.time()
        num = labels.max()

        N = 600
        for i in range(1, num + 1):
            pts = np.where(labels == i)
            if len(pts[0]) < N:
                labels[pts] = 0
        for y in range(1, num + 1):
            pts = np.where(labels == y)
            if len(pts[0]) > 2500:
                labels[pts] = 0

        print("Time passed: {:.3f} ms".format(1000 * (time.time() - ts)))
        # Time passed: 4.607 ms

        ##################################################

        # Map component labels to hue val
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0
        labeled_img=labeled_img[:,:,0]
        print("labeled_img size",labeled_img.ndim)
        # cv2.imshow('labeled'+x+'.png', labeled_img)
        cv2.imwrite('labeled' + str(x) + str(mSize) + '.png', labeled_img)
        ###cv2.waitKey(0)

    img1 = cv2.imread('labeled0' + str(mSize) + '.png')
    img2 = cv2.imread('labeled1' + str(mSize) + '.png')
    dst = cv2.addWeighted(img1, 1, img2, 1, 0)
    img3 = cv2.imread('labeled2' + str(mSize) + '.png')
    dst1 = cv2.addWeighted(dst, 1, img3, 1, 0)
    img4 = cv2.imread('labeled3' + str(mSize) + '.png')
    dst2 = cv2.addWeighted(dst1, 1, img4, 1, 0)
    img5 = cv2.imread('labeled4' + str(mSize) + '.png')
    dst3 = cv2.addWeighted(dst2, 1, img5, 1, 0)
    img6 = cv2.imread('labeled5' + str(mSize) + '.png')
    dst4 = cv2.addWeighted(dst3, 1, img6, 1, 0)
    img7 = cv2.imread('labeled6' + str(mSize) + '.png')
    dst5 = cv2.addWeighted(dst4, 1, img7, 1, 0)
    img8 = cv2.imread('labeled7' + str(mSize) + '.png')
    dst6 = cv2.addWeighted(dst5, 1, img8, 1, 0)
    img9 = cv2.imread('labeled8' + str(mSize) + '.png')
    dst7 = cv2.addWeighted(dst6, 1, img9, 1, 0)
    img10 = cv2.imread('labeled9' + str(mSize) + '.png')
    dst8 = cv2.addWeighted(dst7, 1, img10, 1, 0)
    print("dst8 size", dst8.ndim)
    dst8 = dst8[:, :, 0]
    print("dst8 size2", dst8.ndim)
    # cv2.imshow('Blended Image',dst8)
    plt.imsave('labeled10' + str(mSize) + '.png', dst8)



    kernel = np.ones((8, 8), 'uint8')
    # dst8
    dilate_img = cv2.dilate(dst8, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilate_img)
    cv2.imwrite('Dilated Image' + str(mSize) + '.png', dilate_img)
    ###cv2.waitKey(0)
    print("Dilated size", dilate_img.ndim)
    img = img_as_float(io.imread('Dilated Image' + str(mSize) + '.png', as_gray=True))
    for i in range(0, 512):
        for k in range(0, 512):
            if (img[i][k] > 0):
                img[i][k] = 1

    cv2.imshow("BlackandWhite", img)
    plt.imsave('Dilated Image With Defauld_BlackandWhite' + str(mSize) + '.png', img)
    ###cv2.waitKey(0)
    img2 = img_as_float(io.imread('nothasta_yönelim.png'))
    nothasta_yönelim2d=img2[:, :, 0]

    cv2.imshow("nothasta_yönelim2d", nothasta_yönelim2d)
    result = nothasta_yönelim2d * img

    plt.imsave('Dilated Image With Defauld' + str(mSize) + '.png', result)
    cv2.imshow("result", result)




    ###cv2.waitKey(0)

    img = cv2.imread('Dilated Image With Defauld' + str(mSize) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 36  # angular resolution in radians of the Hough grid
    threshold = 20  # 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # 50  # minimum number of pixels making up a line
    max_line_gap = 7  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 2  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow("lines_edges", lines_edges)
    plt.imsave('lines_edges' + str(mSize) + '.png', lines_edges)
    cv2.waitKey(0)
#Normal2(10,40,5,"1.tif")


def DilateV2(mSize):

    dst8 = cv2.imread('labeled1040.png')
    nothasta_yönelim2d = img_as_float(io.imread('nothasta_yönelim.png'))
    dst8=dst8[:,:,0]
    nothasta_yönelim2d = nothasta_yönelim2d[:, :, 0]
    print("nothasta_yönelim2d size",nothasta_yönelim2d.ndim)
    print("dst8 size", dst8.ndim)


    kernel = np.ones((10, 10), 'uint8')
    dilate_img = cv2.dilate(dst8, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilate_img)
    #cv2.imwrite('Dilated Image' + str(mSize) + '.png', dilate_img)
    cv2.waitKey(0)
    print("Dilated size 2 -- ", dilate_img.ndim)


    ForZeroOne = dilate_img
    for i in range(0, 512):
        for k in range(0, 512):
            if (ForZeroOne[i][k] > 0):
                ForZeroOne[i][k] = 1
    cv2.imshow("ForZeroOne", ForZeroOne)



    result = ForZeroOne * nothasta_yönelim2d

    plt.imsave('Dilated Image With Defauld' + str(mSize) + '.png', result)
    '''
    cv2.imshow("result", result)
    cv2.waitKey(0)
'''
    img = cv2.imread('Dilated Image With Defauld' + str(mSize) + '.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 60  # 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 70  # 50  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow("lines_edges", lines_edges)
    #plt.imsave('lines_edges' + str(mSize) + '.png', lines_edges)
    cv2.waitKey(0)
#DilateV2(40)


#https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/
def Test(a):
    import cv2 as cv
    from math import atan2, cos, sin, sqrt, pi
    import numpy as np

    def drawAxis(img, p_, q_, color, scale):
        p = list(p_)
        q = list(q_)

        ## [visualization1]
        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
        ## [visualization1]

    def getOrientation(pts, img):
        ## [pca]
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

        # Store the center of the object
        cntr = (int(mean[0, 0]), int(mean[0, 1]))
        ## [pca]

        ## [visualization]
        # Draw the principal components
        cv.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
              cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
              cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
        drawAxis(img, cntr, p1, (255, 255, 0), 1)
        drawAxis(img, cntr, p2, (0, 0, 255), 5)

        angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
        ## [visualization]

        # Label with the rotation angle
        label = "" + str(-int(np.rad2deg(angle)) - 90) + " degrees"
        textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
        cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        return angle

    # Load the image
    img = cv.imread("Dilated Image With Defauld_BlackandWhite40.png")

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)

    cv.imshow('Input Image', img)

    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv.threshold(gray, 10, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 520 or 10000 < area:
            continue

        # Draw each contour only for visualisation purposes
        cv.drawContours(img, contours, i, (0, 0, 255), 2)

        # Find the orientation of each shape
        getOrientation(c, img)

    cv.imshow('Output Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the output image to the current directory
    cv.imwrite("output_img.jpg", img)
Test("work")


def Test2():
    import cv2
    import numpy as np
    import math

    # Read image
    src = cv2.imread('Dilated Image With Defauld_BlackandWhite40.png')
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    # Find contours in image
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # Draw skeleton of banana on the mask
    img = gray.copy()
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 5, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size: done = True
    kernel = np.ones((2, 2), np.uint8)
    skel = cv2.dilate(skel, kernel, iterations=1)
    skeleton_contours, _ = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_skeleton_contour = max(skeleton_contours, key=cv2.contourArea)
    # Extend the skeleton past the edges of the banana
    points = []
    for point in largest_skeleton_contour: points.append(tuple(point[0]))
    x, y = zip(*points)
    z = np.polyfit(x, y, 7)
    f = np.poly1d(z)
    x_new = np.linspace(0, img.shape[1], 300)
    y_new = f(x_new)
    extension = list(zip(x_new, y_new))
    img = src.copy()
    for point in range(len(extension) - 1):
        a = tuple(np.array(extension[point], int))
        b = tuple(np.array(extension[point + 1], int))
        cv2.line(img, a, b, (0, 0, 255), 1)
        cv2.line(mask, a, b, 255, 1)
    mask_px = np.count_nonzero(mask)

    # Find the distance between points in the contour of the banana
    # Only look at distances that cross the mid line
    def is_collision(mask_px, mask, a, b):
        temp_image = mask.copy()
        cv2.line(temp_image, a, b, 0, 2)
        new_total = np.count_nonzero(temp_image)
        if new_total != mask_px:
            return True
        else:
            return False

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    distances = []
    for point_a in cnt[:int(len(cnt) / 2)]:
        temp_distance = 0
        close_distance = img.shape[0] * img.shape[1]
        close_points = (0, 0), (0, 0)
        for point_b in cnt:
            A, B = tuple(point_a[0]), tuple(point_b[0])
            dist = distance(tuple(point_a[0]), tuple(point_b[0]))
            if is_collision(mask_px, mask, A, B):
                if dist < close_distance:
                    close_points = A, B
                    close_distance = dist
        cv2.line(img, close_points[0], close_points[1], (234, 234, 123), 1)
        distances.append((close_distance, close_points))
        cv2.imshow('img', img)
        cv2.waitKey(1)
        max_thickness = max(distances)
        a, b = max_thickness[1][0], max_thickness[1][1]
        cv2.line(img, a, b, (0, 255, 0), 4)
        print("maximum thickness = ", max_thickness[0])
#Test2()

def Test3():
    im = cv2.imread('Dilated Image With Defauld_BlackandWhite40.png')

