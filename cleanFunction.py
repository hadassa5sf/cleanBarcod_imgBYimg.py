"""
First iteration for:
 > external contour
 > thresh 130
 > dis = 0.14
 > nP = 7
Second iteration for:
-----ONE OF THE TEST IS TRUE: DO AGAIN---
 > internal contour
 > thresh 115
 > dis = 0.2
 > nP = 3
Thrid iteration for:
-----ONE OF THE TEST IS TRUE: DO AGAIN---
 > internal contour
 > thresh 130
 > dis = 0.2
 > nP = 3
 Fourth iteration for:
 -----ONE OF THE TEST IS TRUE: DO AGAIN---
 > external contour
 > thresh 160
 > dis = 0.14
 > nP = 7

 ---THE TEST IS
 test1 > len(dis)==0
 test2 > min(dist)>= IDrange
 test3 > len(dist) > 5:
        if iteration == 1 and (min(dist) >= 0.065 and (max(dist) - min(dist))>0.25 ):
 tset4 > if goodPlace(shapesContours[i]): --v
 test5 > if pertOfMin(i, index_min3Con) -->len(chosenContour)==0:
"""
import numpy as np
import cv2
import os
import copy

# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


t_array = []  # complex type
s_arrays = []
shapesContours = []

#IN FUNCTION match
# The range of shapes allowed
#IDrange = 0.14
# nume of point
#nP = 7
def upDate(iteration):
    nP=0
    IDrange = 0
    if iteration==1:
        IDrange = 0.14
        nP = 7
    elif iteration==4 or iteration == 5:
        IDrange = 0.15
        nP = 7
    elif iteration== 2 or iteration == 3:
        IDrange = 0.2
        nP = 3
    return nP, IDrange


def goodPlace(contour):
    """
    check if the contour is not in the side
    :param contour:
    :return:
    """
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    if y<=0 or x<=0:
        return False
    elif y+h >=2016 or x+w>=3840:
        return False
    return True

def yellowToBlack(img_y):
    hsv_frame = cv2.cvtColor(img_y, cv2.COLOR_BGR2HSV)
    low = np.array([22, 50, 0])
    high = np.array([255, 255, 204])
    mask = cv2.inRange(hsv_frame, low, high)
    # result = cv2.bitwise_and(img, img, mask=mask)
    run = img_y
    run[mask != 0] = [0, 0, 0]
    # plt.imshow(img)
    # plt.show()
    lower2 = np.array([22, 93, 0], dtype="uint8")
    upper2 = np.array([45, 255, 255], dtype="uint8")
    # create the mask and use it to change the colors
    hsv_frame2 = cv2.cvtColor(run, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame2, lower2, upper2)
    run[mask != 0] = [0, 0, 0]
    # plt.imshow(img)
    # plt.show()
    return run


def filterBy3Min(myDis , IDrang):
    """
    filter by:
    save the 4 index of the smaller distance
    that is also small then IDrange
    if there are no 4 dis in the list put None

    :return: the index
    """
    min = [10, 10, 10]
    index = [-1, -1, -1]

    i = -1
    for item in myDis:
        i = i + 1
        if item < IDrang:
            if item <= max(min):
                del_i = min.index(max(min))
                min[del_i] = item
                index[del_i] = i

    for item in min:
        if item == 10:
            no_i = min.index(10)
            min[no_i] = None
            index[no_i] = None

    return index

def pertOfMin(index , listmin):
    """
    if we will want to save this contour to be draw
    check by testing if his index his part of the min index
    :return: boolean
    """
    for t in listmin:
        if index==t:
            return True

    return False

def getContours(img,iteration):
    """
    get a img and fix it to be white shape on black background.
    For cv2.findcontoure
    :param img: BGR
    :return: shapes countoure
    """
    nP, IDrange = upDate(iteration)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retvalth, imgthreshold = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((nP, nP), np.uint8)
    imgdilation = cv2.dilate(imgthreshold, kernel, iterations=2)
    contours = []
    # two vertion of cv2 for findcontours-> (old vertion): imgcontours, contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if iteration == 2 :
        contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif  iteration == 3:
        contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ##imgcontours, contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def getCV(file1, type, imgOri,iteration):
    """
    move each shape to the center .
    for the shapes it is make an array that each value mach to a signal shape countour
    :param type: data shapes or temp
    """
    del s_arrays[:]
    del shapesContours[:]

    if type == 'shapes':
        spContours = getContours(imgOri,iteration)
        file1.write(str(len(spContours)) + '\n')
        print(len(spContours))
        spContours = clean_Con(spContours)
        file1.write(str(len(spContours)) + '\n')
        print(len(spContours))
        cv2.drawContours(imgOri, spContours, -1, (0, 255, 128), 5)
        # del s_arrays[:]
        # for each shape
        for cons in spContours:
            sampleComVector = []
            x, y, w, h = cv2.boundingRect(cons)
            cv2.rectangle(imgOri, (x, y), (x + w, y + h), (100, 100, 100), 1)

            # move the points to center
            for point in cons:
                sampleComVector.append(complex(point[0][0] - x, (point[0][1] - y)))
            # sampleComVectors store CV of all testees contours
            s_arrays.append(sampleComVector)
            # sampleContours store all testees contours, same order with sampleComVectors
            shapesContours.append(cons)

    elif type == 'temp':
        # Automatically find templete contour
        templetTrue = imgOri
        tpContour = getContours(templetTrue,iteration)
        for contour in tpContour:
            x, y, w, h = cv2.boundingRect(contour)
            #
            for point in contour:
                # -x and -y are to make left and upper boundry start from 0
                t_array.append(complex(point[0][0] - x, (point[0][1] - y)))


def FourierDescriptor(type):
    """
    calc the FourierDescriptor for contoure array of complex - vector
    :param type: templet or shapes
    :return: for temple it is an array
             for shapes it is an array of array
    """
    if type == 'temp':
        return np.fft.fft(t_array)
    elif type == 'shapes':
        FDs = []
        for sampleVector in s_arrays:
            sampleFD = np.fft.fft(sampleVector)
            FDs.append(sampleFD)

        return FDs


def Normalised(fourierDesc ,iteration):
    """
    this function will mack the fourier array to be unit
    independent by scale placement and rotation
    :param fourierDesc: array
    :return:
    """
    nP, IDrange = upDate(iteration)
    # it make FD invariant to rotaition and start point
    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = np.absolute(value)

    # Scaling divaide by first element
    firstVal = fourierDesc[0]

    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = value / firstVal

    # transInvariant
    fourierDesc = fourierDesc[1:len(fourierDesc)]

    return fourierDesc[:nP]


def match(file1, tpFD, spFDs, imgOri ,iteration):
    """
    check similar
    :param tpFD: templet fft.contour
    :param spFDs: list of fft.contour
    :return:
    """
    nP, IDrange = upDate(iteration)
    # iteration2_imgBW = copy.copy(imgOri)
    tpFD = Normalised(tpFD, iteration)
    # dist store the distance, same order as spContours
    dist = []
    chosenContour = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    # calc the dis
    for spFD in spFDs:
        spFD = Normalised(spFD, iteration)
        # Calculate Euclidean distance between templete and testee
        dist.append(np.linalg.norm(np.array(spFD) - np.array(tpFD)))

    # check if we need a new calculation contour because no good Tresh
    if len(dist) == 0:
        bool = 'Fals ,re-do a new Tresh'
        file1.write('no dis at all\n')
        return bool , None

    elif min(dist)>= IDrange:
        bool = 'Fals ,re-do a new Tresh'
        file1.write('the smaller dis is: '+ str(min(dist)) + "\n")
        return bool, None

    elif len(dist) > 5:
        if iteration == 1 and (min(dist) >= 0.065 and (max(dist) - min(dist))>0.25 ):
            bool = 'Fals ,re-do a new Tresh'
            file1.write('To many contour:  ' + str(len(dist)) + "\n")
            file1.write('the smaller dis is: ' + str(min(dist)) + "\n" +'the range dis is:'+str(max(dist) - min(dist)))
            return bool, None




    # get the index of the min 3 contour that we should want to save
    index_min3Con = filterBy3Min(dist, IDrange)

    for i in range(len(dist)):
        x, y, w, h = cv2.boundingRect(shapesContours[i])
        # Draw distance on image
        distText = str(round(dist[i], 2))
        cv2.putText(imgOri, distText, (x, y - 8), font, 2, (0, 0, 255), 2, cv2.LINE_AA)


        # if distance is less than IDrange, it will be good match.
        if dist[i] < IDrange:
            if goodPlace(shapesContours[i]):
                #save only if it is take part of the min distance
                if pertOfMin(i, index_min3Con):
                    chosenContour.append(shapesContours[i])
                # anywhere draw the all good range dis
                cv2.rectangle(imgOri, (x - nP, y - nP), (x + w + nP, y + h + nP), (128, 0, 128), 4)
                file1.write(str(dist[i]) + " **\n")
                print(dist[i], "**")

        else:
            file1.write(str(dist[i]) + '\n')
            print(dist[i])
    #if all contour was in out of the image:
    if len(chosenContour)==0:
        bool = 'Fals ,re-do a new Tresh'
        file1.write('no con at all in the good plase of the image\n')
        return bool, None

    bool = 'good'
    return bool, chosenContour


def drawOnImg(file1, imgBW, chosenDis,iteration):
    """
    draw the violet rectangel on the img
    chosenDis: list of array[contour][dis]
    :return:
    """
    nP, IDrange = upDate(iteration)
    chosenContour = []
    x, y, w, h = cv2.boundingRect(shapesContours[len(chosenDis) - 1])
    cv2.rectangle(imgBW, (x - nP, y - nP), (x + w + nP, y + h + nP), (128, 0, 128), 4)
    chosenContour.append(shapesContours[len(chosenDis) - 1])
    file1.write(str(chosenDis[len(chosenDis) - 1]) + " **\n")
    print(chosenDis[len(chosenDis) - 1], "**")

    file1.write(str(chosenDis[len(chosenDis) - 1]) + '\n')
    print(chosenDis[len(chosenDis) - 1])


def clean_Con(list):
    cleanUp = []
    # clean all the very small list
    for i in range(len(list) - 1):
        if len(list[i]) > 600:
            cleanUp.append(list[i])
    return cleanUp


def offset(img, y_s, y_e, x_s, x_e):
    """
    get offset to the barcod for get the croping img
    :return:
    """
    if y_s < 0:
        y_s = 0
    if y_e > img.shape[0]:
        y_e = img.shape[0]
    if x_s < 0:
        x_s = 0
    if x_e > img.shape[1]:
        x_e = img.shape[1]
    new_img = img[y_s:y_e, x_s:x_e]
    return new_img


def crop(image, countours, num_img, path_des):
    """
    save the smale image of the contour in the out folder

    """
    i = 0
    if len(countours) == 0:  # we d'ont fine a contour save the all image
        path = path_des
        cv2.imwrite(os.path.join(path, 'Barcod') + '_' + 'num' + str(i) + '_' + 'img' + num_img, image)
        i = i + 1
    for cntr in countours:
        # creates an approximate rectangle around contour
        x, y, w, h = cv2.boundingRect(cntr)
        of = 50
        # pulls crop out of the image based on dimensions
        # new_img = image[y-of:y + h+of, x-of:x + w+of]
        new_img = offset(image, y - of, y + h + of, x - of, x + w + of)

        path = path_des
        cv2.imwrite(os.path.join(path, 'Barcod') + '_' + 'num' + str(i) + '_' + 'img' + num_img, new_img)

        i = i + 1
