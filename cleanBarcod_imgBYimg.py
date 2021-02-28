"""
date 22/01/2021

this code will run image by image and not folder by folder
it will give as the opportunity to rework on the original image for other iteration
- in case of bed detection.

list:
- get img and change to B&W in Tresh = 130
- get contour in test distance of IDrange = 0.14

    before drawing the result check
    >> if there are no contour:
        -del the B&W image
        -del the copy RGB drawing img
        -redo the step, with one change of Tresh = 160
    >> if there are to many fined contours
        save the two smaller dis and del the rest
    in the 3 iteration change the contoure function to:
                contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    and Tresh = 115 , np = 3 , IDrange = 0.6

    then draw,
    make sure that it draw only on a copy of RGB image

help function : fine the min ,save no too mach big contour , clear, redo, a lot of matematic and algoritem- good job:)
"""

from cleanFunction import *
import glob
import timeit
import sys
import os

#static params
TRESH = 130
allRGBimages = []
allBW = []
allRGBimagesW = []
names_img = []

#get the folder src and barcod
path_src = 'C:\\Users\\hadas\\Desktop\\spinframe_onWork\\Barcod\\script_Code\\3_iteration\\src_test' #sys.argv[1]
path_des = 'C:\\Users\\hadas\\Desktop\\spinframe_onWork\\Barcod\\script_Code\\3_iteration\\out' #sys.argv[2]

#open timer
start = timeit.default_timer()

#part 1
#read all the image in tha data sourc folder
for img in glob.glob(path_src+'/*'):
    allRGBimages.append(cv2.imread(img))  # list of RGB images for final drawing
    allRGBimagesW.append(cv2.imread(img)) # list of RGB images for working, we will change to B&W
    n = os.path.basename(img) # get the image name - 'name.jpg'
    names_img.append(n) # list of name same index like the image list

#size of the image and potensial croping from size
h_img, w_img, c_img = allRGBimagesW[0].shape
right = 0 #R = 460 , right crop
left = 0  #L = 500 , left crop
#creat new folder of binary images
os.mkdir(path_src + '/BW')
#path for save the B&W to the folder
pathBW = path_src + '/BW'
#creat new folder of detect images
os.mkdir(path_src+'/detected')
#creat a txt file1 for write analyse
completeName = os.path.join(path_src+'/detected', 'analyze'+".txt")
file1 = open(completeName, "w")

#read the Template
templetTrue = cv2.imread('C:\\Users\\hadas\\Desktop\\spinframe_onWork\\Barcod\\script_Code\\3_iteration\\MyTrueTempletRectangle.png')
getCV(file1, 'temp', templetTrue,0)
tpFD = FourierDescriptor('temp')

#part2
#img by img , while: (flag = start ->turn to-> flag =finish)
#change to BW than find contour, if: the test check is bad del the img and re-do, else: flag = finish

i=0
imgStatus = False # False == on work / True == finish
#for next iteration we need to reread the image so: cv2.imread((path_src+'\\'+names_img[0]))
for imgO in allRGBimagesW:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',i)
    imgStatus = False
    iteration = 0
    TRESH = 130
    # a single img untill we get the good contour
    while(imgStatus == False):

        iteration = iteration + 1

        if iteration==2:
            file1.write('\n>>> @@: ITERATION 2 with cv2.findContours(imgdilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) Tresh = 115:' + '\n')
            #read a clean img
            img0 = cv2.imread((path_src+'\\'+names_img[i]))
            #chang the tresh to TRESH = 160
            TRESH = 115

        if iteration==4:
            file1.write('\n>>> @@: ITERATION 4 with Tresh = 160:' + '\n')
            #read a clean img
            img0 = cv2.imread((path_src+'\\'+names_img[i]))
            #chang the tresh to TRESH = 160
            TRESH = 160

        if iteration==3:
            file1.write('\n>>> @@: ITERATION 3 with cv2.findContours(imgdilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) Tresh = 130:' + '\n')
            #read a clean img
            img0 = cv2.imread((path_src+'\\'+names_img[i]))
            #chang the tresh to TRESH = 160
            TRESH = 130

        #do a yellow mask
        imgW = yellowToBlack(imgO)
        #crop the size image if it need
        imgW = imgW[0:h_img, right : w_img - left]
        #convert to grey scale
        grayImage = cv2.cvtColor(imgW, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, TRESH, 255, cv2.THRESH_BINARY)
        #clean the image by blur filter
        blur_img = cv2.blur(blackAndWhiteImage, (3, 3))
        #get contour
        ret, thresh = cv2.threshold(blur_img, 115, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #use fild contoures
        img = np.zeros((imgO.shape[0], imgO.shape[1]))  # create a single channel 200x200 pixel black image
        cv2.fillPoly(img, contours, color=(255, 255, 255))
        #save the image BW
        cv2.imwrite(os.path.join(pathBW, 'fillB&W') +'_itr_'+ str(iteration) + '_' + names_img[i] , img)

        #part 3
        #do the Fourier Descriptor
        file1.write('\nimg_' + str(names_img[i])+'\n')
        imgBW = cv2.imread(os.path.join(pathBW, 'fillB&W') +'_itr_'+ str(iteration) + '_' + names_img[i] )
        # Get complex vector
        getCV(file1, 'shapes', imgBW, iteration)
        # Get fourider descriptor
        sampleFDs = FourierDescriptor('shapes')
        # real match function
        bool, con = match(file1, tpFD, sampleFDs, imgBW, iteration)
        # if it is good we can drew and save the crop img of the barcode
        print(names_img[i]+',  ,',bool)
        if bool == 'good':
            # drawOnImg()
            path = path_src + '/detected'
            cv2.imwrite(os.path.join(path, 'contours') + '_' + names_img[i], imgBW)
            myImg = allRGBimages[i][0:h_img, right: w_img - left]
            crop(myImg, con, names_img[i], path_des)  # have to get the num/name image
            cv2.drawContours(myImg, con, -1, (255, 0, 0), 3)
            path = path_src + '/detected'
            cv2.imwrite(os.path.join(path, 'markBarcod.png') + names_img[i], myImg)

            print('finish img', names_img[i])
            print('num of iteration', iteration)
            imgStatus = True
            i = i + 1

        # if there are a problem for not get a infinite loop make a break
        if iteration == 5:
            imgStatus = True
            i = i + 1
        # else continue to a next iteration




cv2.destroyAllWindows()
file1.close()

stop = timeit.default_timer()
print('Time final: ', stop - start, '[s]')