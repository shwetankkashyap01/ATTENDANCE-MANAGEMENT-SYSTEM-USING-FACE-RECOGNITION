from gettext import npgettext
import cv2
import numpy as np
import face_recognition
import os  # Path we ask our prog to find this folder and ask no.of images it as
#(Images attedence) and import them and find encodings for them ##333
from datetime import datetime


#1.*************************************************************************************************************************************************************************************************************************************
                    #<----we ask our prog to find this folder and ask no.of images it as---->
                    # (Images attedence) and import them and find encodings for them ##333
                    # we can create a list that will get img from our folder directly and than it will generate encodings automitcally and than
path = 'ImagesAttendance'
# it try to find in our web cam
# list call
images = []  # we will take name output result use this name # manually type list we just take it images itself
classNames = []
myList = os.listdir(path)  # grab images of this folder (img-attendance)
print(myList)
# use this names and  import images them 1 by 1
for cl in myList:
    # we r going to import each class we use load img but use imread
   # current img
    curImg = cv2.imread(f'{path}/{cl}')  # cl is name of our img
    images.append(curImg)
    # we just want bill gates noy .jpg slipt cl grab
    classNames.append(os.path.splitext(cl)[0])
    # first element of it - bill gates
    print(classNames)
             
                   # if u look at this when we bring new image we have to write it manually
                   # it give it us name and than store it and find it encodings.


#2.*************************************************************************************************************************************************************************************************************************************************************************
                        #<--- import img find encoding one of them create fun to encode all of them---- >


def findEncodings(images):
    encodeList = []  # empty list encodings at end
    for img in images:  # loop through img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert it into rgb
        encode = face_recognition.face_encodings(
            img)[0]  # this find encodings
        encodeList.append(encode)  # than we will append it to our list
  
    return encodeList



#3.********************************************************************************************************************************************************************************************************************************************************************************************
                               # <--------MARK ATTEDENCE--------> link database


def markAttendence(name):
    with open('Attendance.csv', 'r+') as f:  # read nd write same time r+

        # read all lines in data so  arrived not repeated
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:  # only 1 line later on multiple line we go through ech line to find entry repeated
            # split based on comma name and time seperated
            entry = line.split(',')
            # we want to put all name we find in this list
            # entry is 2 value we just want first one
            # entry at 0 will be name append only name in out list
            nameList.append(entry[0])
    # list of all name we going to check current name is present or not
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print("Attendance of", name, "updated.")

   # markAttendence('ELON')  # call it here



#2.1 Continued***************************************************************************************************************************************************************************************************************************************************************************

# to run this we call this  function
# sending our images #calling this function
encodeListKnown = findEncodings(images)
print('Encoding Complete')  # print when we ever find encodings
#  print(len(encodeListKnown)) #to check



#4.********************************************************* THIRD STEP(ACC UT) **********************************************************************************************************************************************************
            # FIND MATCHES BETWEEN OUR ENCODINGS BUT WE DONT HAVE IMG TO MATCH THAT IMG COMING FROM
                                                      #<----- WEB CAM----------->
                                                      ### NEXT STEP IS TO FIND ENCODINGS OF WEBCAM #######
cap = cv2.VideoCapture(0)  # 0 as id

while True:  # to get each frame one by one
    success, img = cap.read()  # this will give us our image
    # in real time we reduce size of img this help us speeding the process --
    # do not define any pixel size(0,0) scale 0.25 it 1/4th of size.
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # than we can find all location in our small image
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(
        imgS, facesCurFrame)  # WEB CAM HAVE MULTIPLE FACES WE R
    # GOING TO FIND LOCATION OF OUR FACES AND THAN SENDING LOCATION
    # TO OUR ENCODINGS FUCNTION



#5. *********************************************************Fourth Step(ACC UT)*********************************************************************************************************************************************************************************************
   
                   # ITERATES THROUGH ALL FACES THAT WE HAVE FOUND IN CURRENT FRAME AND THAN WE COMAPRE ALL THIS FACE
    # WITH ALL THE ENCODINGS THAT WE FOUND BEFORE

    # LOOP THROUGH TOgether
    for Encodeface, faceloc in zip(encodeCurrFrame, facesCurFrame):
        # one by one it will grab one face location from faces current frame list and than it will grab the encodings
        # of encode face from encode currents frame 
        # #want in same loop so zip matching
        # compare with one of the encodings with encodeface
        matches = face_recognition.compare_faces(encodeListKnown, Encodeface)
   # sending list of known faces
   
 
   
   
#6. *************************************************************************************************************************************************************************************************************************************************************************************8
                                                       # finding distance
        faceDis = face_recognition.face_distance(encodeListKnown, Encodeface)
 # sending encode list to face distance func it returns list us well three values distance of each of them
 # lowest distance best of them

       # print(faceDis)  # find lowest element best match
        # numpy dots minimum at oure face distance 0,1,2
        matchIndex = np.argmin(faceDis)
        # we have that index now we knw which person we talking about

        # writing name
        # scale down 1/4th image and performing this calculation
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()  # upper case
          #  print(name)
            # create a rectangle we find location we have all location face_location
            y1, x2, y2, x1 = faceloc  # loc of our faces
            # in order to revie actual value multiply 4 -> 1/4th
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # draw rectangle on original img with tchnique 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0),
                          cv2.FILLED)  # box filled rectangle
            cv2.putText(img, name, (x1+6, y2-6),  # name in string dont convert
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # color thickness
            # print(name)
            markAttendence(name)  # call it here

    cv2.imshow('Webcam', img)  # want to show original img not small img
    # cv2.waitKey(esc)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break

#---------------------------------------------------X------------------------------------------------------------------------------------------#
  
