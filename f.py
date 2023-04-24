import cv2
import numpy as num
import face_recognition

# 1.********************************** STEP 1***********************************************************************************************
# we will get encoding of normal image and than we use test img so that it can
# find the pic of elon over here or not

imgElon = face_recognition.load_image_file(
    'ImagesBasics/Elon Musk.jpg')  # fun to load images
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)  # convert into rgb

# import test img
imgTest = face_recognition.load_image_file('ImagesBasics/Bill gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# 2.************************************STEP 2 IS TO FINDING FACES IN OUR IMAGES AND THAN FINDING THEIR ENCODINGS AS WELL********************************************************************************

# first detect the face
# send the img elon ) bcz we r sending the
faceLoc = face_recognition.face_locations(imgElon)[0]
# single img we will get the first element of this

encodeElon = face_recognition.face_encodings(imgElon)[0]  # encoding the face
# face location we r doing so we can see where we detected the faces where it is location properly or not


# 3.***************************Rectangle box in name*****************************************

cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]),
              (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
# (142, 1122, 409, 854) value of top right bottom left we give x1 x2 y1 y2
# print(faceLoc)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),
              (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)


# 4.***************************************************3rd step(UT)*************************************************************************************************
# comparing this faces and finding distance between them.Now going to comapre
# elon and test encode

### 128 measurments of both the faces####

#### Linear SVM  FIND OUT WEATHER THEY MATCH OR NOT##########

# list of known one face elon - 1 elon match with test
results = face_recognition.compare_faces([encodeElon], encodeTest)
# encode elon comparing with encodings (test)

print(results)  # verify same hai ya ni######elon = test not bill gates

####### Encoding do not match with elon = bill gate##########

# results = face_recognition.compare_faces([encodeElon],encodeTest)

#### best match we find distance##
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
####### Lower the distance the better the match is####
# Best match - [0.36001764] #clear distance when they match when they do not match
print(results, faceDis)
# distance[0.75228577] - not match


# 5.******************************************************************************************************************************************************************************************************************

# just to display actual result image round at 2
# origin 50 50 put text at img test variable

cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (
    50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


# 6.********************************************************************************************************************************************************************************************************

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
