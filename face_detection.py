# import cv2
# import numpy as npy
# import face_recognition as face_rec
#
# #function
# def resize(img,size):
#     width = int(img.shape[1]*size)
#     height = int(img.shape[0] * size)
#     dimension = (width,height)
#     return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
#
#
# #img declaration
# mageshwari=face_rec.load_image_file('sample_images\mageshwari.jpg')
# mageshwari=cv2.cvtColor(mageshwari,cv2.COLOR_BGR2RGB)
# mageshwari = resize(mageshwari, 0.50)
# mageshwari_test=face_rec.load_image_file('sample_images\mageshwari_test.jpg')
# mageshwari_test = resize(mageshwari_test, 0.50)
# mageshwari_test=cv2.cvtColor(mageshwari_test,cv2.COLOR_BGR2RGB)
#
# #finding face location
#
# faceLocation_mageshwari = face_rec.face_locations(mageshwari)[0]
# encode_mageshwari = face_rec.face_encodings(mageshwari)[0]
# cv2.rectangle(mageshwari,(faceLocation_mageshwari[3],faceLocation_mageshwari[0]),(faceLocation_mageshwari[1] , faceLocation_mageshwari[2]) , (255,0,255), 3)
#
#
# faceLocation_mageshwaritest = face_rec.face_locations(mageshwari_test)[0]
# encode_mageshwaritest = face_rec.face_encodings(mageshwari_test)[0]
# cv2.rectangle(mageshwari_test,(faceLocation_mageshwari[3],faceLocation_mageshwari[0]),(faceLocation_mageshwari[1] , faceLocation_mageshwari[2]) , (255,0,255), 3)
#
#
# cv2.imshow('main_img', mageshwari)
#
# cv2.imshow('test_img',mageshwari_test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









import cv2
import numpy as npy
import face_recognition as face_rec

# function
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# img declaration
mageshwari = face_rec.load_image_file('sample_images/mageshwari.jpg')
mageshwari = cv2.cvtColor(mageshwari, cv2.COLOR_BGR2RGB)
mageshwari = resize(mageshwari, 0.50)
mageshwari_test = face_rec.load_image_file('sample_images/mageshwari_test.jpg')
mageshwari_test = resize(mageshwari_test, 0.50)
mageshwari_test = cv2.cvtColor(mageshwari_test, cv2.COLOR_BGR2RGB)

# finding face location
face_locations_mageshwari = face_rec.face_locations(mageshwari)
if len(face_locations_mageshwari) > 0:
    encode_mageshwari = face_rec.face_encodings(mageshwari)[0]
    cv2.rectangle(mageshwari, (face_locations_mageshwari[0][3], face_locations_mageshwari[0][0]),
                  (face_locations_mageshwari[0][1], face_locations_mageshwari[0][2]), (255, 0, 255), 3)
else:
    print("No faces found in main image")

face_locations_mageshwaritest = face_rec.face_locations(mageshwari_test)
if len(face_locations_mageshwaritest) > 0:
    encode_mageshwaritest = face_rec.face_encodings(mageshwari_test)[0]
    cv2.rectangle(mageshwari_test, (face_locations_mageshwaritest[0][3], face_locations_mageshwaritest[0][0]),
                  (face_locations_mageshwaritest[0][1], face_locations_mageshwaritest[0][2]), (255, 0, 255), 3)
else:
    print("No faces found in test image")

results = face_rec.compare_faces([encode_mageshwari] , encode_mageshwaritest)
print(results)
cv2.putText(mageshwari_test, f'{results}',(50,50),cv2.FONT_HERSHEY_COMPLEX, 1 , (0,0,255) , 2)

cv2.imshow('main_img', mageshwari)
cv2.imshow('test_img', mageshwari_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
