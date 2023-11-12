import face_recognition
import cv2

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("Images/biden.jpg")
obama_image = face_recognition.load_image_file("Images/obama.jpg")
unknown_image = face_recognition.load_image_file("Images/test.jpg")

# Face Location
coordinates = face_recognition.face_locations(biden_image, model="hog") # model="cnn"

for coord in coordinates:
    cv2.rectangle(biden_image, pt1=(coord[3], coord[0]), pt2=(coord[1], coord[2]), color=(255,0,0), thickness=1)

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Biden? {}".format(results[0]))
print("Is the unknown face a picture of Obama? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

while True:
    cv2.imshow("Face", cv2.cvtColor(cv2.resize(biden_image,(0,0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1)
    if key == 27:
        break

