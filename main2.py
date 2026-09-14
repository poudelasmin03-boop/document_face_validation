import cv2


###### Checking the FaceRecognizerSF model is loaded or not

face_reconize = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec (1).onnx",
    ""
)

print("SFace loaded successfully!")


#### Loading image one

image1 = cv2.imread("ci.jpeg")


#### Checking whether image1 is None or not

if image1 is None:
    print("Image1 failed to load")
    exit()


###### Checking image1 blur

image1_score = cv2.Laplacian(
    image1,
    cv2.CV_64F
).var()

print(f"Image1 score: {image1_score}")


if image1_score < 100:
    print("Image1 is not clear")
    exit()


#### Opening camera

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera failed to open")
    exit()


##### YuNet face detection model

face_detect = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (image1.shape[1], image1.shape[0])
)


##### Face recognition model

face_reconize = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec (1).onnx",
    ""
)



face_detect.setInputSize(
    (image1.shape[1], image1.shape[0])
)

_, face1 = face_detect.detect(image1)

print(f"The result of face1: {face1}")


# Checking face1

if face1 is None or len(face1) == 0:

    print("Failed to detect face1")
    cam.release()
    exit()


# Take first face if multiple faces are detected

face1 = face1[0]

print(f"The selected face1: {face1}")


# =========================================================
# CROP AND ALIGN IMAGE1
# =========================================================

face1 = face_reconize.alignCrop(
    image1,
    face1
)


print("Image1 face aligned successfully!")




feature1 = face_reconize.feature(face1)

print("Feature1 extracted successfully!")


# =========================================================
# CAMERA LOOP
# =========================================================

while True:

    ret, image2 = cam.read()


    # Check camera frame

    if not ret:

        print("Error of video camera")
        break


  

    face_detect.setInputSize(
        (image2.shape[1], image2.shape[0])
    )


    _, face2 = face_detect.detect(image2)

    print(f"The result of face2: {face2}")


  

    if face2 is None or len(face2) == 0:

        cv2.imshow("New window", image2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue


 

    face2 = face2[0]

    print(f"The face2 is {face2}")


   

    face2 = face_reconize.alignCrop(
        image2,
        face2
    )


  
    feature2 = face_reconize.feature(face2)

    print("Feature2 extracted successfully!")


   

    score = face_reconize.match(
        feature1,
        feature2,
        cv2.FaceRecognizerSF_FR_COSINE
    )


    print(f"The score is: {score}")


   

    if score >= 0.5:

        print("Image is valid")

        cv2.putText(
            image2,
            "Face Matched",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("New window", image2)

        cv2.waitKey(1000)

        break


    else:

        print("Image is not valid")

        cv2.putText(
            image2,
            "Face Not Matched",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow("New window", image2)


    # Press q to quit

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# =========================================================
# RELEASE CAMERA
# =========================================================

cam.release()

cv2.destroyAllWindows()