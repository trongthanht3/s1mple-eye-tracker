import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def eye_detect(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray,1.3,5)
    height = face_gray.shape[0]
    width = face_gray.shape[1]
    eye_left = None;
    eye_right = None;
    for (ex,ey,ew,eh) in eyes:
        if ey+eh > height/2:
            pass
        # cv2.rectangle(img, (ex + x, ey + y), (ex + ew + x, ey + eh + y), (0, 255, 255), 2)
        eye_center = ex + ew/2
        if eye_center < width/2:
            eye_left = face[ey:ey+eh, ex:ex+ew]
            # cv2.imshow('left', eye_left)
        else:
            eye_right = face[ey:ey+eh, ex:ex+ew]
            # cv2.imshow('right', eye_right)

    return eye_left, eye_right

def face_detect(img):
    # conver to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('',img_gray)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 4)

    # rectangle for face
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    biggest = (0, 0, 0, 0)
    if len(faces) > 1:
        biggest = (0,0,0,0)
        for face in faces:
            if face[3] > biggest[3]:
                biggest = face
    elif len(faces) == 1:
        biggest = faces
    else:
        return None
    print(np.squeeze(biggest))
    x,y,w,h = np.squeeze(biggest)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_cut = img[y:y+h, x:x+w]
    print("face_cut location: ", biggest)
    return face_cut

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.filterByCircularity = False
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def cut_eyebrows(eye):
    height, width = eye.shape[:2]
    eyebrow_h = int(height / 4)
    eye = eye[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return eye

def blob_process(eye, detector):
    ret, eye = cv2.threshold(eye, 60, 255, cv2.THRESH_BINARY)
    print('eye_shape: ', eye.shape)
    eye = cv2.erode(eye, None, iterations=2) #1
    eye = cv2.dilate(eye, None, iterations=4) #2
    eye = cv2.medianBlur(eye, 5) #3
    keypoints = detector.detect(eye)
    return keypoints

# img = cv2.imread('20201110_235621.jpg')
# x0, y0, c0 = img.shape
# img = cv2.resize(img, (int(y0 * 0.5), int(x0 * 0.5)))

if __name__ == '__main__':


    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        x0, y0, c0 = img.shape
        img = cv2.resize(img, (int(y0 * 0.5), int(x0 * 0.5)))
        print("image_shape: ", img.shape)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f_frame = face_cascade.detectMultiScale(img_gray, 1.3,5)
        if f_frame is not None:
            for (x,y,w,h) in f_frame:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        face = face_detect(img)
        if face is not None:
            eyes_gray = eye_detect(face)
            for eye in eyes_gray:
                if eye is not None:
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('my image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



# eye_bin = cut_eyebrows(eye_bin)
# print("eye_shape: ", eye_bin.shape)
# keypoints = blob_process(eye_bin, detector)
# # keypoints = detector.detect(eye_bin)
# eye_bin = cv2.drawKeypoints(eye_bin, keypoints, eye_bin, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#
# print("keypoint: ", keypoints)
# cv2.drawKeypoints(eye_bin, keypoint, eye_bin, (0,0,255))

# print(eyes[0].shape)

# cv2.imshow('left', eye_bin)
# # cv2.imshow('right', eyes[1])
# cv2.waitKey()

