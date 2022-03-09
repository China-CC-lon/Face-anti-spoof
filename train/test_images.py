import cv2
import dlib

detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('M:/xilinx_comtest/face-anit\data/data_dlib_bat/shape_predictor_68_face_landmarks.dat')
# face_reco_model = dlib.face_recognition_model_v1("M:/xilinx_comtest/face-anit\data/data_dlib_bat/dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    reg, frame = cap.read()
    faces = detector(frame, 0)
    if len(faces) != 0:
        for t, d in enumerate(faces):
            h = d.bottom() - d.top()
            w = d.right() - d.left()
            hd = int(h/2)
            wd = int(w/2)

            cv2.rectangle(frame, tuple([d.left() - wd, d.top() - hd]),
                          tuple([d.right() + wd, d.bottom() + hd]),
                          (255,255,255), 2)
        cv2.imshow('figure', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



