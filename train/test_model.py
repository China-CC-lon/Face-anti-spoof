import time
import numpy as np
import cv2
import dlib
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_net = torch.load("face_anti_model2.pth").to(device)
model_net.eval()
detector = dlib.get_frontal_face_detector()

camera = cv2.VideoCapture(0)

while True:

    if camera.isOpened():
        print("Camera is opened")
    # frames = []
    reg, frame = camera.read()
    cv2.imshow("figure", frame)

    frame_t = torch.tensor(cv2.resize(frame, dsize=(256, 256)), dtype=torch.float32).unsqueeze(3).to(device)
    frame_t = torch.transpose(frame_t, 2, 1)
    frame_t = torch.transpose(frame_t, 3, 0)
    print(frame_t.shape)

    # print(len(frames))
        # print(len(frames))
    with torch.no_grad():
        # frame_t.to(device)
        D, F = model_net(frame_t)
        print(F, torch.norm(D).pow(2))
        value = torch.norm(F).pow(2) + 0.015 * torch.norm(D).pow(2)
        print(value)
        if value > 1644:
            print("你是真活人")
        else:
            print("你是假活人")
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
