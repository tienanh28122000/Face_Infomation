import f_Face_info
import cv2
import time
import imutils
import argparse

parser = argparse.ArgumentParser(description="Face Info")
parser.add_argument('--input', type=str, default= 'video',
                    help="video or image")
parser.add_argument('--path_im', type=str,
                    help="path of image")
args = vars(parser.parse_args())

type_input = args['input']
if type_input == 'image':
    # image 
    # doc data
    frame = cv2.imread(args['path_im'])
    # lay data
    out = f_Face_info.get_face_info(frame)
    # chen khung face detection
    res_img = f_Face_info.bounding_box(out,frame)
    cv2.imshow('Face info',res_img)
    cv2.waitKey(0)

if type_input == 'video':
    # video stream
    cv2.namedWindow("Face info")
    cam = cv2.VideoCapture(0)
    while True:
        star_time = time.time()
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=720)
        
        # lay data
        out = f_Face_info.get_face_info(frame)
        # chen khung face detection
        res_img = f_Face_info.bounding_box(out,frame)

        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(5,25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow('Face info',res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
