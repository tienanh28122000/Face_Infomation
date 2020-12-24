from my_face_recognition import f_face_recognition as rec_face
from my_face_recognition import f_storage as st
import traceback
import numpy as np
import cv2

# bat dau main thread cho face recognition
class rec():
    def __init__(self):
        '''
        -db_names: [name1,name2,...,namen] list cac chuoi la ten nguoi luu tru trong db
        -db_features: array(array,array,...,array) moi array dai dien cho dac diem cua moi nguoi (db_names)
        '''
        self.db_names, self.db_features = st.load_images_to_database()

    def recognize_face(self,im):
        '''
        Input:
            -imb64: images
        Output:
            res:{'status': tra ve "ok" neu chay dc, tra ve loi duoc tim thay neu k chay dc
                'faces': [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] moi tuple trong lists dai dien cho mot khuon mat duoc phat hien
                'names': ['name', 'unknown'] list cac ten, "name" neu tim duoc, "unknown" neu khong tim duoc       
        '''
        try:
            # face dectection 
            box_faces = rec_face.detect_face(im)
            # dieu kien neu khong phat hien duoc khuon mat
            if  not box_faces:
                res = {
                    'status':'ok',
                    'faces':[],
                    'names':[]}
                return res
            else:
                if not self.db_names:
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':['unknown']*len(box_faces)}
                    return res
                else:
                    # lay cac dac diem
                    actual_features = rec_face.get_features(im,box_faces)
                    # so sach cac dac diem khuon mat tu input voi nhung nguoi duoc luu tru trong database
                    match_names = rec_face.compare_faces(actual_features,self.db_features,self.db_names)
                    # ket qua
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':match_names}
                    return res
        except Exception as ex:
            error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            res = {
                'status':'error: ' + str(error),
                'faces':[],
                'names':[]}
            return res
        
    def recognize_face2(self,im,box_faces):
        try:
            if not self.db_names:
                res = 'unknown'
                return res
            else:
                # lay cac dac diem
                actual_features = rec_face.get_features(im,box_faces)
                # so sach cac dac diem khuon mat tu input voi nhung nguoi duoc luu tru trong database
                match_names = rec_face.compare_faces(actual_features,self.db_features,self.db_names)
                # ketqua
                res = match_names
                return res
        except:
            res = []
            return res


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (38,0,237),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (38,0,237), 2)
    return img

if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("-im","--path_im",help="path image")
    parse = parse.parse_args()
    
    path_im = parse.path_im
    im = cv2.imread(path_im)
    recognizer = rec()
    res = recognizer.recognize_face(im)
    im = bounding_box(im,res["faces"],res["names"])
    cv2.imshow("face recogntion", im)
    cv2.waitKey(0)
    print(res)
