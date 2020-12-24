import face_recognition 
import numpy as np

def detect_face(image):
    '''
    Input: images co dang numpy.ndarray, shape=(W,H,3)
    Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] moi tuple dai dien cho mot khuon mat duoc phat hien
    Neu khong co gi duoc phat hien  --> Output: []
    '''
    Output = face_recognition.face_locations(image)
    return Output

def get_features(img,box):
    '''
    Input:
        -img:images co dang numpy.ndarray, shape=(W,H,3)
        -box: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] moi tuple dai dien cho mot khuon mat duoc phat hien
    Output:
        -features: [array,array,...,array] moi mang trong list dai dien cho cac dac diem cua mot khuon mat
    '''
    features = face_recognition.face_encodings(img,box)
    return features

def compare_faces(face_encodings,db_features,db_names):
    '''
    Input:
        db_features = [array,array,...,array] moi mang trong list dai dien cho cac dac diem trong mot khuon mat 
        db_names =  array(array,array,...,array) moi mang dai dien cho cac dac diem cua moi nguoi (db_names)
    Output:
        -match_name: ['name', 'unknown'] list cac ten, "name" neu tim duoc, "unknown" neu khong tim duoc 
    '''
    match_name = []
    names_temp = db_names
    Feats_temp = db_features           

    for face_encoding in face_encodings:
        try:
            dist = face_recognition.face_distance(Feats_temp,face_encoding)
        except:
            dist = face_recognition.face_distance([Feats_temp],face_encoding)
        index = np.argmin(dist)
        if dist[index] <= 0.6:
            match_name = match_name + [names_temp[index]]
        else:
            match_name = match_name + ["unknown"]
    return match_name