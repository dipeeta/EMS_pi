import os
from datetime import datetime

import face_recognition
import cv2
import numpy as np
import psycopg2

import train



def face_encodings(image):
    rects = face_recognition.face_locations(image, model='hog')
    if len(rects)>0:
        bboxes = list(map(lambda rect: train.rect2bbox(rect), rects))
        encodings = face_recognition.face_encodings(image, rects)
        return (bboxes, encodings)
    else:
        return ([], [])


def save_recognition_images(image, emp_id, cam_id, timestamp):
    try:
        # for count, bbox in enumerate(bboxes):
        #     print ('{}\t{}\t{}'.format(bbox, emp_ids[count], timestamp))
        #     start = (bbox[0], bbox[1])
        #     end = (bbox[0]+bbox[2], bbox[1]+bbox[3])
        #     image = cv2.rectangle(image, start, end, (0,255,0), 2)
        #     # image = cv2.putText(image, emp_id, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        directory = 'recognized_image/{}/{}'.format(cam_id, emp_id)
        filename = '{}/{}.png'.format(directory, str(timestamp))
        train.check_dir(directory)
        cv2.imwrite(filename, image)
        return filename
    except:
        pass



def process_frame(frame, cam_id, timestamp, cam_category):
    model_file = 'models/recognition_model.pickle'
    known_encodings, known_emp_ids = train.load_known_encodings(model_file)
    bboxes, encodings = face_encodings(frame)
    employee_recog = []
    # print ('************\t{}\t************'.format(str(timestamp)))
    if len(encodings)>0:
        # print ('{} faces detected'.format(len(encodings)))
        for count, encoding in enumerate(encodings):
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(distances)
            if distances[best_match_index]<0.6:
                employee_recog.append(known_emp_ids[best_match_index])
                recog = True
            else:
                employee_recog.append(0)
                recog = False
            bbox = bboxes[count]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            face_image = frame[y:y+h, x:x+w]
            # print ('{}\t{}\t{}'.format(bboxes[count], known_emp_ids[best_match_index], distances[best_match_index]))
            conn = psycopg2.connect(
                host='database-2.cqn3tkgs4pfl.us-east-2.rds.amazonaws.com',
                database='facialdb',
                port=5432,
                user='postgres',
                password='Luffyking1')
            cursor = conn.cursor()
            if recog:
                cursor.execute('INSERT INTO api_recognitioninfo (camera_id, employee_id, timestamp, category) VALUES (%s, %s, %s, %s)', (str(cam_id), str(known_emp_ids[best_match_index]), str(timestamp), cam_category))
                filename = save_recognition_images(face_image, known_emp_ids[best_match_index], cam_id, timestamp)
            else:
                cursor.execute('INSERT INTO api_recognitioninfo (camera_id, timestamp, category) VALUES (%s, %s, %s)', (str(cam_id), str(timestamp), cam_category))
                filename = save_recognition_images(face_image, 0, cam_id, timestamp)
            conn.commit()
            conn.close()


def recognition(cam_url, cam_id, cam_category):
    status_file = 'camera/{}.txt'.format(cam_id)
    print (status_file)
    if cam_url=='0':
        source = cv2.VideoCapture(0)
    else:
        source = cv2.VideoCapture(cam_url)
    while True:
        if source.isOpened():
            if os.path.exists(status_file):
                timestamp = datetime.utcnow()
                ret, frame = source.read()
                if ret:
                    process_frame(frame, cam_id, timestamp, cam_category)
                else:
                    continue
            else:
                break
        else:
            continue
        
# recognition('0', 1)