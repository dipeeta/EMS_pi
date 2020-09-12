import os
import pickle

from urllib.error import HTTPError
from skimage import io
import face_recognition
import cv2



def read_image(url):
    try:
        image = io.imread(url)
        return image
    except HTTPError:
        return []


def bgr2rgb(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    except:
        return []


def rect2bbox(rect):
    return (rect[3], rect[0], rect[1]-rect[3], rect[2]-rect[0])


def bbox2rect(bbox):
    return (bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1], bbox[0])


def face_encodings(image):
    rects = face_recognition.face_locations(image, model='hog')
    if len(rects)>0:
        bboxes = list(map(lambda rect: rect2bbox(rect), rects))
        encodings = face_recognition.face_encodings(image, rects)
        return ([bboxes[0]], [encodings[0]])
    else:
        return ([], [])


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_known_encodings(filename):
    try:
        with open(filename, 'rb') as file:
            known_data = pickle.load(file)
        encodings = known_data['encoding']
        employee_ids = known_data['employee_id']
        if len(encodings)==0:
            encodings = []
            employee_ids = []
    except:
        encodings = []
        employee_ids = []
    return (encodings, employee_ids)


def save_model(new_encodings, new_employee_ids):
    model_directory = 'models'
    model_name = '{}/recognition_model.pickle'.format(model_directory)
    check_dir(model_directory)
    (encodings, employee_ids) = load_known_encodings(model_name)
    for count in range(len(new_encodings)):
        encodings.append(new_encodings[count])
        employee_ids.append(new_employee_ids[count])
    with open(model_name, 'wb') as file:
        data = {'encoding':encodings, 'employee_id':employee_ids}
        file.write(pickle.dumps(data))


def save_trained_images(image, bbox, emp_id, image_name):
    try:
        print ('{}\t{}\t{}'.format(bbox, emp_id, image_name))
        start = (bbox[0], bbox[1])
        end = (bbox[0]+bbox[2], bbox[1]+bbox[3])
        image = cv2.rectangle(image, start, end, (0,255,0), 2)
        # image = cv2.putText(image, emp_id, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        directory = 'trained_images'
        image_name = image_name.replace('/', '_')
        image_name = image_name.replace('.', '_')
        filename = '{}/{}.png'.format(directory, image_name)
        check_dir(directory)
        cv2.imwrite(filename, image)
    except:
        pass


def train_images(images, employee_ids):
    base_url = 'http://127.0.0.1:8000/media/'
    new_encodings = []
    new_employee_ids = []
    for i in range(len(images)):
        url = base_url + images[i]
        image = read_image(url)
        if len(image)>0:
            rgb_image = bgr2rgb(image)
            bbox, encoding = face_encodings(rgb_image)
            if len(bbox)>0:
                new_encodings.append(encoding[0])
                new_employee_ids.append(employee_ids[i])
                save_trained_images(image, bbox[0], employee_ids[i], images[i])
    save_model(new_encodings, new_employee_ids)

        