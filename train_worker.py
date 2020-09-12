import os
import json

import pika

import train
import recog

host = 'peacock.rmq.cloudamqp.com'
vhost = 'jhqupmiz'
user =  'jhqupmiz'
password = 'ewngApID2bS6OdQFVJ3Q8Zz9kQZjLwA2'

#passing connection parameters for rabbit MQ
parameters = pika.ConnectionParameters(
    host=host,
    virtual_host=vhost,
    credentials=pika.PlainCredentials(user,password)
)

#callback function for traning
def train_callback(ch, method, properties, body):
    #Acquire message from the queue data
    option = json.loads(body)['type']
    if (option=='train'):
        image_data = json.loads(body)['data']
        #Intialize list for url and id employee
        images = []
        employee_ids = []
        for data in image_data:
            images.append(data['image'])
            employee_ids.append(data['employee'])    
        #train the images
        if len(image_data)>0:
            train.train_images(images, employee_ids)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    elif (option=='camera'):
        cam_data = json.loads(body)['data']
        url = cam_data['camera_url']
        status = cam_data['status']
        cam_id = cam_data['camera_id']
        cam_category = cam_data['camera_category']
        cam_directory = 'camera'
        cam_file = '{}/{}.txt'.format(cam_directory, cam_id)
        train.check_dir(cam_directory)
        if status:
            with open(cam_file, 'w+') as file:
                file.write('true')
            recog.recognition(url, cam_id, cam_category)
        else:
            os.remove(cam_file)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    


with pika.BlockingConnection(parameters) as connection:
    print ('*** Waiting for messages ***')
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue='ems_train_queue',
        on_message_callback=train_callback
    )
    channel.start_consuming()