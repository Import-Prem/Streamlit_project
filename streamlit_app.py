#Project by Prem, Date: 10-09-2021


#importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import urllib


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Import-Prem/Streamlit_project/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

readme_text = st.markdown(get_file_content_as_string('instructions.md'))



#function for landmark identification
def landmark_detection() :
    st.title('Landmark identification')
    st.subheader("This project takes the input image and identifies the landmark in the image[only Asia's landmarks].")
    uploaded_file = st.file_uploader("Upload a image",type='jpg')
    

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.')
        my_bar = st.progress(0)
        TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
        LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
        IMAGE_SHAPE = (321, 321)
        
        
        classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                 input_shape=IMAGE_SHAPE+(3,),
                                                 output_key="predictions:logits")])
        df = pd.read_csv(LABEL_MAP_URL)
        label_map = dict(zip(df.id, df.name))
    
    
        img = image.resize(IMAGE_SHAPE)
        img = np.array(img)/255.0
    
    
        img = img[np.newaxis, ...]
        prediction = classifier.predict(img)
        st.header(label_map[np.argmax(prediction)])
        my_bar.progress(100)


#function for object detection
def obj_detection():
    
    st.title('Object Detection')
    st.subheader("This object detection projects takes the input as image and outputs the image with objects bounded in a rectangle with confidence score.")

    uploaded_file = st.file_uploader("Upload a image",type='jpg')
    if uploaded_file != None:
        image1 = Image.open(uploaded_file)
        image2 =np.array(image1)
        
        st.image(image1, caption='Uploaded Image.')
        
        my_bar = st.progress(0)
        
        confThreshold =st.slider('Confidence', 0, 100, 50)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        whT = 320
        #### LOAD MODEL
        ## Coco Names
        classesFile = "coco.names"
        classNames = []
        with open(classesFile, 'rt') as f:
            classNames = f.read().split('\n')
            
        
        ## Model Files        
        modelConfiguration = "yolov3.cfg"
        modelWeights = "yolov3.weights"
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        #finding the objects
        def findObjects(outputs,img):
            hT, wT, cT = image2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
            obj_list=[]
            confi_list =[]
            #drawing rectangle around object
            for i in indices:
                i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(image2, (x, y), (x+w,y+h), (255, 0 , 255), 2)
                #print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(image2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
            if st.checkbox("Show Object's list" ):
                
                st.write(df)
            if st.checkbox("Show Confidence bar chart" ):
                st.subheader('Bar chart for confidence levels')
                
                st.bar_chart(df["Confidence"])
           
        blob = cv2.dnn.blobFromImage(image2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,image2)
    
        st.image(image2, caption='Proccesed Image.')
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        my_bar.progress(100)
        
        
st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show Instruction","Landmark identification", "Object detection","Show the source code", "About"])
 
if app_mode == "Show Instructions":
    st.sidebar.success('Select a operation to perform')
    
elif app_mode == "Landmark identification":
    readme_text.empty()
    landmark_detection()    
        
elif app_mode == "Show the source code":
    readme_text.empty()
    st.code(get_file_content_as_string("streamlit_app.py"))
    
elif app_mode == "Object detection":
    readme_text.empty()
    obj_detection()
    
elif app_mode == "About":
    readme_text.empty()
    st.markdown(get_file_content_as_string('about.md'))