import cv2
import torch
import streamlit as st

model_yolo = torch.hub.load(./yolov5', 'custom', path='./Models/last_custom.pt', force_reload=False)

CONFIDENCE_THRESHOLD = 0.8

st.title("Vérificateur de l'uniforme :construction_worker:")

run = st.button('Run')
stop = st.button('Stop')

FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)
st.sidebar.title("Etat de conformité de l'uniforme détecté :vertical_traffic_light:")
remise_a_zero = st.sidebar.empty()

if run:
    while run : 
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        data = model_yolo(frame).pandas().xyxy[0]
        data = data.loc[data.reset_index().groupby(["class"])["confidence"].idxmax()]
        data = data.to_numpy()
     
        dict = {}
        for ls in data:
            dict[ls[6]] = ls[0:5] 
        
        for key, value in dict.items():
            if "Casque_NO" in key:
                if float(value[4]) > CONFIDENCE_THRESHOLD :
                    x = int(value[0])
                    y = int(value[1])
                    w = int(value[2])
                    h = int(value[3])
                                    
                    cv2.rectangle(frame, (x,y),(x+(w-x),y+(h-y)), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y - 20), (x+(w-x),y), (255,0,0), -1)
                    cv2.putText(frame, str(key),(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0) )
        
            if "Casque_OK" in key:
                if float(value[4]) > CONFIDENCE_THRESHOLD :
                    x = int(value[0])
                    y = int(value[1])
                    w = int(value[2])
                    h = int(value[3])

                    cv2.rectangle(frame, (x,y),(x+(w-x),y+(h-y)), (35, 127, 82), 2)
                    cv2.rectangle(frame, (x, y - 20), (x+(w-x),y), (35, 127, 82), -1)
                    cv2.putText(frame, str(key),(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0) )
                
            if "Gilet_NO" in key :
                if float(value[4]) > CONFIDENCE_THRESHOLD :
                    x = int(value[0])
                    y = int(value[1])
                    w = int(value[2])
                    h = int(value[3])

                    cv2.rectangle(frame, (x,y),(x+(w-x),y+(h-y)), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y - 20), (x+(w-x),y), (255,0,0), -1)
                    cv2.putText(frame, str(key),(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0) )

            if "Gilet_OK" in key :
                if float(value[4]) > CONFIDENCE_THRESHOLD :
                    x = int(value[0])
                    y = int(value[1])
                    w = int(value[2])
                    h = int(value[3])
                                        
                    cv2.rectangle(frame, (x,y),(x+(w-x),y+(h-y)), (35, 127, 82), 2)
                    cv2.rectangle(frame, (x, y - 20), (x+(w-x),y), (35, 127, 82), -1)
                    cv2.putText(frame, str(key),(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0) )  

        with st.sidebar:    
            if ("Casque_NO" not in dict or "Gilet_NO" not in dict) and ("Casque_OK" in dict and "Gilet_OK" in dict):
                remise_a_zero.success("Uniforme vérifié :heavy_check_mark:")
            else : 
                remise_a_zero.warning("Uniforme non vérifié :x:")

        FRAME_WINDOW.image(frame)

if stop:
    cv2.destroyAllWindows()
