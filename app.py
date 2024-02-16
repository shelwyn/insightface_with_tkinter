import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import itertools
import cv2
from flask import Flask, render_template, Response, jsonify, redirect, url_for
import requests
import random
import datetime
import cv2
import insightface
from insightface.app import FaceAnalysis
captured_frame = None
p_gender="-"
p_age=0



def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def generate_avatar(gender):
    print("Starting")
    random_image_number = random.randint(1, 50)
    ########################### SWAP FACE WITH INSIGHT  ###########
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread('image.png')
    faces = app.get(img)
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    human_face = cv2.imread('face.png')
    dest_faces = app.get(human_face)
    dest_face = dest_faces[0]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, dest_face, paste_back=True)

    cv2.imwrite("edited.png", res)
    current_time = datetime.datetime.now()
    unique_file_name = current_time.strftime("%m_%d_%H_%M_%S")
    #file_name = p_gender + "_" + random_image_number + "_" + unique_file_name
    #swap_save_path = 'static/uploads/swaps/' + file_name + ".png"
    #cv2.imwrite(swap_sav   e_path,res)
    #edited_image_path = 'static/uploads/edited.png'
    edited_image_path="XX"
    display_edited_image()
    return edited_image_path  # Return the path of the edited avatar image

def predict_gender(image_path):
    gender=""
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    ageProto = "models/age_deploy.prototxt"
    ageModel = "models/age_net.caffemodel"
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    frame = cv2.imread('face.jpg')
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    padding = 20
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                    :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        GA= gender + "|" + age
        
    return GA

def gen_frames():
    global captured_frame
    while True:
        if captured_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + captured_frame + b'\r\n\r\n')
            continue
        success, frame = video.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def upload_image(image_type):
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    if file_path:
        # Save the image as face.png or image.png based on image_type
        save_as = 'face.png' if image_type == 'face' else 'image.png'
        img = Image.open(file_path)
        img.save(save_as)

        # Resize for display
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        # Display the image below the corresponding button
        if image_type == 'face':
            label_face_image.configure(image=img_tk)
            label_face_image.image = img_tk  # Keep a reference
        else:
            label_image_image.configure(image=img_tk)
            label_image_image.image = img_tk  # Keep a reference

def display_edited_image():
    print("XXX")
    try:
        img = Image.open('edited.png')
        img = img.resize((250, 250), Image.Resampling.LANCZOS)  # Resize for consistency
        img_tk = ImageTk.PhotoImage(img)
        label_edited_image.configure(image=img_tk)
        label_edited_image.image = img_tk  # Keep a reference!
    except FileNotFoundError:
        print("File 'edited.png' not found.")

def save_image(event=None):
    print(label_edited_image.__dict__)
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if file_path:
            try:
                img = Image.open('edited.png')
                img.save(file_path)
                print(f"Image saved as {file_path}")
            except Exception as e:
                print(f"Error saving the image: {e}")
    except:
        print("No image to save.")

root = tk.Tk()
root.title("Image Upload App")

# Calculate the desired size (50% width, 100% height of the screen)
screen_width = int(root.winfo_screenwidth() * 0.8) 
screen_height = int(root.winfo_screenheight() * 0.5) 
root.geometry(f"{screen_width}x{screen_height}")
root.resizable(False, False)
# Divide the screen into three sections
frame_left = tk.Frame(root, width=screen_width//3, height=screen_height)
frame_center = tk.Frame(root, width=screen_width//3, height=screen_height)
frame_right = tk.Frame(root, width=screen_width//3, height=screen_height)

frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
frame_center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
frame_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add buttons and labels
button_face_upload = tk.Button(frame_left, text="Face Upload", command=lambda: upload_image('face'))
button_face_upload.pack()
label_face_image = tk.Label(frame_left)
label_face_image.pack()

button_image_upload = tk.Button(frame_center, text="Image Upload", command=lambda: upload_image('image'))
button_image_upload.pack()
label_image_image = tk.Label(frame_center)
label_image_image.pack()

# Adding "Update Image" button in the third section
button_update_image = tk.Button(frame_right, text="Update Image", command=lambda: generate_avatar("male"))
button_update_image.pack()




frame_bottom = tk.Frame(root, height=50)  # Adjust height as needed
frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)


# The label for displaying the edited image is still in the frame_right section
label_edited_image = tk.Label(frame_right)
label_edited_image.pack()

# Bind the right-click event
label_edited_image.bind("<Button-3>", save_image)

root.mainloop()
