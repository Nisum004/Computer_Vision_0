import os
import cv2
import mediapipe as mp
import pathlib
import argparse

filePath = '/Users/nisumlimbu/PycharmProjects/OPEN_CV/data/face.jpg'
saved_path = '/Users/nisumlimbu/PycharmProjects/OPEN_CV/saved'
os.makedirs(saved_path, exist_ok=True)

# CREATE A FUNCTION FOR PROCESSING A SINGLE FRAME
def process_img(frame, face_detection):

    # extract height and width
    H,W = frame.shape[:2]

    # cvt bgr frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect the face
    out = face_detection.process(frame_rgb)

    if out.detections is not None:
        for detection in out.detections:

            #out.detections -> location_data -> relative_bounding_box
            location = detection.location_data
            bbox = location.relative_bounding_box

            # extract pixel locations of face
            x1,y1,w,h = bbox.xmin,bbox.ymin,bbox.width,bbox.height

            # convert them to the actual pixel coordinates
            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            frame[y1:y1+h, x1:x1+w] = cv2.blur(frame[y1:y1+h, x1:x1+w], (30,30))
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
            cv2.putText(frame,'Face_detected',  (x1-10,y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    return frame

args = argparse.ArgumentParser()
args.add_argument('--mode', default='image')
args.add_argument('--filePath', default='/Users/nisumlimbu/PycharmProjects/OPEN_CV/data/face.jpg')
args = args.parse_args()

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:

    if args.mode == 'image':

        # import image
        img = cv2.imread(args.filePath)

        # process the image
        processed_img = process_img(img, face_detection)

        # save image
        cv2.imwrite(os.path.join(saved_path, 'output_img.jpg'), processed_img)

        # display image
        cv2.imshow('Image', processed_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'video':

        # import video
        cap = cv2.VideoCapture(args['filePath'])
        ret, frame = cap.read()

        # save video
        frame_height, frame_width = frame.shape[:2]
        output_video = cv2.VideoWriter(
            os.path.join(saved_path, 'output_video3.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (frame_width,frame_height)
        )

        while ret:

            # process each frame
            processed_frame = process_img(frame, face_detection)

            output_video.write(processed_frame)

            # display video by displaying each frame continuously
            cv2.imshow('Video', processed_frame)

            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    elif args.mode == 'webcam':

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]

        while True:
            ret, frame = cap.read()
            processed_img = process_img(frame, face_detection)
            cv2.imshow('Video', processed_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

