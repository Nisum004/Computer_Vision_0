import os
import cv2
import mediapipe as mp
import argparse
import time

filePath = '../data/video2.mp4'



def process_img(img, face_detection):

    H, W, _ = img.shape  # _ = no of channel usually 3 for BGR

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    out = face_detection.process(img_rgb)

    if out.detections is not None:  # out.detection -> list of all faces detected in image and its detail
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box  # gives face bounding box coordinates in fraction value

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)  # MediaPipe returns fractions, e.g., xmin = 0.2
            y1 = int(y1 * H)  # Multiply by W and H to get actual pixel coordinates.
            w = int(w * W)
            h = int(h * H)

            # here, (x1,y1) is the top left corner of the face, w,h are with and height in pixels

            # Blur Faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(img, 'face_detected', (x1-10,y1-10), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,0,255), 2)
    return img

args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default='../data/video2.mp4')
args = args.parse_args()

saved_path = '../saved'
os.makedirs(saved_path, exist_ok=True)


# Detect Faces
mp_face_detection = mp.solutions.face_detection

    # model selection = 0 or 0.5 for near images, 1 for far images
with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
    # Read Image

    if args.mode == 'image':
        img = cv2.imread(args.filePath)


        img = process_img(img, face_detection)

        # Save Image
        cv2.imwrite(os.path.join(saved_path, 'output2.jpg'), img)

    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
        output_video = cv2.VideoWriter(os.path.join(saved_path, 'video_output.mp4'),
                                       cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
        prev_time = time.time()
        while ret:
            frame = process_img(frame, face_detection)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            output_video.write(frame)

            cv2.imshow("Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    elif args.mode == 'webcam':
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        while True:
            ret, frame = cap.read()

            frame = process_img(frame, face_detection)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Real Time Face Blur", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
