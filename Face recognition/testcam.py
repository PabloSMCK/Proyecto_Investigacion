from threading import Thread
import cv2, time
import numpy as np
import face_recognition as fr

pablo_image = fr.load_image_file("foto1.jpg")
pablo_face_encoding = fr.face_encodings(pablo_image)[0]

known_face_encodings = [pablo_face_encoding]
known_face_names = ["Pablo"]

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(0)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()


    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            self.rgb_frame = self.frame[:, :, ::-1]
            self.face_locations = fr.face_locations(self.rgb_frame)
            self.face_encodings = fr.face_encodings(self.rgb_frame, self.face_locations)

            for (top , right, bottom, left), face_encoding in zip(self.face_locations, self.face_encodings):

                self.matches = fr.compare_faces(known_face_encodings, face_encoding)

                self.name = "Uknown"

                self.face_distances = fr.face_distance(known_face_encodings, face_encoding)

                self.best_match_index = np.argmin(self.face_distances)
                if self.matches[self.best_match_index]:
                    self.name = known_face_names[self.best_match_index]

                cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(self.frame, (left, bottom -35), (right, bottom), (0, 0 , 255), cv2.FILLED)
                self.font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.frame, self.name, (left +6, bottom -6), self.font, 1.0, (255, 255, 255), 1)


            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
          video_stream_widget.show_frame()
        except AttributeError:
            pass