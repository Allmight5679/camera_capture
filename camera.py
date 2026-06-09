import cv2
import numpy as np
import pose


def main():
    print("One or Two Cameras?")
    num_of_camera = input()

    detector = pose.create_detector()

    cap1 = cv2.VideoCapture(0)
    cap2 = None

    if num_of_camera == "2":
        cap2 = cv2.VideoCapture(1)

    while True:
        success1, frame1 = cap1.read()

        if not success1:
            print("Camera 1 not working.")
            break

        frame1 = pose.process_frame(detector, frame1)

        if cap2 is not None:
            success2, frame2 = cap2.read()

            if success2:
                frame2 = pose.process_frame(detector, frame2)

                frame1 = cv2.resize(frame1, (640, 480))
                frame2 = cv2.resize(frame2, (640, 480))

                split_screen = np.hstack((frame1, frame2))

                cv2.imshow("Two Camera Pose Detection", split_screen)
            else:
                cv2.imshow("Camera 1 Pose Detection", frame1)
        else:
            cv2.imshow("Camera 1 Pose Detection", frame1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()

    if cap2 is not None:
        cap2.release()

    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()