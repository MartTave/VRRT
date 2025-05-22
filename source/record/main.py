import cv2 as cv

def main():
    cap = cv.VideoCapture(2)

    while True:
        ret, frame = cap.read()

        assert ret

        cv.imshow("frame", frame)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
