import cv2
import cv2.aruco as aruco
from cv2.aruco import CharucoBoard

# Define parameters for the ChArUco board
squares_x = 5      # Number of chessboard squares along the X-axis
squares_y = 7      # Number of chessboard squares along the Y-axis
square_length = 0.04   # Square length in meters
marker_length = 0.02   # Marker length in meters

# Define the dictionary used for markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# Create the Charuco board

test = CharucoBoard((4, 6), 0.03, 0.02, cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_50))

img = test.generateImage((1080, 1920))
cv2.imshow("test", img)
cv2.waitKey()

# Save the board image
cv2.imwrite("charuco_board.pdf", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
