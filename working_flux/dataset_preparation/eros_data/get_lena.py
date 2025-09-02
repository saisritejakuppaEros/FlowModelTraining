import cv2
import urllib.request
import numpy as np

# URL of Lena image (classic test image from OpenCV sample)
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

# Download and decode image
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
lena = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Resize to 256x256
lena_resized = cv2.resize(lena, (256, 256))

# Save locally
cv2.imwrite("lena_256.png", lena_resized)

print("Lena image saved as lena_256.png")
