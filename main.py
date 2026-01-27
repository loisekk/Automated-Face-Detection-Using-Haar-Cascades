import cv2

# Load Haar Cascade model
model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(model_path)

if face_cascade.empty():
    raise IOError("❌ Haar Cascade model not loaded")

# Read image
image_path = "./Black Myth_ Wukong.jpg"
image = cv2.imread(image_path)

if image is None:
    raise IOError("❌ Image not found or path is incorrect")

# Resize image if too large (better visualization)
max_width = 900
if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=6,
    minSize=(40, 40)
)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display face count
cv2.putText(
    image,
    f"Faces Detected: {len(faces)}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# Show image
cv2.imshow("Cricket Team - Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
