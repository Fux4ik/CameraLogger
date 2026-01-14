import cv2
import os

os.makedirs('dataset/person', exist_ok=True)

camera = cv2.VideoCapture(0)
count = 0

# Fixed ROI
ret, frame = camera.read()
h, w = frame.shape[:2]

# ROI в центре: 200x200 пикселей
x1, y1 = w//2 - 100, h//2 - 100
x2, y2 = w//2 + 100, h//2 + 150

print("Space to save frame")
print("q to exit")

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Camera', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 32:  # SPACE
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        filename = f'dataset/person/face_{count}.jpg'
        cv2.imwrite(filename, roi_gray)
        count += 1
        print(f"Saved {count} frame")
    
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()