#!/usr/bin/env python3
"""
Simple RGB Camera View
Shows only RGB Camera (/dev/video4)
"""

import cv2
import time

# Initialize RGB camera - /dev/video4
print("Opening RGB camera: /dev/video4")
rgb_cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L)

if not rgb_cap.isOpened():
    print("ERROR: Cannot open RGB camera /dev/video4")
    exit()

print("✓ RGB camera opened successfully")
print("Controls: q to quit, s to save screenshot")

cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Get RGB frame
        ret, frame = rgb_cap.read()
        
        if ret and frame is not None:
            # Add label
            cv2.putText(frame, "RGB Camera /dev/video4", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('RGB Camera', frame)
        else:
            print("Error reading frame")
            break
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"rgb_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

except KeyboardInterrupt:
    print("\nShutdown requested")

# Cleanup
rgb_cap.release()
cv2.destroyAllWindows()
print("Done")
