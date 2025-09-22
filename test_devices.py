#!/usr/bin/env python3
"""
Test virtual device access
"""
import cv2
import subprocess

def test_virtual_devices():
    print("Testing virtual device access...")
    
    # Check if devices exist
    import os
    if not os.path.exists('/dev/video10'):
        print("✗ /dev/video10 does not exist")
        return False
    if not os.path.exists('/dev/video11'):
        print("✗ /dev/video11 does not exist") 
        return False
    
    print("✓ Virtual devices exist")
    
    # Test permissions
    result = subprocess.run(['ls', '-la', '/dev/video10', '/dev/video11'], 
                           capture_output=True, text=True)
    print("Device permissions:")
    print(result.stdout)
    
    # Test OpenCV access (for reading - this should work)
    rgb_cap = cv2.VideoCapture('/dev/video10', cv2.CAP_V4L)
    if rgb_cap.isOpened():
        print("✓ Can open /dev/video10 for reading")
        rgb_cap.release()
    else:
        print("Note: Cannot open /dev/video10 for reading (normal until streaming starts)")
    
    thermal_cap = cv2.VideoCapture('/dev/video11', cv2.CAP_V4L)
    if thermal_cap.isOpened():
        print("✓ Can open /dev/video11 for reading")
        thermal_cap.release()
    else:
        print("Note: Cannot open /dev/video11 for reading (normal until streaming starts)")
    
    return True

if __name__ == "__main__":
    test_virtual_devices()
