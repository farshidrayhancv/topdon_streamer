#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import time
import io

def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): 
                return True
    except Exception: 
        pass
    return False

class TC001Interface:
    def __init__(self, device_id=6):
        self.device_id = device_id
        self.cap = None
        self.running = False
        self.isPi = is_raspberrypi()
        
        self.thermal_width = 256
        self.thermal_height = 192
        
        self.scale = 3
        self.alpha = 1.0
        self.colormap = 0
        self.rad = 0
        self.threshold = 2
        self.hud = True
        
        self.newWidth = self.thermal_width * self.scale
        self.newHeight = self.thermal_height * self.scale
        
        self.colormaps = [
            (cv2.COLORMAP_JET, 'Jet'),
            (cv2.COLORMAP_HOT, 'Hot'),
            (cv2.COLORMAP_MAGMA, 'Magma'),
            (cv2.COLORMAP_INFERNO, 'Inferno'),
            (cv2.COLORMAP_PLASMA, 'Plasma'),
            (cv2.COLORMAP_BONE, 'Bone'),
            (cv2.COLORMAP_SPRING, 'Spring'),
            (cv2.COLORMAP_AUTUMN, 'Autumn'),
            (cv2.COLORMAP_VIRIDIS, 'Viridis'),
            (cv2.COLORMAP_PARULA, 'Parula'),
        ]
        
    def initialize_camera(self):
        print(f"Initializing TC001 at /dev/video{self.device_id}")
        
        self.cap = cv2.VideoCapture(f'/dev/video{self.device_id}', cv2.CAP_V4L)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open /dev/video{self.device_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ TC001 initialized: {width}x{height} @ {fps}fps")
        
        return True
    
    def process_tc001_frame(self, frame):
        if frame is None:
            return None, None
        
        raw_thermal_data, thermal_process_data = np.array_split(frame, 2)
        
        temp_info = self.calculate_temperatures(thermal_process_data)
        
        raw_thermal_display = self.create_raw_thermal_display(raw_thermal_data, temp_info)
        thermal_process_display = self.create_thermal_process_display(thermal_process_data, temp_info)
        
        return raw_thermal_display, thermal_process_display
    
    def calculate_temperatures(self, thdata):
        temp_info = {}
        
        hi = int(thdata[96][128][0])
        lo = int(thdata[96][128][1])
        lo = lo * 256
        rawtemp = hi + lo
        center_temp = (rawtemp/64) - 273.15
        temp_info['center'] = round(center_temp, 2)
        
        lomax = int(thdata[..., 1].max())
        posmax = thdata[..., 1].argmax()
        mcol, mrow = divmod(posmax, self.thermal_width)
        himax = int(thdata[mcol][mrow][0])
        lomax = lomax * 256
        maxtemp = ((himax + lomax)/64) - 273.15
        temp_info['max'] = round(maxtemp, 2)
        temp_info['max_pos'] = (mrow, mcol)
        
        lomin = int(thdata[..., 1].min())
        posmin = thdata[..., 1].argmin()
        lcol, lrow = divmod(posmin, self.thermal_width)
        himin = int(thdata[lcol][lrow][0])
        lomin = lomin * 256
        mintemp = ((himin + lomin)/64) - 273.15
        temp_info['min'] = round(mintemp, 2)
        temp_info['min_pos'] = (lrow, lcol)
        
        loavg = float(thdata[..., 1].mean())
        hiavg = float(thdata[..., 0].mean())
        loavg = loavg * 256
        avgtemp = ((loavg + hiavg)/64) - 273.15
        temp_info['avg'] = round(avgtemp, 2)
        
        return temp_info
    
    def create_raw_thermal_display(self, raw_thermal_data, temp_info):
        bgr = cv2.cvtColor(raw_thermal_data, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)
        
        thermal_scaled = cv2.resize(bgr, (self.newWidth, self.newHeight), interpolation=cv2.INTER_CUBIC)
        if self.rad > 0:
            thermal_scaled = cv2.blur(thermal_scaled, (self.rad, self.rad))
        
        colormap_cv, colormap_name = self.colormaps[self.colormap]
        heatmap = cv2.applyColorMap(thermal_scaled, colormap_cv)
        
        if self.colormap == len(self.colormaps) - 1:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        self.draw_crosshairs(heatmap, temp_info['center'])
        self.draw_temperature_spots(heatmap, temp_info)
        
        if self.hud:
            self.draw_hud(heatmap, temp_info, colormap_name)
        
        cv2.putText(heatmap, f"RAW THERMAL DATA ({colormap_name})", (10, self.newHeight - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return heatmap
    
    def create_thermal_process_display(self, thdata, temp_info):
        hi_data = thdata[..., 0].astype(np.float32)
        lo_data = thdata[..., 1].astype(np.float32) * 256
        temp_data = ((hi_data + lo_data) / 64) - 273.15
        
        raw_scaled = cv2.resize(temp_data, (self.newWidth, self.newHeight), interpolation=cv2.INTER_NEAREST)
        
        temp_min, temp_max = temp_data.min(), temp_data.max()
        temp_norm = (raw_scaled - temp_min) / (temp_max - temp_min)
        temp_8bit = (temp_norm * 255).astype(np.uint8)
        
        raw_display = cv2.cvtColor(temp_8bit, cv2.COLOR_GRAY2BGR)
        
        cx, cy = self.newWidth // 2, self.newHeight // 2
        
        cv2.line(raw_display, (cx, cy-10), (cx, cy+10), (255, 255, 255), 1)
        cv2.line(raw_display, (cx-10, cy), (cx+10, cy), (255, 255, 255), 1)
        
        temp_text = f"{temp_info['center']:.1f}C"
        cv2.putText(raw_display, temp_text, (cx+15, cy-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(raw_display, "THERMAL PROCESS DATA", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(raw_display, f"Temp Range: {temp_min:.1f}C - {temp_max:.1f}C", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(raw_display, "For Object Detection", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(raw_display, "THERMAL PROCESS DATA", (10, self.newHeight - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return raw_display
    
    def draw_crosshairs(self, heatmap, center_temp):
        cx, cy = self.newWidth // 2, self.newHeight // 2
        
        cv2.line(heatmap, (cx, cy-20), (cx, cy+20), (255, 255, 255), 2)
        cv2.line(heatmap, (cx-20, cy), (cx+20, cy), (255, 255, 255), 2)
        cv2.line(heatmap, (cx, cy-20), (cx, cy+20), (0, 0, 0), 1)
        cv2.line(heatmap, (cx-20, cy), (cx+20, cy), (0, 0, 0), 1)
        
        temp_text = f"{center_temp}°C"
        cv2.putText(heatmap, temp_text, (cx+10, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, temp_text, (cx+10, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    
    def draw_temperature_spots(self, heatmap, temp_info):
        if temp_info['max'] > temp_info['avg'] + self.threshold:
            x, y = temp_info['max_pos']
            x, y = x * self.scale, y * self.scale
            cv2.circle(heatmap, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(heatmap, f"{temp_info['max']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f"{temp_info['max']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        if temp_info['min'] < temp_info['avg'] - self.threshold:
            x, y = temp_info['min_pos']
            x, y = x * self.scale, y * self.scale
            cv2.circle(heatmap, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(heatmap, f"{temp_info['min']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f"{temp_info['min']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    
    def draw_hud(self, heatmap, temp_info, colormap_name):
        cv2.rectangle(heatmap, (0, 60), (200, 200), (0, 0, 0), -1)
        
        y_pos = 76
        line_height = 14
        
        cv2.putText(heatmap, f"Avg: {temp_info['avg']}°C", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Max: {temp_info['max']}°C", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Min: {temp_info['min']}°C", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Colormap: {colormap_name}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Scale: {self.scale}x", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Blur: {self.rad}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Contrast: {self.alpha}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
        
        cv2.putText(heatmap, f"Threshold: {self.threshold}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    
    def create_dual_thermal_display(self, raw_thermal, thermal_process):
        combined = np.hstack((raw_thermal, thermal_process))
        return combined
    
    def run(self):
        if not self.initialize_camera():
            return
        
        print("\n" + "="*70)
        print("TC001 DUAL THERMAL VIEW INTERFACE")
        print("="*70)
        print("Left: Raw Thermal Data")
        print("Right: Thermal Process Data")
        print("="*70)
        print("Controls:")
        print("  q - Quit")
        print("  m - Cycle colormap")
        print("  +/- - Adjust blur")
        print("  s/x - Adjust temperature threshold") 
        print("  f/v - Adjust contrast")
        print("  h - Toggle HUD")
        print("  p - Save screenshot")
        print("="*70)
        
        cv2.namedWindow('TC001 Dual Thermal View', cv2.WINDOW_AUTOSIZE)
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error reading frame")
                    break
                
                raw_thermal, thermal_process = self.process_tc001_frame(frame)
                
                if raw_thermal is None or thermal_process is None:
                    print("Error processing frame")
                    break
                
                display = self.create_dual_thermal_display(raw_thermal, thermal_process)
                
                cv2.imshow('TC001 Dual Thermal View', display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.colormap = (self.colormap + 1) % len(self.colormaps)
                    print(f"Colormap: {self.colormaps[self.colormap][1]}")
                elif key == ord('=') or key == ord('+'):
                    self.rad += 1
                    print(f"Blur: {self.rad}")
                elif key == ord('-'):
                    self.rad = max(0, self.rad - 1)
                    print(f"Blur: {self.rad}")
                elif key == ord('s'):
                    self.threshold += 1
                    print(f"Threshold: {self.threshold}")
                elif key == ord('x'):
                    self.threshold = max(0, self.threshold - 1)
                    print(f"Threshold: {self.threshold}")
                elif key == ord('f'):
                    self.alpha = min(3.0, round(self.alpha + 0.1, 1))
                    print(f"Contrast: {self.alpha}")
                elif key == ord('v'):
                    self.alpha = max(0.1, round(self.alpha - 0.1, 1))
                    print(f"Contrast: {self.alpha}")
                elif key == ord('h'):
                    self.hud = not self.hud
                    print(f"HUD: {'On' if self.hud else 'Off'}")
                elif key == ord('p'):
                    filename = f"tc001_dual_thermal_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nShutdown requested")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    if len(sys.argv) > 1:
        device_id = int(sys.argv[1])
    else:
        device_id = 6
    
    print(f"Starting TC001 Dual Thermal View interface with /dev/video{device_id}")
    
    interface = TC001Interface(device_id)
    interface.run()

if __name__ == "__main__":
    main()
