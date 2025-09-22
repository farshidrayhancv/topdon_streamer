#!/usr/bin/env python3

import cv2
import numpy as np
import subprocess
import threading
import time
import sys
import os

def debug_print(message, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def kill_competing_processes():
    debug_print("Killing any competing video processes...")
    try:
        subprocess.run(['pkill', '-f', 'ffmpeg'], capture_output=True)
        subprocess.run(['pkill', '-f', 'gstreamer'], capture_output=True)
        subprocess.run(['pkill', '-f', 'v4l'], capture_output=True)
        time.sleep(1)
        debug_print("Cleaned up competing processes")
    except:
        pass

def reset_video_devices(thermal_path, rgb_path):
    debug_print("Resetting video devices...")
    
    try:
        subprocess.run(['v4l2-ctl', '-d', thermal_path, '--set-fmt-video=width=256,height=384,pixelformat=YUYV'], 
                      timeout=3, capture_output=True)
        debug_print(f"Reset {thermal_path}")
        
        subprocess.run(['v4l2-ctl', '-d', rgb_path, '--set-fmt-video=width=320,height=240,pixelformat=MJPG'], 
                      timeout=3, capture_output=True)
        debug_print(f"Reset {rgb_path}")
        
        time.sleep(0.5)
        
    except Exception as e:
        debug_print(f"Device reset failed: {e}")

class TC001DirectInterface:
    def __init__(self, thermal_device=6, rgb_device=4):
        self.thermal_device_path = f'/dev/video{thermal_device}'
        self.rgb_device_path = f'/dev/video{rgb_device}'
        
        self.thermal_process = None
        self.rgb_process = None
        
        self.thermal_frame = None
        self.raw_thermal_frame = None
        self.rgb_frame = None
        self.running = False
        
        self.thermal_width = 256
        self.thermal_height = 192
        self.thermal_scale = 3
        
        self.rgb_width = 640
        self.rgb_height = 480
        
        self.colormap = 0
        self.alpha = 1.0
        self.blur_radius = 0
        self.temperature_threshold = 2
        self.show_hud = True
        
        self.colormaps = [
            (cv2.COLORMAP_JET, 'Jet'),
            (cv2.COLORMAP_HOT, 'Hot'),
            (cv2.COLORMAP_MAGMA, 'Magma'),
            (cv2.COLORMAP_INFERNO, 'Inferno'),
            (cv2.COLORMAP_PLASMA, 'Plasma'),
        ]
        
        self.thermal_frame_count = 0
        self.rgb_frame_count = 0
        
    def start_thermal_ffmpeg(self):
        debug_print(f"=== STARTING THERMAL CAMERA ===")
        
        cmd = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error',
            '-f', 'v4l2',
            '-input_format', 'yuyv422',
            '-video_size', '256x384',
            '-framerate', '25',
            '-i', self.thermal_device_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'yuyv422',
            '-'
        ]
        
        debug_print(f"Thermal command: {' '.join(cmd)}")
        
        try:
            self.thermal_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7
            )
            
            time.sleep(0.5)
            test_data = self.thermal_process.stdout.read(1024)
            if len(test_data) > 0:
                debug_print(f"SUCCESS: Thermal working ({len(test_data)} bytes)")
                return True
            else:
                debug_print(f"ERROR: Thermal not producing data", "ERROR")
                return False
                
        except Exception as e:
            debug_print(f"ERROR: Thermal failed: {e}", "ERROR")
            return False
    
    def start_rgb_direct(self):
        debug_print(f"=== STARTING RGB CAMERA ===")
        debug_print(f"Since it works on Windows/Android, let's make Linux work too")
        
        debug_print(f"Strategy 1: Delayed start...")
        time.sleep(2)
        
        if self.try_rgb_ffmpeg_delayed():
            return True
            
        debug_print(f"Strategy 2: Process isolation...")
        if self.try_rgb_with_isolation():
            return True
            
        debug_print(f"Strategy 3: V4L2 loopback...")
        if self.try_rgb_loopback():
            return True
            
        debug_print(f"ERROR: All RGB strategies failed", "ERROR")
        return False
    
    def try_rgb_ffmpeg_delayed(self):
        debug_print(f"Trying RGB FFmpeg with delay...")
        
        cmd = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'warning',
            '-f', 'v4l2',
            '-input_format', 'mjpeg',
            '-video_size', '640x480',
            '-framerate', '25',
            '-i', self.rgb_device_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ]
        
        debug_print(f"RGB command: {' '.join(cmd)}")
        
        try:
            self.rgb_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**6
            )
            
            debug_print(f"RGB process started, testing...")
            time.sleep(2)
            
            test_data = self.rgb_process.stdout.read(1024)
            if len(test_data) > 0:
                debug_print(f"SUCCESS: RGB FFmpeg working ({len(test_data)} bytes)")
                self.rgb_width = 640
                self.rgb_height = 480
                return True
            else:
                try:
                    stderr_data = self.rgb_process.stderr.read(1024).decode('utf-8', errors='ignore')
                    debug_print(f"RGB stderr: {stderr_data}")
                except:
                    pass
                    
                self.rgb_process.terminate()
                self.rgb_process = None
                debug_print(f"RGB FFmpeg delayed failed")
                return False
                
        except Exception as e:
            debug_print(f"RGB FFmpeg delayed exception: {e}")
            return False
    
    def try_rgb_with_isolation(self):
        debug_print(f"Trying RGB with process isolation...")
        
        wrapper_script = '''#!/bin/bash
exec ffmpeg -hide_banner -loglevel error \\
    -f v4l2 -input_format mjpeg \\
    -video_size 320x240 -framerate 15 \\
    -i /dev/video4 \\
    -f rawvideo -pix_fmt bgr24 -
'''
        
        try:
            with open('/tmp/rgb_wrapper.sh', 'w') as f:
                f.write(wrapper_script)
            os.chmod('/tmp/rgb_wrapper.sh', 0o755)
            
            cmd = [
                'nice', '-n', '10',
                'ionice', '-c', '3',
                '/tmp/rgb_wrapper.sh'
            ]
            
            self.rgb_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**6
            )
            
            time.sleep(2)
            test_data = self.rgb_process.stdout.read(1024)
            
            if len(test_data) > 0:
                debug_print(f"SUCCESS: RGB isolation working ({len(test_data)} bytes)")
                self.rgb_width = 320
                self.rgb_height = 240
                return True
            else:
                self.rgb_process.terminate()
                self.rgb_process = None
                return False
                
        except Exception as e:
            debug_print(f"RGB isolation failed: {e}")
            return False
    
    def try_rgb_loopback(self):
        debug_print(f"Trying V4L2 loopback approach...")
        
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'v4l2loopback' not in result.stdout:
                debug_print(f"V4L2 loopback not available")
                return False
        except:
            return False
        
        debug_print(f"V4L2 loopback requires setup - skipping")
        return False
    
    def thermal_capture_thread(self):
        debug_print(f"=== THERMAL CAPTURE THREAD STARTED ===")
        
        frame_size = 256 * 384 * 2
        
        while self.running:
            try:
                if not self.thermal_process:
                    time.sleep(1)
                    continue
                
                data = self.thermal_process.stdout.read(frame_size)
                
                if len(data) != frame_size:
                    if len(data) == 0:
                        debug_print(f"Thermal process ended", "ERROR")
                        break
                    time.sleep(0.01)
                    continue
                
                frame = np.frombuffer(data, dtype=np.uint8).reshape((384, 256, 2))
                
                processed_frame, raw_frame = self.process_thermal_frame(frame)
                
                if processed_frame is not None and raw_frame is not None:
                    self.thermal_frame = processed_frame
                    self.raw_thermal_frame = raw_frame
                    self.thermal_frame_count += 1
                    
                    if self.thermal_frame_count % 50 == 0:
                        debug_print(f"Thermal frames: {self.thermal_frame_count}")
                
            except Exception as e:
                debug_print(f"Thermal capture error: {e}", "ERROR")
                time.sleep(0.1)
    
    def rgb_capture_thread(self):
        debug_print(f"=== RGB CAPTURE THREAD STARTED ===")
        
        while self.running:
            try:
                if not self.rgb_process:
                    time.sleep(1)
                    continue
                
                frame_size = self.rgb_width * self.rgb_height * 3
                
                data = b''
                attempts = 0
                max_attempts = 10
                
                while len(data) < frame_size and attempts < max_attempts and self.running:
                    remaining = frame_size - len(data)
                    chunk_size = min(remaining, 65536)
                    
                    try:
                        chunk = self.rgb_process.stdout.read(chunk_size)
                        if not chunk:
                            debug_print(f"RGB: No data in chunk, attempt {attempts}", "WARN")
                            attempts += 1
                            time.sleep(0.01)
                            continue
                        data += chunk
                    except Exception as e:
                        debug_print(f"RGB read error: {e}", "ERROR")
                        break
                
                if len(data) != frame_size:
                    debug_print(f"RGB incomplete frame: {len(data)}/{frame_size}, skipping")
                    continue
                
                try:
                    frame = np.frombuffer(data, dtype=np.uint8)
                    
                    expected_pixels = self.rgb_width * self.rgb_height * 3
                    if len(frame) != expected_pixels:
                        debug_print(f"RGB pixel count mismatch: {len(frame)}/{expected_pixels}")
                        continue
                    
                    frame = frame.reshape((self.rgb_height, self.rgb_width, 3))
                    
                except Exception as e:
                    debug_print(f"RGB reshape error: {e}", "ERROR")
                    continue
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Doesnt work
                
                # DEBUG: Try different color conversions if skin appears blueish
                # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Reverse conversion
                frame_bgr = frame[:,:,[0,1,2]]  # Manual channel swap (RGB->BGR)
                
                debug_print(f"RGB color format: {frame.shape}, min: {frame.min()}, max: {frame.max()}")
                
                height, width = frame_bgr.shape[:2]
                split_point = width // 2 + 16
                
                if split_point >= width:
                    split_point = width // 2
                
                left_half = frame_bgr[:, :split_point]
                right_half = frame_bgr[:, split_point:]
                
                frame_bgr = np.hstack((right_half, left_half))
                
                debug_print(f"RGB: Split at pixel {split_point} (not {width//2}), swapped halves")
                
                thermal_display_height = self.thermal_height * self.thermal_scale
                thermal_display_width = self.thermal_width * self.thermal_scale
                
                resized_frame = cv2.resize(frame_bgr, (thermal_display_width, thermal_display_height))
                
                cv2.putText(resized_frame, f"RGB Camera {self.rgb_width}x{self.rgb_height}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                self.rgb_frame = resized_frame
                self.rgb_frame_count += 1
                
                if self.rgb_frame_count % 50 == 0:
                    debug_print(f"RGB frames: {self.rgb_frame_count} - Complete frame: {frame.shape}")
                
            except Exception as e:
                debug_print(f"RGB capture error: {e}", "ERROR")
                time.sleep(0.1)
    
    def process_thermal_frame(self, frame):
        try:
            raw_thermal_data, thermal_process_data = np.array_split(frame, 2)
            temp_info = self.calculate_temperatures(thermal_process_data)
            thermal_display = self.create_thermal_display(raw_thermal_data, temp_info)
            raw_thermal_display = self.create_raw_thermal_data_display(thermal_process_data, temp_info)
            return thermal_display, raw_thermal_display
        except Exception as e:
            debug_print(f"Error processing thermal frame: {e}", "ERROR")
            return None, None
    
    def calculate_temperatures(self, thdata):
        temp_info = {}
        try:
            hi = int(thdata[96][128][0])
            lo = int(thdata[96][128][1]) * 256
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
            
        except Exception as e:
            debug_print(f"Error calculating temperatures: {e}", "ERROR")
            temp_info = {'center': 20.0, 'max': 25.0, 'min': 15.0, 'avg': 20.0, 
                        'max_pos': (128, 96), 'min_pos': (128, 96)}
        
        return temp_info
    
    def create_thermal_display(self, raw_thermal_data, temp_info):
        try:
            bgr = cv2.cvtColor(raw_thermal_data, cv2.COLOR_YUV2BGR_YUYV)
            bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)
            
            thermal_scaled = cv2.resize(bgr, 
                                      (self.thermal_width * self.thermal_scale, 
                                       self.thermal_height * self.thermal_scale), 
                                      interpolation=cv2.INTER_CUBIC)
            
            if self.blur_radius > 0:
                thermal_scaled = cv2.blur(thermal_scaled, (self.blur_radius, self.blur_radius))
            
            colormap_cv, colormap_name = self.colormaps[self.colormap]
            heatmap = cv2.applyColorMap(thermal_scaled, colormap_cv)
            
            self.draw_crosshairs(heatmap, temp_info['center'])
            self.draw_temperature_spots(heatmap, temp_info)
            
            if self.show_hud:
                self.draw_hud(heatmap, temp_info, colormap_name)
            
            cv2.putText(heatmap, f"Thermal Camera - {colormap_name}", 
                       (10, heatmap.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return heatmap
            
        except Exception as e:
            debug_print(f"Error creating thermal display: {e}", "ERROR")
            return None
    
    def create_raw_thermal_data_display(self, thdata, temp_info):
        try:
            hi_data = thdata[..., 0].astype(np.float32)
            lo_data = thdata[..., 1].astype(np.float32) * 256
            temp_data = ((hi_data + lo_data) / 64) - 273.15
            
            raw_scaled = cv2.resize(temp_data, 
                                  (self.thermal_width * self.thermal_scale, 
                                   self.thermal_height * self.thermal_scale), 
                                  interpolation=cv2.INTER_NEAREST)
            
            temp_min, temp_max = temp_data.min(), temp_data.max()
            temp_norm = (raw_scaled - temp_min) / (temp_max - temp_min) if temp_max > temp_min else raw_scaled
            temp_8bit = (temp_norm * 255).astype(np.uint8)
            
            raw_display = cv2.cvtColor(temp_8bit, cv2.COLOR_GRAY2BGR)
            
            cx, cy = raw_display.shape[1] // 2, raw_display.shape[0] // 2
            cv2.line(raw_display, (cx, cy-10), (cx, cy+10), (255, 255, 255), 1)
            cv2.line(raw_display, (cx-10, cy), (cx+10, cy), (255, 255, 255), 1)
            
            temp_text = f"{temp_info['center']:.1f}C"
            cv2.putText(raw_display, temp_text, (cx+15, cy-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(raw_display, "RAW THERMAL DATA (Object Detection)", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(raw_display, f"Temp Range: {temp_min:.1f}C - {temp_max:.1f}C", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(raw_display, f"Avg: {temp_info['avg']:.1f}C  Max: {temp_info['max']:.1f}C  Min: {temp_info['min']:.1f}C", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return raw_display
            
        except Exception as e:
            debug_print(f"Error creating raw thermal display: {e}", "ERROR")
            return None
    
    def draw_crosshairs(self, heatmap, center_temp):
        cx = heatmap.shape[1] // 2
        cy = heatmap.shape[0] // 2
        
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
        if temp_info['max'] > temp_info['avg'] + self.temperature_threshold:
            x, y = temp_info['max_pos']
            x, y = x * self.thermal_scale, y * self.thermal_scale
            cv2.circle(heatmap, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(heatmap, f"{temp_info['max']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f"{temp_info['max']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        if temp_info['min'] < temp_info['avg'] - self.temperature_threshold:
            x, y = temp_info['min_pos']
            x, y = x * self.thermal_scale, y * self.thermal_scale
            cv2.circle(heatmap, (x, y), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(heatmap, f"{temp_info['min']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f"{temp_info['min']}°C", (x+10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    
    def draw_hud(self, heatmap, temp_info, colormap_name):
        cv2.rectangle(heatmap, (0, 60), (220, 220), (0, 0, 0), -1)
        
        y_pos = 76
        line_height = 16
        
        info_lines = [
            f"Avg: {temp_info['avg']}°C",
            f"Max: {temp_info['max']}°C", 
            f"Min: {temp_info['min']}°C",
            f"Colormap: {colormap_name}",
            f"Scale: {self.thermal_scale}x",
            f"Blur: {self.blur_radius}",
            f"Contrast: {self.alpha}",
            f"Threshold: {self.temperature_threshold}",
        ]
        
        for line in info_lines:
            cv2.putText(heatmap, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            y_pos += line_height
    
    def run(self):
        debug_print(f"=== TC001 DIRECT SOLUTION ===")
        debug_print(f"Hardware works fine (Windows/Android proof)")
        debug_print(f"Solving Linux V4L2 driver conflicts directly")
        
        kill_competing_processes()
        reset_video_devices(self.thermal_device_path, self.rgb_device_path)
        
        thermal_ok = self.start_thermal_ffmpeg()
        if not thermal_ok:
            debug_print(f"CRITICAL: Thermal failed", "ERROR")
            return
        
        rgb_ok = self.start_rgb_direct()
        if not rgb_ok:
            debug_print(f"WARNING: RGB failed, continuing thermal-only", "WARN")
        
        debug_print(f"Starting capture threads...")
        self.running = True
        
        thermal_thread = threading.Thread(target=self.thermal_capture_thread, daemon=True)
        thermal_thread.start()
        
        if rgb_ok:
            rgb_thread = threading.Thread(target=self.rgb_capture_thread, daemon=True)
            rgb_thread.start()
        
        cv2.namedWindow('TC001 Thermal Camera', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('TC001 Raw Thermal Data', cv2.WINDOW_AUTOSIZE)
        if rgb_ok:
            cv2.namedWindow('TC001 RGB Camera', cv2.WINDOW_AUTOSIZE)
        
        debug_print(f"=== MAIN LOOP ===")
        debug_print(f"Three windows: Thermal (colorized), Raw Thermal (object detection), RGB")
        debug_print(f"Watch for frame counts!")
        
        try:
            while self.running:
                if self.thermal_frame is not None:
                    cv2.imshow('TC001 Thermal Camera', self.thermal_frame)
                
                if self.raw_thermal_frame is not None:
                    cv2.imshow('TC001 Raw Thermal Data', self.raw_thermal_frame)
                
                if rgb_ok and self.rgb_frame is not None:
                    cv2.imshow('TC001 RGB Camera', self.rgb_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.colormap = (self.colormap + 1) % len(self.colormaps)
                    debug_print(f"Colormap: {self.colormaps[self.colormap][1]}")
                elif key == ord('d'):
                    debug_print(f"=== STATUS ===")
                    debug_print(f"Thermal: {self.thermal_frame_count}")
                    debug_print(f"RGB: {self.rgb_frame_count}")
        
        except KeyboardInterrupt:
            debug_print(f"Shutdown requested")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        debug_print(f"=== CLEANUP ===")
        self.running = False
        
        if self.thermal_process:
            self.thermal_process.terminate()
            try:
                self.thermal_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.thermal_process.kill()
        
        if self.rgb_process:
            self.rgb_process.terminate()
            try:
                self.rgb_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.rgb_process.kill()
        
        cv2.destroyAllWindows()
        debug_print(f"Cleanup complete")

def main():
    if len(sys.argv) > 2:
        thermal_device = int(sys.argv[1])
        rgb_device = int(sys.argv[2])
    else:
        thermal_device = 6
        rgb_device = 4
    
    debug_print(f"TC001 Direct Solution")
    debug_print(f"Thermal: /dev/video{thermal_device}")
    debug_print(f"RGB: /dev/video{rgb_device}")
    debug_print(f"Hardware works fine - fixing Linux driver conflicts")
    
    interface = TC001DirectInterface(thermal_device, rgb_device)
    interface.run()

if __name__ == "__main__":
    main()
