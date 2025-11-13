# """
# WebRTC-Based Interview Recording and Violation Detection System
# FULL BROWSER COMPATIBILITY VERSION
# - Works in Streamlit Cloud (no server camera/mic needed)
# - Captures video/audio from user's browser
# - Maintains ALL original functionality
# - Fixed cv2.FONT_HERSHEY_BOLD error (use FONT_HERSHEY_SIMPLEX)
# """

# import cv2
# import numpy as np
# import threading
# import time
# import tempfile
# import os
# import speech_recognition as sr
# import warnings
# from collections import deque
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
# import av
# import queue

# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # WebRTC Configuration for Streamlit Cloud
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )
# # RTC_CONFIGURATION = RTCConfiguration({
# #     "iceServers": [
# #         # Keep the STUN server
# #         {"urls": ["stun:stun.l.google.com:19302"]},
        
# #         # ADD YOUR TURN SERVER CREDENTIALS
# #         {
# #             "urls": ["turn:your-turn-server-address.com:3478"],
# #             "username": "your-turn-username",
# #             "credential": "your-turn-password"
# #         }
# #     ]
# # })
# class InterviewVideoProcessor(VideoTransformerBase):
#     """
#     Real-time video processor for interview recording
#     Handles frame capture, violation detection, and display
#     """
    
#     def __init__(self, models_dict, frame_margin=50):
#         self.models = models_dict
#         self.frame_margin = frame_margin
        
#         # Recording state
#         self.is_recording = False
#         self.frames_buffer = deque(maxlen=600)  # 20 seconds at 30fps
#         self.current_frames = []
        
#         # Violation detection state
#         self.violation_detected = False
#         self.violation_reason = ""
#         self.violation_frame = None
        
#         # Baseline environment
#         self.baseline_environment = None
#         self.baseline_set = False
        
#         # Face tracking
#         self.face_box = None
#         self.no_face_start = None
#         self.look_away_start = None
#         self.blink_count = 0
#         self.prev_blink = False
#         self.eye_contact_frames = 0
#         self.total_frames = 0
        
#         # Status display
#         self.status_text = "Initializing..."
#         self.attention_status = "No Face"
#         self.lighting_status = "Unknown"
        
#         # Question info
#         self.current_question_num = 0
#         self.total_questions = 0
#         self.recording_start_time = None
#         self.question_duration = 20
        
#         # Setup phase
#         self.in_setup_phase = False
#         self.setup_stable_frames = 0
#         self.setup_required_frames = 30
        
#         # Initialize pose detection
#         try:
#             import mediapipe as mp
#             self.mp_pose = mp.solutions.pose
#             self.pose_detector = self.mp_pose.Pose(
#                 static_image_mode=False,
#                 model_complexity=1,
#                 smooth_landmarks=True,
#                 min_detection_confidence=0.5,
#                 min_tracking_confidence=0.5
#             )
#             self.pose_available = True
#         except:
#             self.pose_detector = None
#             self.pose_available = False
    
#     def transform(self, frame):
#         """Process each video frame from browser"""
#         img = frame.to_ndarray(format="bgr24")
#         h, w = img.shape[:2]
        
#         # Always store frame if recording
#         if self.is_recording:
#             self.frames_buffer.append(img.copy())
#             self.current_frames.append(img.copy())
        
#         # Setup phase - just show boundaries and instructions
#         if self.in_setup_phase:
#             return self._process_setup_frame(img)
        
#         # Recording phase - full violation detection
#         if self.is_recording:
#             img = self._process_recording_frame(img)
        
#         return av.VideoFrame.from_ndarray(img, format="bgr24")
    
#     def _process_setup_frame(self, frame):
#         """Process frame during setup phase"""
#         h, w = frame.shape[:2]
        
#         # Draw boundaries
#         frame_with_boundaries = self.draw_frame_boundaries(frame)
        
#         # Analyze face position
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         is_ready = False
#         status_color = (255, 165, 0)  # Orange
        
#         if self.models['face_mesh'] is not None:
#             face_results = self.models['face_mesh'].process(rgb_frame)
            
#             if face_results.multi_face_landmarks:
#                 num_faces = len(face_results.multi_face_landmarks)
                
#                 if num_faces > 1:
#                     self.status_text = "⚠️ Multiple faces detected! Only ONE person allowed"
#                     status_color = (0, 0, 255)
#                     self.setup_stable_frames = 0
                
#                 elif num_faces == 1:
#                     face_landmarks = face_results.multi_face_landmarks[0]
                    
#                     # Get face bounding box
#                     landmarks_2d = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
#                     x_coords = landmarks_2d[:, 0]
#                     y_coords = landmarks_2d[:, 1]
#                     self.face_box = (int(np.min(x_coords)), int(np.min(y_coords)), 
#                                    int(np.max(x_coords) - np.min(x_coords)), 
#                                    int(np.max(y_coords) - np.min(y_coords)))
                    
#                     # Check boundaries
#                     within_bounds, boundary_msg, boundary_status = self.check_frame_boundaries(frame, self.face_box)
                    
#                     # Check for others outside frame
#                     outside_detected, obj_type, location = self.detect_person_outside_frame(frame)
                    
#                     if outside_detected:
#                         self.status_text = f"⚠️ {obj_type.upper()} detected outside frame ({location} side)!"
#                         status_color = (0, 0, 255)
#                         self.setup_stable_frames = 0
#                     elif not within_bounds:
#                         self.status_text = f"⚠️ {boundary_msg} - Please adjust!"
#                         status_color = (0, 0, 255)
#                         self.setup_stable_frames = 0
                        
#                         # Highlight violated boundary
#                         if boundary_status == "LEFT_VIOLATION":
#                             cv2.rectangle(frame_with_boundaries, (0, 0), (self.frame_margin, h), (0, 0, 255), -1)
#                         elif boundary_status == "RIGHT_VIOLATION":
#                             cv2.rectangle(frame_with_boundaries, (w - self.frame_margin, 0), (w, h), (0, 0, 255), -1)
#                         elif boundary_status == "TOP_VIOLATION":
#                             cv2.rectangle(frame_with_boundaries, (0, 0), (w, self.frame_margin), (0, 0, 255), -1)
#                     else:
#                         # Good position!
#                         self.setup_stable_frames += 1
#                         progress = min(100, int((self.setup_stable_frames / self.setup_required_frames) * 100))
#                         self.status_text = f"✅ Good position! Hold steady... {progress}%"
#                         status_color = (0, 255, 0)
                        
#                         if self.setup_stable_frames >= self.setup_required_frames:
#                             is_ready = True
#             else:
#                 self.status_text = "❌ No face detected - Please position yourself in frame"
#                 status_color = (0, 0, 255)
#                 self.setup_stable_frames = 0
        
#         # Draw status overlay
#         overlay_height = 140
#         overlay = frame_with_boundaries.copy()
#         cv2.rectangle(overlay, (0, h - overlay_height), (w, h), (0, 0, 0), -1)
#         frame_with_boundaries = cv2.addWeighted(frame_with_boundaries, 0.7, overlay, 0.3, 0)
        
#         cv2.putText(frame_with_boundaries, self.status_text, (10, h - 110),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
#         cv2.putText(frame_with_boundaries, "Instructions:", (10, h - 80),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         cv2.putText(frame_with_boundaries, "• Keep your face within GREEN boundaries", (10, h - 60),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#         cv2.putText(frame_with_boundaries, "• Ensure no one else is visible", (10, h - 40),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#         cv2.putText(frame_with_boundaries, "• Remove all unauthorized items from view", (10, h - 20),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
#         return frame_with_boundaries
    
#     def _process_recording_frame(self, frame):
#         """Process frame during recording with full violation detection"""
#         h, w = frame.shape[:2]
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         self.total_frames += 1
        
#         # Analyze lighting
#         self.lighting_status, brightness = self.analyze_lighting(frame)
        
#         # Face detection and violation checks
#         num_faces = 0
#         looking_at_camera = False
        
#         if self.models['face_mesh'] is not None:
#             face_results = self.models['face_mesh'].process(rgb_frame)
            
#             if face_results.multi_face_landmarks:
#                 num_faces = len(face_results.multi_face_landmarks)
                
#                 # Check multiple bodies
#                 is_multi_body, multi_msg, body_count = self.detect_multiple_bodies(frame, num_faces)
#                 if is_multi_body:
#                     self.violation_detected = True
#                     self.violation_reason = multi_msg
#                     self.violation_frame = frame.copy()
#                     self.attention_status = "VIOLATION"
#                     return self._add_violation_overlay(frame, multi_msg)
                
#                 # Multiple faces
#                 if num_faces > 1:
#                     violation_msg = f"Multiple persons detected ({num_faces} faces)"
#                     self.violation_detected = True
#                     self.violation_reason = violation_msg
#                     self.violation_frame = frame.copy()
#                     self.attention_status = "VIOLATION"
#                     return self._add_violation_overlay(frame, violation_msg)
                
#                 # Single face - full analysis
#                 elif num_faces == 1:
#                     self.no_face_start = None
#                     face_landmarks = face_results.multi_face_landmarks[0]
                    
#                     # Get face box
#                     landmarks_2d = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
#                     x_coords = landmarks_2d[:, 0]
#                     y_coords = landmarks_2d[:, 1]
#                     self.face_box = (int(np.min(x_coords)), int(np.min(y_coords)), 
#                                    int(np.max(x_coords) - np.min(x_coords)), 
#                                    int(np.max(y_coords) - np.min(y_coords)))
                    
#                     # Check boundaries
#                     within_bounds, boundary_msg, boundary_status = self.check_frame_boundaries(frame, self.face_box)
#                     if not within_bounds:
#                         self.violation_detected = True
#                         self.violation_reason = boundary_msg
#                         self.violation_frame = frame.copy()
#                         self.attention_status = "VIOLATION"
#                         return self._add_violation_overlay(frame, boundary_msg)
                    
#                     # Check person outside frame
#                     outside_detected, obj_type, location = self.detect_person_outside_frame(frame)
#                     if outside_detected:
#                         violation_msg = f"{obj_type.upper()} detected outside frame ({location} side)"
#                         self.violation_detected = True
#                         self.violation_reason = violation_msg
#                         self.violation_frame = frame.copy()
#                         self.attention_status = "VIOLATION"
#                         return self._add_violation_overlay(frame, violation_msg)
                    
#                     # Check intrusions
#                     is_intrusion, intrusion_msg = self.detect_intrusion_at_edges(frame, self.face_box)
#                     if is_intrusion:
#                         self.violation_detected = True
#                         self.violation_reason = intrusion_msg
#                         self.violation_frame = frame.copy()
#                         self.attention_status = "VIOLATION"
#                         return self._add_violation_overlay(frame, intrusion_msg)
                    
#                     # Check hands outside
#                     is_hand_violation, hand_msg = self.detect_hands_outside_main_person(frame, self.face_box)
#                     if is_hand_violation:
#                         self.violation_detected = True
#                         self.violation_reason = hand_msg
#                         self.violation_frame = frame.copy()
#                         self.attention_status = "VIOLATION"
#                         return self._add_violation_overlay(frame, hand_msg)
                    
#                     # Check suspicious movements
#                     is_suspicious, sus_msg = self.detect_suspicious_movements(frame)
#                     if is_suspicious:
#                         self.violation_detected = True
#                         self.violation_reason = sus_msg
#                         self.violation_frame = frame.copy()
#                         self.attention_status = "VIOLATION"
#                         return self._add_violation_overlay(frame, sus_msg)
                    
#                     # Head pose and gaze
#                     yaw, pitch, roll = self.estimate_head_pose(face_landmarks, frame.shape)
#                     gaze_centered = self.calculate_eye_gaze(face_landmarks, frame.shape)
                    
#                     # Blink detection
#                     is_blink = self.detect_blink(face_landmarks)
#                     if is_blink and not self.prev_blink:
#                         self.blink_count += 1
#                     self.prev_blink = is_blink
                    
#                     # Check attention
#                     head_looking_forward = abs(yaw) <= 20 and abs(pitch) <= 20
                    
#                     if head_looking_forward and gaze_centered:
#                         self.look_away_start = None
#                         looking_at_camera = True
#                         self.eye_contact_frames += 1
#                         self.attention_status = "Looking at Camera ✓"
#                     else:
#                         if self.look_away_start is None:
#                             self.look_away_start = time.time()
#                             self.attention_status = "Looking Away"
#                         else:
#                             elapsed = time.time() - self.look_away_start
#                             if elapsed > 2.0:
#                                 violation_msg = "Looking away for >2 seconds"
#                                 self.violation_detected = True
#                                 self.violation_reason = violation_msg
#                                 self.violation_frame = frame.copy()
#                                 self.attention_status = "VIOLATION"
#                                 return self._add_violation_overlay(frame, violation_msg)
#                             else:
#                                 self.attention_status = f"Looking Away ({elapsed:.1f}s)"
#             else:
#                 # No face detected
#                 if self.no_face_start is None:
#                     self.no_face_start = time.time()
#                     self.attention_status = "No Face Visible"
#                 else:
#                     elapsed = time.time() - self.no_face_start
#                     if elapsed > 2.0:
#                         violation_msg = "No face visible for >2 seconds"
#                         self.violation_detected = True
#                         self.violation_reason = violation_msg
#                         self.violation_frame = frame.copy()
#                         self.attention_status = "VIOLATION"
#                         return self._add_violation_overlay(frame, violation_msg)
#                     else:
#                         self.attention_status = f"No Face ({elapsed:.1f}s)"
        
#         # Check for new objects (every 20 frames)
#         if self.total_frames % 20 == 0 and self.baseline_set:
#             new_detected, new_items = self.detect_new_objects(frame)
#             if new_detected:
#                 violation_msg = f"New item(s) brought into view: {', '.join(new_items)}"
#                 self.violation_detected = True
#                 self.violation_reason = violation_msg
#                 self.violation_frame = frame.copy()
#                 self.attention_status = "VIOLATION"
#                 return self._add_violation_overlay(frame, violation_msg)
        
#         # Add status overlay (no violation)
#         return self._add_status_overlay(frame)
    
#     def _add_status_overlay(self, frame):
#         """Add status information overlay to frame"""
#         h, w = frame.shape[:2]
        
#         # Create semi-transparent black bar at top
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
#         frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
#         # Status color
#         status_color = (0, 255, 0) if not self.violation_detected else (0, 165, 255)
        
#         # Question info
#         if self.recording_start_time:
#             elapsed = time.time() - self.recording_start_time
#             remaining = max(0, int(self.question_duration - elapsed))
#             time_text = f"Time: {remaining}s"
#         else:
#             time_text = "Ready"
        
#         cv2.putText(frame, f"Q{self.current_question_num}/{self.total_questions} - {self.attention_status}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
#         cv2.putText(frame, f"Lighting: {self.lighting_status}", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
#         eye_contact_pct = int((self.eye_contact_frames / max(self.total_frames, 1)) * 100)
#         cv2.putText(frame, f"Eye Contact: {eye_contact_pct}%", 
#                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         cv2.putText(frame, time_text, 
#                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
#         return frame
    
#     def _add_violation_overlay(self, frame, violation_msg):
#         """Add red violation overlay to frame"""
#         h, w = frame.shape[:2]
        
#         # Red tinted overlay
#         red_overlay = frame.copy()
#         cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 255), -1)
#         frame = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
        
#         # Red border
#         cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
        
#         # Violation text
#         cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
#         cv2.putText(frame, "VIOLATION DETECTED", (w//2 - 200, 50),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
#         # Violation reason at bottom
#         cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 0), -1)
        
#         # Split long text
#         words = violation_msg.split()
#         lines = []
#         current_line = ""
#         for word in words:
#             test_line = current_line + " " + word if current_line else word
#             if len(test_line) > 50:
#                 lines.append(current_line)
#                 current_line = word
#             else:
#                 current_line = test_line
#         if current_line:
#             lines.append(current_line)
        
#         y_offset = h - 90
#         for line in lines[:2]:
#             cv2.putText(frame, line, (10, y_offset),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             y_offset += 30
        
#         return frame
    
#     def start_recording(self, question_num, total_questions, duration):
#         """Start recording for a question"""
#         self.is_recording = True
#         self.current_question_num = question_num
#         self.total_questions = total_questions
#         self.question_duration = duration
#         self.recording_start_time = time.time()
#         self.current_frames = []
#         self.frames_buffer.clear()
#         self.violation_detected = False
#         self.violation_reason = ""
#         self.violation_frame = None
#         self.total_frames = 0
#         self.eye_contact_frames = 0
#         self.blink_count = 0
#         self.no_face_start = None
#         self.look_away_start = None
    
#     def stop_recording(self):
#         """Stop recording and return collected frames"""
#         self.is_recording = False
#         frames = list(self.current_frames)
#         self.current_frames = []
#         return frames
    
#     def set_baseline_environment(self, frame):
#         """Set baseline environment from frame"""
#         self.baseline_environment = self.scan_environment(frame)
#         self.baseline_set = True
    
#     def start_setup_phase(self):
#         """Start the setup phase"""
#         self.in_setup_phase = True
#         self.setup_stable_frames = 0
    
#     def is_setup_complete(self):
#         """Check if setup is complete"""
#         return self.setup_stable_frames >= self.setup_required_frames
    
#     def end_setup_phase(self):
#         """End setup phase"""
#         self.in_setup_phase = False
    
#     # ==================== HELPER METHODS (from original Recording_system.py) ====================
    
#     def draw_frame_boundaries(self, frame):
#         """Draw visible frame boundaries"""
#         h, w = frame.shape[:2]
#         margin = self.frame_margin
        
#         overlay = frame.copy()
        
#         cv2.line(overlay, (margin, 0), (margin, h), (0, 255, 0), 3)
#         cv2.line(overlay, (w - margin, 0), (w - margin, h), (0, 255, 0), 3)
#         cv2.line(overlay, (0, margin), (w, margin), (0, 255, 0), 3)
#         cv2.rectangle(overlay, (margin, margin), (w - margin, h), (0, 255, 0), 2)
        
#         frame_with_boundaries = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
#         corner_size = 30
#         cv2.line(frame_with_boundaries, (margin, margin), (margin + corner_size, margin), (0, 255, 0), 3)
#         cv2.line(frame_with_boundaries, (margin, margin), (margin, margin + corner_size), (0, 255, 0), 3)
        
#         cv2.line(frame_with_boundaries, (w - margin, margin), (w - margin - corner_size, margin), (0, 255, 0), 3)
#         cv2.line(frame_with_boundaries, (w - margin, margin), (w - margin, margin + corner_size), (0, 255, 0), 3)
        
#         cv2.putText(frame_with_boundaries, "Stay within GREEN boundaries", 
#                     (w//2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         return frame_with_boundaries
    
#     def check_frame_boundaries(self, frame, face_box):
#         """Check if person is within frame boundaries"""
#         if face_box is None:
#             return False, "No face detected", "NO_FACE"
        
#         h, w = frame.shape[:2]
#         margin = self.frame_margin
#         x, y, fw, fh = face_box
        
#         face_left = x
#         face_right = x + fw
#         face_top = y
        
#         if face_left < margin:
#             return False, "Person too close to LEFT edge", "LEFT_VIOLATION"
#         if face_right > (w - margin):
#             return False, "Person too close to RIGHT edge", "RIGHT_VIOLATION"
#         if face_top < margin:
#             return False, "Person too close to TOP edge", "TOP_VIOLATION"
        
#         return True, "Within boundaries", "OK"
    
#     def analyze_lighting(self, frame):
#         """Analyze lighting conditions"""
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         mean_brightness = np.mean(gray)
#         std_brightness = np.std(gray)
        
#         if mean_brightness < 60:
#             return "Too Dark", mean_brightness
#         elif mean_brightness > 200:
#             return "Too Bright", mean_brightness
#         elif std_brightness < 25:
#             return "Low Contrast", mean_brightness
#         else:
#             return "Good", mean_brightness
    
#     def detect_blink(self, face_landmarks):
#         """Detect if eye is blinking"""
#         upper_lid = face_landmarks.landmark[159]
#         lower_lid = face_landmarks.landmark[145]
#         eye_openness = abs(upper_lid.y - lower_lid.y)
#         return eye_openness < 0.01
    
#     def calculate_eye_gaze(self, face_landmarks, frame_shape):
#         """Calculate if eyes are looking at camera"""
#         left_eye_indices = [468, 469, 470, 471, 472]
#         right_eye_indices = [473, 474, 475, 476, 477]
#         left_eye_center = [33, 133, 157, 158, 159, 160, 161, 163, 144, 145, 153, 154, 155]
#         right_eye_center = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 373, 374, 390]
        
#         landmarks = face_landmarks.landmark
        
#         left_iris_x = np.mean([landmarks[i].x for i in left_eye_indices if i < len(landmarks)])
#         left_eye_x = np.mean([landmarks[i].x for i in left_eye_center if i < len(landmarks)])
        
#         right_iris_x = np.mean([landmarks[i].x for i in right_eye_indices if i < len(landmarks)])
#         right_eye_x = np.mean([landmarks[i].x for i in right_eye_center if i < len(landmarks)])
        
#         left_gaze_ratio = (left_iris_x - left_eye_x) if left_iris_x and left_eye_x else 0
#         right_gaze_ratio = (right_iris_x - right_eye_x) if right_iris_x and right_eye_x else 0
        
#         avg_gaze = (left_gaze_ratio + right_gaze_ratio) / 2
        
#         return abs(avg_gaze) < 0.02
    
#     def estimate_head_pose(self, face_landmarks, frame_shape):
#         """Estimate head pose angles"""
#         h, w = frame_shape[:2]
#         landmarks_3d = np.array([(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark])
        
#         required_indices = [1, 33, 263, 61, 291]
#         image_points = np.array([landmarks_3d[i] for i in required_indices], dtype="double")
        
#         model_points = np.array([
#             (0.0, 0.0, 0.0), (-30.0, -125.0, -30.0),
#             (30.0, -125.0, -30.0), (-60.0, -70.0, -60.0),
#             (60.0, -70.0, -60.0)
#         ])
        
#         focal_length = w
#         center = (w / 2, h / 2)
#         camera_matrix = np.array([
#             [focal_length, 0, center[0]],
#             [0, focal_length, center[1]],
#             [0, 0, 1]
#         ], dtype="double")
#         dist_coeffs = np.zeros((4, 1))
        
#         success, rotation_vector, _ = cv2.solvePnP(
#             model_points, image_points, camera_matrix, 
#             dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
#         )
        
#         if success:
#             rmat, _ = cv2.Rodrigues(rotation_vector)
#             pose_mat = cv2.hconcat((rmat, rotation_vector))
#             _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
#             yaw, pitch, roll = [float(a) for a in euler]
#             return yaw, pitch, roll
        
#         return 0, 0, 0
    
#     def scan_environment(self, frame):
#         """Scan and catalog the environment"""
#         if self.models['yolo'] is None:
#             return {'objects': [], 'positions': []}
        
#         try:
#             results = self.models['yolo'].predict(frame, conf=0.25, verbose=False)
            
#             environment_data = {
#                 'objects': [],
#                 'positions': [],
#                 'person_position': None
#             }
            
#             if results and len(results) > 0:
#                 names = self.models['yolo'].names
#                 boxes = results[0].boxes
                
#                 for box in boxes:
#                     cls_id = int(box.cls[0])
#                     obj_name = names[cls_id]
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
#                     environment_data['objects'].append(obj_name)
#                     environment_data['positions'].append({
#                         'name': obj_name,
#                         'bbox': (int(x1), int(y1), int(x2), int(y2)),
#                         'center': (int((x1+x2)/2), int((y1+y2)/2))
#                     })
                    
#                     if obj_name == 'person':
#                         environment_data['person_position'] = (int((x1+x2)/2), int((y1+y2)/2))
            
#             return environment_data
            
#         except Exception as e:
#             return {'objects': [], 'positions': []}
    
#     def detect_new_objects(self, frame):
#         """Detect NEW objects that weren't in baseline environment"""
#         if self.models['yolo'] is None or self.baseline_environment is None:
#             return False, []
        
#         try:
#             results = self.models['yolo'].predict(frame, conf=0.25, verbose=False)
            
#             if results and len(results) > 0:
#                 names = self.models['yolo'].names
#                 boxes = results[0].boxes
                
#                 current_objects = []
#                 for box in boxes:
#                     cls_id = int(box.cls[0])
#                     obj_name = names[cls_id]
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                     current_center = (int((x1+x2)/2), int((y1+y2)/2))
                    
#                     current_objects.append({
#                         'name': obj_name,
#                         'center': current_center,
#                         'bbox': (int(x1), int(y1), int(x2), int(y2))
#                     })
                
#                 baseline_objects = self.baseline_environment['positions']
#                 new_items = []
                
#                 for curr_obj in current_objects:
#                     if curr_obj['name'] == 'person':
#                         continue
                    
#                     is_baseline = False
#                     for base_obj in baseline_objects:
#                         if curr_obj['name'] == base_obj['name']:
#                             dist = np.sqrt(
#                                 (curr_obj['center'][0] - base_obj['center'][0])**2 +
#                                 (curr_obj['center'][1] - base_obj['center'][1])**2
#                             )
#                             if dist < 100:
#                                 is_baseline = True
#                                 break
                    
#                     if not is_baseline:
#                         new_items.append(curr_obj['name'])
                
#                 if new_items:
#                     return True, list(set(new_items))
            
#             return False, []
            
#         except Exception as e:
#             return False, []
    
#     def detect_person_outside_frame(self, frame):
#         """Detect if any person/living being is outside boundaries"""
#         if self.models['yolo'] is None:
#             return False, "", ""
        
#         h, w = frame.shape[:2]
#         margin = self.frame_margin
        
#         try:
#             results = self.models['yolo'].predict(frame, conf=0.4, verbose=False)
            
#             if results and len(results) > 0:
#                 names = self.models['yolo'].names
#                 boxes = results[0].boxes
                
#                 living_beings = ['person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 
#                                 'elephant', 'bear', 'zebra', 'giraffe']
                
#                 for i, box in enumerate(boxes):
#                     cls_id = int(box.cls[0])
#                     obj_name = names[cls_id]
                    
#                     if obj_name.lower() in living_beings:
#                         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
#                         if x1 < margin or x2 < margin:
#                             return True, obj_name, "LEFT"
#                         if x1 > (w - margin) or x2 > (w - margin):
#                             return True, obj_name, "RIGHT"
#                         if y1 < margin or y2 < margin:
#                             return True, obj_name, "TOP"
        
#         except Exception as e:
#             pass
        
#         return False, "", ""
    
#     def detect_multiple_bodies(self, frame, num_faces):
#         """Detect multiple bodies using pose and hand detection"""
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         body_count = 0
#         detected_parts = []
        
#         if self.pose_available and self.pose_detector:
#             try:
#                 pose_results = self.pose_detector.process(rgb_frame)
                
#                 if pose_results.pose_landmarks:
#                     body_count += 1
#                     detected_parts.append("body")
                    
#                     landmarks = pose_results.pose_landmarks.landmark
                    
#                     visible_shoulders = sum(1 for idx in [11, 12] 
#                                           if landmarks[idx].visibility > 0.5)
#                     visible_elbows = sum(1 for idx in [13, 14] 
#                                         if landmarks[idx].visibility > 0.5)
                    
#                     if visible_shoulders > 2 or visible_elbows > 2:
#                         return True, "Multiple body parts detected (extra shoulders/arms)", body_count + 1
                        
#             except Exception as e:
#                 pass
        
#         if self.models['hands'] is not None:
#             try:
#                 hand_results = self.models['hands'].process(rgb_frame)
                
#                 if hand_results.multi_hand_landmarks:
#                     num_hands = len(hand_results.multi_hand_landmarks)
                    
#                     if num_hands > 2:
#                         detected_parts.append(f"{num_hands} hands")
#                         return True, f"Multiple persons detected ({num_hands} hands visible)", 2
                    
#                     if num_hands == 2:
#                         hand1 = hand_results.multi_hand_landmarks[0].landmark[0]
#                         hand2 = hand_results.multi_hand_landmarks[1].landmark[0]
                        
#                         distance = np.sqrt((hand1.x - hand2.x)**2 + (hand1.y - hand2.y)**2)
                        
#                         if distance > 0.7:
#                             detected_parts.append("widely separated hands")
#                             return True, "Suspicious hand positions (possible multiple persons)", 2
                            
#             except Exception as e:
#                 pass
        
#         if num_faces == 1 and body_count > 1:
#             return True, "Body parts from multiple persons detected", 2
        
#         if num_faces > 1:
#             return True, f"Multiple persons detected ({num_faces} faces)", num_faces
        
#         return False, "", max(num_faces, body_count)
    
#     def detect_hands_outside_main_person(self, frame, face_box):
#         """Detect hands outside main person's area"""
#         if self.models['hands'] is None or face_box is None:
#             return False, ""
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w = frame.shape[:2]
        
#         try:
#             hand_results = self.models['hands'].process(rgb_frame)
            
#             if hand_results.multi_hand_landmarks:
#                 x, y, fw, fh = face_box
                
#                 expected_left = max(0, x - fw)
#                 expected_right = min(w, x + fw * 2)
#                 expected_top = max(0, y - fh)
#                 expected_bottom = min(h, y + fh * 4)
                
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     hand_x = hand_landmarks.landmark[0].x * w
#                     hand_y = hand_landmarks.landmark[0].y * h
                    
#                     if (hand_x < expected_left - 50 or hand_x > expected_right + 50 or
#                         hand_y < expected_top - 50 or hand_y > expected_bottom + 50):
#                         return True, "Hand detected outside main person's area"
                
#         except Exception as e:
#             pass
        
#         return False, ""
    
#     def detect_suspicious_movements(self, frame):
#         """Detect suspicious hand movements"""
#         if self.models['hands'] is None:
#             return False, ""
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w = frame.shape[:2]
        
#         try:
#             hand_results = self.models['hands'].process(rgb_frame)
            
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     wrist = hand_landmarks.landmark[0]
#                     index_tip = hand_landmarks.landmark[8]
                    
#                     wrist_y = wrist.y * h
#                     tip_y = index_tip.y * h
                    
#                     if wrist_y > h * 0.75:
#                         return True, "Hand movement below desk level detected"
                    
#                     if wrist_y < h * 0.15:
#                         return True, "Suspicious hand movement at top of frame"
        
#         except Exception as e:
#             pass
        
#         return False, ""
    
#     def has_skin_tone(self, region):
#         """Check if region contains skin-like colors"""
#         if region.size == 0:
#             return False
        
#         hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
#         lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
#         upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
#         lower_skin2 = np.array([0, 20, 0], dtype=np.uint8)
#         upper_skin2 = np.array([20, 150, 255], dtype=np.uint8)
        
#         mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
#         mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
#         mask = cv2.bitwise_or(mask1, mask2)
        
#         skin_ratio = np.sum(mask > 0) / mask.size
#         return skin_ratio > 0.3
    
#     def detect_intrusion_at_edges(self, frame, face_box):
#         """Detect body parts intruding from frame edges"""
#         if face_box is None:
#             return False, ""
        
#         h, w = frame.shape[:2]
#         x, y, fw, fh = face_box
        
#         edge_width = 80
        
#         left_region = frame[:, :edge_width]
#         right_region = frame[:, w-edge_width:]
#         top_left = frame[:edge_width, :w//3]
#         top_right = frame[:edge_width, 2*w//3:]
        
#         face_center_x = x + fw // 2
#         face_far_from_left = face_center_x > w * 0.3
#         face_far_from_right = face_center_x < w * 0.7
        
#         if face_far_from_left and self.has_skin_tone(left_region):
#             if self.models['hands']:
#                 rgb_region = cv2.cvtColor(left_region, cv2.COLOR_BGR2RGB)
#                 try:
#                     result = self.models['hands'].process(rgb_region)
#                     if result.multi_hand_landmarks:
#                         return True, "Body part detected at left edge (another person)"
#                 except:
#                     pass
        
#         if face_far_from_right and self.has_skin_tone(right_region):
#             if self.models['hands']:
#                 rgb_region = cv2.cvtColor(right_region, cv2.COLOR_BGR2RGB)
#                 try:
#                     result = self.models['hands'].process(rgb_region)
#                     if result.multi_hand_landmarks:
#                         return True, "Body part detected at right edge (another person)"
#                 except:
#                     pass
        
#         if y > h * 0.2:
#             if self.has_skin_tone(top_left) or self.has_skin_tone(top_right):
#                 return True, "Body part detected at top edge (another person)"
        
#         return False, ""


# class RecordingSystemWebRTC:
#     """WebRTC-based Recording System - Browser Compatible"""
    
#     def __init__(self, models_dict):
#         self.models = models_dict
#         self.violation_images_dir = tempfile.mkdtemp(prefix="violations_")
    
#     def save_violation_image(self, frame, question_number, violation_reason):
#         """Save violation image with overlay"""
#         try:
#             timestamp = int(time.time() * 1000)
#             filename = f"violation_q{question_number}_{timestamp}.jpg"
#             filepath = os.path.join(self.violation_images_dir, filename)
            
#             overlay_frame = frame.copy()
#             h, w = overlay_frame.shape[:2]
            
#             # Red overlay
#             red_overlay = overlay_frame.copy()
#             cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 255), -1)
#             overlay_frame = cv2.addWeighted(overlay_frame, 0.7, red_overlay, 0.3, 0)
            
#             # Red border
#             cv2.rectangle(overlay_frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            
#             # Title
#             text = "VIOLATION DETECTED"
#             cv2.rectangle(overlay_frame, (0, 0), (w, 80), (0, 0, 0), -1)
#             cv2.putText(overlay_frame, text, (w//2 - 200, 50),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
#             # Reason
#             cv2.rectangle(overlay_frame, (0, h-100), (w, h), (0, 0, 0), -1)
#             words = violation_reason.split()
#             lines = []
#             current_line = ""
#             for word in words:
#                 test_line = current_line + " " + word if current_line else word
#                 if len(test_line) > 50:
#                     lines.append(current_line)
#                     current_line = word
#                 else:
#                     current_line = test_line
#             if current_line:
#                 lines.append(current_line)
            
#             y_offset = h - 90
#             for line in lines[:2]:
#                 cv2.putText(overlay_frame, line, (10, y_offset),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                 y_offset += 30
            
#             cv2.imwrite(filepath, overlay_frame)
#             return filepath
            
#         except Exception as e:
#             print(f"Error saving violation image: {e}")
#             return None
    
#     def transcribe_audio_from_bytes(self, audio_bytes):
#         """Transcribe audio from bytes"""
#         r = sr.Recognizer()
#         try:
#             # Save bytes to temp file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#                 f.write(audio_bytes)
#                 temp_path = f.name
            
#             with sr.AudioFile(temp_path) as source:
#                 audio = r.record(source)
            
#             text = r.recognize_google(audio)
#             os.unlink(temp_path)
#             return text if text.strip() else "[Could not understand audio]"
#         except sr.UnknownValueError:
#             return "[Could not understand audio]"
#         except sr.RequestError:
#             return "[Speech recognition service unavailable]"
#         except Exception as e:
#             return "[Could not understand audio]"
    
#     def record_continuous_interview_webrtc(self, questions_list, duration_per_question, ui_callbacks):
#         """
#         WebRTC-based continuous interview recording
#         Uses Streamlit components for browser video/audio capture
#         """
        
#         st.markdown("---")
#         st.subheader("🎬 Interview Session")
        
#         # Create containers for dynamic content
#         video_container = st.container()
#         status_container = st.container()
#         control_container = st.container()
        
#         with video_container:
#             st.markdown("### 📹 Camera View")
#             video_placeholder = st.empty()
        
#         with status_container:
#             status_col1, status_col2, status_col3 = st.columns(3)
#             with status_col1:
#                 status_text = st.empty()
#             with status_col2:
#                 timer_text = st.empty()
#             with status_col3:
#                 progress_text = st.empty()
        
#         # Initialize video processor
#         processor = InterviewVideoProcessor(self.models)
        
#         # ========== SETUP PHASE ==========
#         st.info("🔧 **Step 1: Position Setup** - Adjust your position within the GREEN boundaries")
        
#         processor.start_setup_phase()
        
#         # WebRTC streamer for setup
#         setup_ctx = webrtc_streamer(
#             key="setup_phase",
#             video_processor_factory=lambda: processor,
#             rtc_configuration=RTC_CONFIGURATION,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True,
#         )
        
#         # Wait for setup completion
#         with control_container:
#             setup_status = st.empty()
#             setup_status.warning("⏳ Adjusting position... Please stay within boundaries")
        
#         setup_complete = False
#         setup_timeout = 90
#         setup_start = time.time()
        
#         while not setup_complete and (time.time() - setup_start) < setup_timeout:
#             if setup_ctx.video_processor and setup_ctx.video_processor.is_setup_complete():
#                 setup_complete = True
#                 setup_status.success("✅ Setup complete! Scanning environment...")
                
#                 # Get a frame for baseline
#                 if len(processor.frames_buffer) > 0:
#                     baseline_frame = list(processor.frames_buffer)[-1]
#                     processor.set_baseline_environment(baseline_frame)
                
#                 time.sleep(2)
#                 break
            
#             time.sleep(0.1)
        
#         if not setup_complete:
#             st.error("❌ Setup timeout - Please try again")
#             return {"error": "Setup phase failed or timeout"}
        
#         processor.end_setup_phase()
#         setup_ctx.video_processor = None  # Stop setup stream
        
#         # ========== INSTRUCTIONS ==========
#         st.success("✅ Setup complete!")
#         st.info(f"""
#         **📋 TEST INSTRUCTIONS:**
#         - You will answer **{len(questions_list)} questions** continuously
#         - Each question has **{duration_per_question} seconds** to answer
#         - **Important:** Even if a violation is detected, the interview will continue
#         - All violations will be reviewed at the end
#         - Stay within boundaries and maintain focus throughout
        
#         **The test will begin in 10 seconds...**
#         """)
#         time.sleep(10)
        
#         # ========== MAIN INTERVIEW LOOP ==========
#         all_results = []
#         session_violations = []
        
#         # Session video recording setup
#         session_video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
#         session_video_path = session_video_temp.name
#         session_video_temp.close()
        
#         fourcc = cv2.VideoWriter_fourcc(*"XVID")
#         video_writer = None
        
#         for q_idx, question_data in enumerate(questions_list):
#             question_text = question_data.get('question', 'No question text')
#             question_tip = question_data.get('tip', 'Speak clearly and confidently')
            
#             st.markdown("---")
#             st.markdown(f"### 📝 Question {q_idx + 1} of {len(questions_list)}")
#             st.markdown(f"**{question_text}**")
#             st.caption(f"💡 Tip: {question_tip}")
            
#             # Countdown
#             for i in range(3, 0, -1):
#                 st.info(f"⏱️ Starting in {i}s...")
#                 time.sleep(1)
            
#             # Start recording
#             processor.start_recording(q_idx + 1, len(questions_list), duration_per_question)
            
#             # WebRTC streamer for this question
#             question_ctx = webrtc_streamer(
#                 key=f"question_{q_idx}",
#                 video_processor_factory=lambda: processor,
#                 rtc_configuration=RTC_CONFIGURATION,
#                 media_stream_constraints={"video": True, "audio": True},
#                 async_processing=True,
#             )
            
#             # Audio recording using streamlit-audio-recorder
#             st.markdown("**🎤 Record your answer:**")
#             from audio_recorder_streamlit import audio_recorder
            
#             audio_bytes = audio_recorder(
#                 text="",
#                 recording_color="#e74c3c",
#                 neutral_color="#3498db",
#                 icon_name="microphone",
#                 icon_size="2x",
#                 key=f"audio_{q_idx}"
#             )
            
#             # Recording loop
#             question_start = time.time()
#             question_violations = []
            
#             status_placeholder = st.empty()
#             timer_placeholder = st.empty()
            
#             while (time.time() - question_start) < duration_per_question:
#                 elapsed = time.time() - question_start
#                 remaining = max(0, int(duration_per_question - elapsed))
                
#                 timer_placeholder.info(f"⏱️ Time remaining: **{remaining}s**")
                
#                 # Check for violations
#                 if question_ctx.video_processor and question_ctx.video_processor.violation_detected:
#                     violation_reason = question_ctx.video_processor.violation_reason
#                     violation_frame = question_ctx.video_processor.violation_frame
                    
#                     if violation_frame is not None:
#                         violation_img_path = self.save_violation_image(
#                             violation_frame, q_idx + 1, violation_reason
#                         )
                        
#                         question_violations.append({
#                             'reason': violation_reason,
#                             'timestamp': elapsed,
#                             'image_path': violation_img_path
#                         })
                    
#                     st.error(f"⚠️ Violation detected: {violation_reason}")
#                     break
                
#                 # Update status
#                 if question_ctx.video_processor:
#                     eye_contact = question_ctx.video_processor.eye_contact_frames
#                     total = max(question_ctx.video_processor.total_frames, 1)
#                     eye_pct = int((eye_contact / total) * 100)
                    
#                     status_placeholder.markdown(f"""
#                     **Status:**
#                     - 👁️ Eye Contact: {eye_pct}%
#                     - 😴 Blinks: {question_ctx.video_processor.blink_count}
#                     - 💡 Lighting: {question_ctx.video_processor.lighting_status}
#                     - ⚠️ Attention: {question_ctx.video_processor.attention_status}
#                     """)
                
#                 time.sleep(0.1)
            
#             # Stop recording
#             frames = processor.stop_recording()
            
#             # Initialize video writer if needed
#             if video_writer is None and len(frames) > 0:
#                 h, w = frames[0].shape[:2]
#                 video_writer = cv2.VideoWriter(session_video_path, fourcc, 15.0, (w, h))
            
#             # Write frames to video
#             for frame in frames:
#                 if video_writer:
#                     video_writer.write(frame)
            
#             # Transcribe audio
#             transcript = "[Could not understand audio]"
#             if audio_bytes:
#                 transcript = self.transcribe_audio_from_bytes(audio_bytes)
                
#                 # Save audio to temp file
#                 audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#                 audio_temp.write(audio_bytes)
#                 audio_path = audio_temp.name
#                 audio_temp.close()
#             else:
#                 audio_path = ""
            
#             # Store results
#             if question_violations:
#                 session_violations.extend([f"Q{q_idx+1}: {v['reason']}" for v in question_violations])
            
#             question_result = {
#                 'question_number': q_idx + 1,
#                 'question_text': question_text,
#                 'audio_path': audio_path,
#                 'frames': frames,
#                 'violations': question_violations,
#                 'violation_detected': len(question_violations) > 0,
#                 'eye_contact_pct': (processor.eye_contact_frames / max(processor.total_frames, 1)) * 100,
#                 'blink_count': processor.blink_count,
#                 'face_box': processor.face_box,
#                 'transcript': transcript,
#                 'lighting_status': processor.lighting_status
#             }
            
#             all_results.append(question_result)
            
#             # Show completion message
#             if question_violations:
#                 st.warning(f"⚠️ Violation detected in Q{q_idx + 1}! Continuing to next question in 3s...")
#                 time.sleep(3)
#             elif q_idx < len(questions_list) - 1:
#                 st.success(f"✅ Question {q_idx + 1} complete! Next question in 3s...")
#                 time.sleep(3)
        
#         # Cleanup
#         if video_writer:
#             video_writer.release()
        
#         # Final message
#         total_violations = sum(len(r.get('violations', [])) for r in all_results)
        
#         if total_violations > 0:
#             st.warning(f"⚠️ **TEST COMPLETED WITH {total_violations} VIOLATION(S)**")
#         else:
#             st.success("✅ **TEST COMPLETED SUCCESSFULLY!**")
        
#         return {
#             'questions_results': all_results,
#             'session_video_path': session_video_path,
#             'total_questions': len(questions_list),
#             'completed_questions': len(all_results),
#             'session_violations': session_violations,
#             'total_violations': total_violations,
#             'violation_images_dir': self.violation_images_dir,
#             'session_duration': duration_per_question * len(questions_list)
#         }


####


"""
WebRTC-Based Interview Recording and Violation Detection System
FULL BROWSER COMPATIBILITY VERSION
- Works in Streamlit Cloud (no server camera/mic needed)
- Captures video/audio from user's browser
- Maintains ALL original functionality
- Fixed cv2.FONT_HERSHEY_BOLD error (use FONT_HERSHEY_SIMPLEX)
"""

import cv2
import numpy as np
import threading
import time
import tempfile
import os
import speech_recognition as sr
import warnings
from collections import deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import queue

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# WebRTC Configuration - FIXED for local and cloud
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

class InterviewVideoProcessor(VideoTransformerBase):
    """
    Real-time video processor for interview recording
    Handles frame capture, violation detection, and display
    """
    
    def __init__(self, models_dict, frame_margin=50):
        self.models = models_dict
        self.frame_margin = frame_margin
        
        # Recording state
        self.is_recording = False
        self.frames_buffer = deque(maxlen=600)  # 20 seconds at 30fps
        self.current_frames = []
        
        # Violation detection state
        self.violation_detected = False
        self.violation_reason = ""
        self.violation_frame = None
        
        # Baseline environment
        self.baseline_environment = None
        self.baseline_set = False
        
        # Face tracking
        self.face_box = None
        self.no_face_start = None
        self.look_away_start = None
        self.blink_count = 0
        self.prev_blink = False
        self.eye_contact_frames = 0
        self.total_frames = 0
        
        # Status display
        self.status_text = "Initializing..."
        self.attention_status = "No Face"
        self.lighting_status = "Unknown"
        
        # Question info
        self.current_question_num = 0
        self.total_questions = 0
        self.recording_start_time = None
        self.question_duration = 20
        
        # Setup phase
        self.in_setup_phase = False
        self.setup_stable_frames = 0
        self.setup_required_frames = 30
        
        # Initialize pose detection
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_available = True
        except:
            self.pose_detector = None
            self.pose_available = False
    
    def transform(self, frame):
        """Process each video frame from browser"""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Always store frame if recording
        if self.is_recording:
            self.frames_buffer.append(img.copy())
            self.current_frames.append(img.copy())
        
        # Setup phase - just show boundaries and instructions
        if self.in_setup_phase:
            return self._process_setup_frame(img)
        
        # Recording phase - full violation detection
        if self.is_recording:
            img = self._process_recording_frame(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _process_setup_frame(self, frame):
        """Process frame during setup phase"""
        h, w = frame.shape[:2]
        
        # Draw boundaries
        frame_with_boundaries = self.draw_frame_boundaries(frame)
        
        # Analyze face position
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        is_ready = False
        status_color = (255, 165, 0)  # Orange
        
        if self.models['face_mesh'] is not None:
            face_results = self.models['face_mesh'].process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                num_faces = len(face_results.multi_face_landmarks)
                
                if num_faces > 1:
                    self.status_text = "⚠️ Multiple faces detected! Only ONE person allowed"
                    status_color = (0, 0, 255)
                    self.setup_stable_frames = 0
                
                elif num_faces == 1:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    
                    # Get face bounding box
                    landmarks_2d = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                    x_coords = landmarks_2d[:, 0]
                    y_coords = landmarks_2d[:, 1]
                    self.face_box = (int(np.min(x_coords)), int(np.min(y_coords)), 
                                   int(np.max(x_coords) - np.min(x_coords)), 
                                   int(np.max(y_coords) - np.min(y_coords)))
                    
                    # Check boundaries
                    within_bounds, boundary_msg, boundary_status = self.check_frame_boundaries(frame, self.face_box)
                    
                    # Check for others outside frame
                    outside_detected, obj_type, location = self.detect_person_outside_frame(frame)
                    
                    if outside_detected:
                        self.status_text = f"⚠️ {obj_type.upper()} detected outside frame ({location} side)!"
                        status_color = (0, 0, 255)
                        self.setup_stable_frames = 0
                    elif not within_bounds:
                        self.status_text = f"⚠️ {boundary_msg} - Please adjust!"
                        status_color = (0, 0, 255)
                        self.setup_stable_frames = 0
                        
                        # Highlight violated boundary
                        if boundary_status == "LEFT_VIOLATION":
                            cv2.rectangle(frame_with_boundaries, (0, 0), (self.frame_margin, h), (0, 0, 255), -1)
                        elif boundary_status == "RIGHT_VIOLATION":
                            cv2.rectangle(frame_with_boundaries, (w - self.frame_margin, 0), (w, h), (0, 0, 255), -1)
                        elif boundary_status == "TOP_VIOLATION":
                            cv2.rectangle(frame_with_boundaries, (0, 0), (w, self.frame_margin), (0, 0, 255), -1)
                    else:
                        # Good position!
                        self.setup_stable_frames += 1
                        progress = min(100, int((self.setup_stable_frames / self.setup_required_frames) * 100))
                        self.status_text = f"✅ Good position! Hold steady... {progress}%"
                        status_color = (0, 255, 0)
                        
                        if self.setup_stable_frames >= self.setup_required_frames:
                            is_ready = True
            else:
                self.status_text = "❌ No face detected - Please position yourself in frame"
                status_color = (0, 0, 255)
                self.setup_stable_frames = 0
        
        # Draw status overlay
        overlay_height = 140
        overlay = frame_with_boundaries.copy()
        cv2.rectangle(overlay, (0, h - overlay_height), (w, h), (0, 0, 0), -1)
        frame_with_boundaries = cv2.addWeighted(frame_with_boundaries, 0.7, overlay, 0.3, 0)
        
        cv2.putText(frame_with_boundaries, self.status_text, (10, h - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(frame_with_boundaries, "Instructions:", (10, h - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_with_boundaries, "• Keep your face within GREEN boundaries", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame_with_boundaries, "• Ensure no one else is visible", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame_with_boundaries, "• Remove all unauthorized items from view", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame_with_boundaries
    
    def _process_recording_frame(self, frame):
        """Process frame during recording with full violation detection"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.total_frames += 1
        
        # Analyze lighting
        self.lighting_status, brightness = self.analyze_lighting(frame)
        
        # Face detection and violation checks
        num_faces = 0
        looking_at_camera = False
        
        if self.models['face_mesh'] is not None:
            face_results = self.models['face_mesh'].process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                num_faces = len(face_results.multi_face_landmarks)
                
                # Check multiple bodies
                is_multi_body, multi_msg, body_count = self.detect_multiple_bodies(frame, num_faces)
                if is_multi_body:
                    self.violation_detected = True
                    self.violation_reason = multi_msg
                    self.violation_frame = frame.copy()
                    self.attention_status = "VIOLATION"
                    return self._add_violation_overlay(frame, multi_msg)
                
                # Multiple faces
                if num_faces > 1:
                    violation_msg = f"Multiple persons detected ({num_faces} faces)"
                    self.violation_detected = True
                    self.violation_reason = violation_msg
                    self.violation_frame = frame.copy()
                    self.attention_status = "VIOLATION"
                    return self._add_violation_overlay(frame, violation_msg)
                
                # Single face - full analysis
                elif num_faces == 1:
                    self.no_face_start = None
                    face_landmarks = face_results.multi_face_landmarks[0]
                    
                    # Get face box
                    landmarks_2d = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                    x_coords = landmarks_2d[:, 0]
                    y_coords = landmarks_2d[:, 1]
                    self.face_box = (int(np.min(x_coords)), int(np.min(y_coords)), 
                                   int(np.max(x_coords) - np.min(x_coords)), 
                                   int(np.max(y_coords) - np.min(y_coords)))
                    
                    # Check boundaries
                    within_bounds, boundary_msg, boundary_status = self.check_frame_boundaries(frame, self.face_box)
                    if not within_bounds:
                        self.violation_detected = True
                        self.violation_reason = boundary_msg
                        self.violation_frame = frame.copy()
                        self.attention_status = "VIOLATION"
                        return self._add_violation_overlay(frame, boundary_msg)
                    
                    # Check person outside frame
                    outside_detected, obj_type, location = self.detect_person_outside_frame(frame)
                    if outside_detected:
                        violation_msg = f"{obj_type.upper()} detected outside frame ({location} side)"
                        self.violation_detected = True
                        self.violation_reason = violation_msg
                        self.violation_frame = frame.copy()
                        self.attention_status = "VIOLATION"
                        return self._add_violation_overlay(frame, violation_msg)
                    
                    # Check intrusions
                    is_intrusion, intrusion_msg = self.detect_intrusion_at_edges(frame, self.face_box)
                    if is_intrusion:
                        self.violation_detected = True
                        self.violation_reason = intrusion_msg
                        self.violation_frame = frame.copy()
                        self.attention_status = "VIOLATION"
                        return self._add_violation_overlay(frame, intrusion_msg)
                    
                    # Check hands outside
                    is_hand_violation, hand_msg = self.detect_hands_outside_main_person(frame, self.face_box)
                    if is_hand_violation:
                        self.violation_detected = True
                        self.violation_reason = hand_msg
                        self.violation_frame = frame.copy()
                        self.attention_status = "VIOLATION"
                        return self._add_violation_overlay(frame, hand_msg)
                    
                    # Check suspicious movements
                    is_suspicious, sus_msg = self.detect_suspicious_movements(frame)
                    if is_suspicious:
                        self.violation_detected = True
                        self.violation_reason = sus_msg
                        self.violation_frame = frame.copy()
                        self.attention_status = "VIOLATION"
                        return self._add_violation_overlay(frame, sus_msg)
                    
                    # Head pose and gaze
                    yaw, pitch, roll = self.estimate_head_pose(face_landmarks, frame.shape)
                    gaze_centered = self.calculate_eye_gaze(face_landmarks, frame.shape)
                    
                    # Blink detection
                    is_blink = self.detect_blink(face_landmarks)
                    if is_blink and not self.prev_blink:
                        self.blink_count += 1
                    self.prev_blink = is_blink
                    
                    # Check attention
                    head_looking_forward = abs(yaw) <= 20 and abs(pitch) <= 20
                    
                    if head_looking_forward and gaze_centered:
                        self.look_away_start = None
                        looking_at_camera = True
                        self.eye_contact_frames += 1
                        self.attention_status = "Looking at Camera ✓"
                    else:
                        if self.look_away_start is None:
                            self.look_away_start = time.time()
                            self.attention_status = "Looking Away"
                        else:
                            elapsed = time.time() - self.look_away_start
                            if elapsed > 2.0:
                                violation_msg = "Looking away for >2 seconds"
                                self.violation_detected = True
                                self.violation_reason = violation_msg
                                self.violation_frame = frame.copy()
                                self.attention_status = "VIOLATION"
                                return self._add_violation_overlay(frame, violation_msg)
                            else:
                                self.attention_status = f"Looking Away ({elapsed:.1f}s)"
            else:
                # No face detected
                if self.no_face_start is None:
                    self.no_face_start = time.time()
                    self.attention_status = "No Face Visible"
                else:
                    elapsed = time.time() - self.no_face_start
                    if elapsed > 2.0:
                        violation_msg = "No face visible for >2 seconds"
                        self.violation_detected = True
                        self.violation_reason = violation_msg
                        self.violation_frame = frame.copy()
                        self.attention_status = "VIOLATION"
                        return self._add_violation_overlay(frame, violation_msg)
                    else:
                        self.attention_status = f"No Face ({elapsed:.1f}s)"
        
        # Check for new objects (every 20 frames)
        if self.total_frames % 20 == 0 and self.baseline_set:
            new_detected, new_items = self.detect_new_objects(frame)
            if new_detected:
                violation_msg = f"New item(s) brought into view: {', '.join(new_items)}"
                self.violation_detected = True
                self.violation_reason = violation_msg
                self.violation_frame = frame.copy()
                self.attention_status = "VIOLATION"
                return self._add_violation_overlay(frame, violation_msg)
        
        # Add status overlay (no violation)
        return self._add_status_overlay(frame)
    
    def _add_status_overlay(self, frame):
        """Add status information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent black bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Status color
        status_color = (0, 255, 0) if not self.violation_detected else (0, 165, 255)
        
        # Question info
        if self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, int(self.question_duration - elapsed))
            time_text = f"Time: {remaining}s"
        else:
            time_text = "Ready"
        
        cv2.putText(frame, f"Q{self.current_question_num}/{self.total_questions} - {self.attention_status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Lighting: {self.lighting_status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        eye_contact_pct = int((self.eye_contact_frames / max(self.total_frames, 1)) * 100)
        cv2.putText(frame, f"Eye Contact: {eye_contact_pct}%", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, time_text, 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _add_violation_overlay(self, frame, violation_msg):
        """Add red violation overlay to frame"""
        h, w = frame.shape[:2]
        
        # Red tinted overlay
        red_overlay = frame.copy()
        cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 255), -1)
        frame = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
        
        # Red border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
        
        # Violation text
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, "VIOLATION DETECTED", (w//2 - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Violation reason at bottom
        cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 0), -1)
        
        # Split long text
        words = violation_msg.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) > 50:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        y_offset = h - 90
        for line in lines[:2]:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return frame
    
    def start_recording(self, question_num, total_questions, duration):
        """Start recording for a question"""
        self.is_recording = True
        self.current_question_num = question_num
        self.total_questions = total_questions
        self.question_duration = duration
        self.recording_start_time = time.time()
        self.current_frames = []
        self.frames_buffer.clear()
        self.violation_detected = False
        self.violation_reason = ""
        self.violation_frame = None
        self.total_frames = 0
        self.eye_contact_frames = 0
        self.blink_count = 0
        self.no_face_start = None
        self.look_away_start = None
    
    def stop_recording(self):
        """Stop recording and return collected frames"""
        self.is_recording = False
        frames = list(self.current_frames)
        self.current_frames = []
        return frames
    
    def set_baseline_environment(self, frame):
        """Set baseline environment from frame"""
        self.baseline_environment = self.scan_environment(frame)
        self.baseline_set = True
    
    def start_setup_phase(self):
        """Start the setup phase"""
        self.in_setup_phase = True
        self.setup_stable_frames = 0
    
    def is_setup_complete(self):
        """Check if setup is complete"""
        return self.setup_stable_frames >= self.setup_required_frames
    
    def end_setup_phase(self):
        """End setup phase"""
        self.in_setup_phase = False
    
    # ==================== HELPER METHODS (from original Recording_system.py) ====================
    
    def draw_frame_boundaries(self, frame):
        """Draw visible frame boundaries"""
        h, w = frame.shape[:2]
        margin = self.frame_margin
        
        overlay = frame.copy()
        
        cv2.line(overlay, (margin, 0), (margin, h), (0, 255, 0), 3)
        cv2.line(overlay, (w - margin, 0), (w - margin, h), (0, 255, 0), 3)
        cv2.line(overlay, (0, margin), (w, margin), (0, 255, 0), 3)
        cv2.rectangle(overlay, (margin, margin), (w - margin, h), (0, 255, 0), 2)
        
        frame_with_boundaries = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        corner_size = 30
        cv2.line(frame_with_boundaries, (margin, margin), (margin + corner_size, margin), (0, 255, 0), 3)
        cv2.line(frame_with_boundaries, (margin, margin), (margin, margin + corner_size), (0, 255, 0), 3)
        
        cv2.line(frame_with_boundaries, (w - margin, margin), (w - margin - corner_size, margin), (0, 255, 0), 3)
        cv2.line(frame_with_boundaries, (w - margin, margin), (w - margin, margin + corner_size), (0, 255, 0), 3)
        
        cv2.putText(frame_with_boundaries, "Stay within GREEN boundaries", 
                    (w//2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_with_boundaries
    
    def check_frame_boundaries(self, frame, face_box):
        """Check if person is within frame boundaries"""
        if face_box is None:
            return False, "No face detected", "NO_FACE"
        
        h, w = frame.shape[:2]
        margin = self.frame_margin
        x, y, fw, fh = face_box
        
        face_left = x
        face_right = x + fw
        face_top = y
        
        if face_left < margin:
            return False, "Person too close to LEFT edge", "LEFT_VIOLATION"
        if face_right > (w - margin):
            return False, "Person too close to RIGHT edge", "RIGHT_VIOLATION"
        if face_top < margin:
            return False, "Person too close to TOP edge", "TOP_VIOLATION"
        
        return True, "Within boundaries", "OK"
    
    def analyze_lighting(self, frame):
        """Analyze lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if mean_brightness < 60:
            return "Too Dark", mean_brightness
        elif mean_brightness > 200:
            return "Too Bright", mean_brightness
        elif std_brightness < 25:
            return "Low Contrast", mean_brightness
        else:
            return "Good", mean_brightness
    
    def detect_blink(self, face_landmarks):
        """Detect if eye is blinking"""
        upper_lid = face_landmarks.landmark[159]
        lower_lid = face_landmarks.landmark[145]
        eye_openness = abs(upper_lid.y - lower_lid.y)
        return eye_openness < 0.01
    
    def calculate_eye_gaze(self, face_landmarks, frame_shape):
        """Calculate if eyes are looking at camera"""
        left_eye_indices = [468, 469, 470, 471, 472]
        right_eye_indices = [473, 474, 475, 476, 477]
        left_eye_center = [33, 133, 157, 158, 159, 160, 161, 163, 144, 145, 153, 154, 155]
        right_eye_center = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 373, 374, 390]
        
        landmarks = face_landmarks.landmark
        
        left_iris_x = np.mean([landmarks[i].x for i in left_eye_indices if i < len(landmarks)])
        left_eye_x = np.mean([landmarks[i].x for i in left_eye_center if i < len(landmarks)])
        
        right_iris_x = np.mean([landmarks[i].x for i in right_eye_indices if i < len(landmarks)])
        right_eye_x = np.mean([landmarks[i].x for i in right_eye_center if i < len(landmarks)])
        
        left_gaze_ratio = (left_iris_x - left_eye_x) if left_iris_x and left_eye_x else 0
        right_gaze_ratio = (right_iris_x - right_eye_x) if right_iris_x and right_eye_x else 0
        
        avg_gaze = (left_gaze_ratio + right_gaze_ratio) / 2
        
        return abs(avg_gaze) < 0.02
    
    def estimate_head_pose(self, face_landmarks, frame_shape):
        """Estimate head pose angles"""
        h, w = frame_shape[:2]
        landmarks_3d = np.array([(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark])
        
        required_indices = [1, 33, 263, 61, 291]
        image_points = np.array([landmarks_3d[i] for i in required_indices], dtype="double")
        
        model_points = np.array([
            (0.0, 0.0, 0.0), (-30.0, -125.0, -30.0),
            (30.0, -125.0, -30.0), (-60.0, -70.0, -60.0),
            (60.0, -70.0, -60.0)
        ])
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, 
            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rmat, rotation_vector))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
            yaw, pitch, roll = [float(a) for a in euler]
            return yaw, pitch, roll
        
        return 0, 0, 0
    
    def scan_environment(self, frame):
        """Scan and catalog the environment"""
        if self.models['yolo'] is None:
            return {'objects': [], 'positions': []}
        
        try:
            results = self.models['yolo'].predict(frame, conf=0.25, verbose=False)
            
            environment_data = {
                'objects': [],
                'positions': [],
                'person_position': None
            }
            
            if results and len(results) > 0:
                names = self.models['yolo'].names
                boxes = results[0].boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    obj_name = names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    environment_data['objects'].append(obj_name)
                    environment_data['positions'].append({
                        'name': obj_name,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1+x2)/2), int((y1+y2)/2))
                    })
                    
                    if obj_name == 'person':
                        environment_data['person_position'] = (int((x1+x2)/2), int((y1+y2)/2))
            
            return environment_data
            
        except Exception as e:
            return {'objects': [], 'positions': []}
    
    def detect_new_objects(self, frame):
        """Detect NEW objects that weren't in baseline environment"""
        if self.models['yolo'] is None or self.baseline_environment is None:
            return False, []
        
        try:
            results = self.models['yolo'].predict(frame, conf=0.25, verbose=False)
            
            if results and len(results) > 0:
                names = self.models['yolo'].names
                boxes = results[0].boxes
                
                current_objects = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    obj_name = names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    current_center = (int((x1+x2)/2), int((y1+y2)/2))
                    
                    current_objects.append({
                        'name': obj_name,
                        'center': current_center,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
                
                baseline_objects = self.baseline_environment['positions']
                new_items = []
                
                for curr_obj in current_objects:
                    if curr_obj['name'] == 'person':
                        continue
                    
                    is_baseline = False
                    for base_obj in baseline_objects:
                        if curr_obj['name'] == base_obj['name']:
                            dist = np.sqrt(
                                (curr_obj['center'][0] - base_obj['center'][0])**2 +
                                (curr_obj['center'][1] - base_obj['center'][1])**2
                            )
                            if dist < 100:
                                is_baseline = True
                                break
                    
                    if not is_baseline:
                        new_items.append(curr_obj['name'])
                
                if new_items:
                    return True, list(set(new_items))
            
            return False, []
            
        except Exception as e:
            return False, []
    
    def detect_person_outside_frame(self, frame):
        """Detect if any person/living being is outside boundaries"""
        if self.models['yolo'] is None:
            return False, "", ""
        
        h, w = frame.shape[:2]
        margin = self.frame_margin
        
        try:
            results = self.models['yolo'].predict(frame, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                names = self.models['yolo'].names
                boxes = results[0].boxes
                
                living_beings = ['person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 
                                'elephant', 'bear', 'zebra', 'giraffe']
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    obj_name = names[cls_id]
                    
                    if obj_name.lower() in living_beings:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if x1 < margin or x2 < margin:
                            return True, obj_name, "LEFT"
                        if x1 > (w - margin) or x2 > (w - margin):
                            return True, obj_name, "RIGHT"
                        if y1 < margin or y2 < margin:
                            return True, obj_name, "TOP"
        
        except Exception as e:
            pass
        
        return False, "", ""
    
    def detect_multiple_bodies(self, frame, num_faces):
        """Detect multiple bodies using pose and hand detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        body_count = 0
        detected_parts = []
        
        if self.pose_available and self.pose_detector:
            try:
                pose_results = self.pose_detector.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    body_count += 1
                    detected_parts.append("body")
                    
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    visible_shoulders = sum(1 for idx in [11, 12] 
                                          if landmarks[idx].visibility > 0.5)
                    visible_elbows = sum(1 for idx in [13, 14] 
                                        if landmarks[idx].visibility > 0.5)
                    
                    if visible_shoulders > 2 or visible_elbows > 2:
                        return True, "Multiple body parts detected (extra shoulders/arms)", body_count + 1
                        
            except Exception as e:
                pass
        
        if self.models['hands'] is not None:
            try:
                hand_results = self.models['hands'].process(rgb_frame)
                
                if hand_results.multi_hand_landmarks:
                    num_hands = len(hand_results.multi_hand_landmarks)
                    
                    if num_hands > 2:
                        detected_parts.append(f"{num_hands} hands")
                        return True, f"Multiple persons detected ({num_hands} hands visible)", 2
                    
                    if num_hands == 2:
                        hand1 = hand_results.multi_hand_landmarks[0].landmark[0]
                        hand2 = hand_results.multi_hand_landmarks[1].landmark[0]
                        
                        distance = np.sqrt((hand1.x - hand2.x)**2 + (hand1.y - hand2.y)**2)
                        
                        if distance > 0.7:
                            detected_parts.append("widely separated hands")
                            return True, "Suspicious hand positions (possible multiple persons)", 2
                            
            except Exception as e:
                pass
        
        if num_faces == 1 and body_count > 1:
            return True, "Body parts from multiple persons detected", 2
        
        if num_faces > 1:
            return True, f"Multiple persons detected ({num_faces} faces)", num_faces
        
        return False, "", max(num_faces, body_count)
    
    def detect_hands_outside_main_person(self, frame, face_box):
        """Detect hands outside main person's area"""
        if self.models['hands'] is None or face_box is None:
            return False, ""
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        try:
            hand_results = self.models['hands'].process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                x, y, fw, fh = face_box
                
                expected_left = max(0, x - fw)
                expected_right = min(w, x + fw * 2)
                expected_top = max(0, y - fh)
                expected_bottom = min(h, y + fh * 4)
                
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_x = hand_landmarks.landmark[0].x * w
                    hand_y = hand_landmarks.landmark[0].y * h
                    
                    if (hand_x < expected_left - 50 or hand_x > expected_right + 50 or
                        hand_y < expected_top - 50 or hand_y > expected_bottom + 50):
                        return True, "Hand detected outside main person's area"
                
        except Exception as e:
            pass
        
        return False, ""
    
    def detect_suspicious_movements(self, frame):
        """Detect suspicious hand movements"""
        if self.models['hands'] is None:
            return False, ""
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        try:
            hand_results = self.models['hands'].process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    
                    wrist_y = wrist.y * h
                    tip_y = index_tip.y * h
                    
                    if wrist_y > h * 0.75:
                        return True, "Hand movement below desk level detected"
                    
                    if wrist_y < h * 0.15:
                        return True, "Suspicious hand movement at top of frame"
        
        except Exception as e:
            pass
        
        return False, ""
    
    def has_skin_tone(self, region):
        """Check if region contains skin-like colors"""
        if region.size == 0:
            return False
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([0, 20, 0], dtype=np.uint8)
        upper_skin2 = np.array([20, 150, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        skin_ratio = np.sum(mask > 0) / mask.size
        return skin_ratio > 0.3
    
    def detect_intrusion_at_edges(self, frame, face_box):
        """Detect body parts intruding from frame edges"""
        if face_box is None:
            return False, ""
        
        h, w = frame.shape[:2]
        x, y, fw, fh = face_box
        
        edge_width = 80
        
        left_region = frame[:, :edge_width]
        right_region = frame[:, w-edge_width:]
        top_left = frame[:edge_width, :w//3]
        top_right = frame[:edge_width, 2*w//3:]
        
        face_center_x = x + fw // 2
        face_far_from_left = face_center_x > w * 0.3
        face_far_from_right = face_center_x < w * 0.7
        
        if face_far_from_left and self.has_skin_tone(left_region):
            if self.models['hands']:
                rgb_region = cv2.cvtColor(left_region, cv2.COLOR_BGR2RGB)
                try:
                    result = self.models['hands'].process(rgb_region)
                    if result.multi_hand_landmarks:
                        return True, "Body part detected at left edge (another person)"
                except:
                    pass
        
        if face_far_from_right and self.has_skin_tone(right_region):
            if self.models['hands']:
                rgb_region = cv2.cvtColor(right_region, cv2.COLOR_BGR2RGB)
                try:
                    result = self.models['hands'].process(rgb_region)
                    if result.multi_hand_landmarks:
                        return True, "Body part detected at right edge (another person)"
                except:
                    pass
        
        if y > h * 0.2:
            if self.has_skin_tone(top_left) or self.has_skin_tone(top_right):
                return True, "Body part detected at top edge (another person)"
        
        return False, ""


class RecordingSystemWebRTC:
    """WebRTC-based Recording System - Browser Compatible"""
    
    def __init__(self, models_dict):
        self.models = models_dict
        self.violation_images_dir = tempfile.mkdtemp(prefix="violations_")
    
    def save_violation_image(self, frame, question_number, violation_reason):
        """Save violation image with overlay"""
        try:
            timestamp = int(time.time() * 1000)
            filename = f"violation_q{question_number}_{timestamp}.jpg"
            filepath = os.path.join(self.violation_images_dir, filename)
            
            overlay_frame = frame.copy()
            h, w = overlay_frame.shape[:2]
            
            # Red overlay
            red_overlay = overlay_frame.copy()
            cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 255), -1)
            overlay_frame = cv2.addWeighted(overlay_frame, 0.7, red_overlay, 0.3, 0)
            
            # Red border
            cv2.rectangle(overlay_frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            
            # Title
            text = "VIOLATION DETECTED"
            cv2.rectangle(overlay_frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(overlay_frame, text, (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Reason
            cv2.rectangle(overlay_frame, (0, h-100), (w, h), (0, 0, 0), -1)
            words = violation_reason.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) > 50:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            y_offset = h - 90
            for line in lines[:2]:
                cv2.putText(overlay_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            cv2.imwrite(filepath, overlay_frame)
            return filepath
            
        except Exception as e:
            print(f"Error saving violation image: {e}")
            return None
    
    def transcribe_audio_from_bytes(self, audio_bytes):
        """Transcribe audio from bytes"""
        r = sr.Recognizer()
        try:
            # Save bytes to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            with sr.AudioFile(temp_path) as source:
                audio = r.record(source)
            
            text = r.recognize_google(audio)
            os.unlink(temp_path)
            return text if text.strip() else "[Could not understand audio]"
        except sr.UnknownValueError:
            return "[Could not understand audio]"
        except sr.RequestError:
            return "[Speech recognition service unavailable]"
        except Exception as e:
            return "[Could not understand audio]"
    
    def record_continuous_interview_webrtc(self, questions_list, duration_per_question, ui_callbacks):
        """
        WebRTC-based continuous interview recording
        Uses Streamlit components for browser video/audio capture
        """
        
        st.markdown("---")
        st.subheader("🎬 Interview Session")
        
        # Create containers for dynamic content
        video_container = st.container()
        status_container = st.container()
        control_container = st.container()
        
        with video_container:
            st.markdown("### 📹 Camera View")
            video_placeholder = st.empty()
        
        with status_container:
            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1:
                status_text = st.empty()
            with status_col2:
                timer_text = st.empty()
            with status_col3:
                progress_text = st.empty()
        
        # Initialize video processor
        processor = InterviewVideoProcessor(self.models)
        
        # ========== SETUP PHASE ==========
        st.info("🔧 **Step 1: Position Setup** - Adjust your position within the GREEN boundaries")
        
        processor.start_setup_phase()
        
        # WebRTC streamer for setup
        setup_ctx = webrtc_streamer(
            key="setup_phase",
            video_processor_factory=lambda: processor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Wait for setup completion
        with control_container:
            setup_status = st.empty()
            setup_status.warning("⏳ Adjusting position... Please stay within boundaries")
        
        setup_complete = False
        setup_timeout = 90
        setup_start = time.time()
        
        while not setup_complete and (time.time() - setup_start) < setup_timeout:
            if setup_ctx.video_processor and setup_ctx.video_processor.is_setup_complete():
                setup_complete = True
                setup_status.success("✅ Setup complete! Scanning environment...")
                
                # Get a frame for baseline
                if len(processor.frames_buffer) > 0:
                    baseline_frame = list(processor.frames_buffer)[-1]
                    processor.set_baseline_environment(baseline_frame)
                
                time.sleep(2)
                break
            
            time.sleep(0.1)
        
        if not setup_complete:
            st.error("❌ Setup timeout - Please try again")
            return {"error": "Setup phase failed or timeout"}
        
        processor.end_setup_phase()
        setup_ctx.video_processor = None  # Stop setup stream
        
        # ========== INSTRUCTIONS ==========
        st.success("✅ Setup complete!")
        st.info(f"""
        **📋 TEST INSTRUCTIONS:**
        - You will answer **{len(questions_list)} questions** continuously
        - Each question has **{duration_per_question} seconds** to answer
        - **Important:** Even if a violation is detected, the interview will continue
        - All violations will be reviewed at the end
        - Stay within boundaries and maintain focus throughout
        
        **The test will begin in 10 seconds...**
        """)
        time.sleep(10)
        
        # ========== MAIN INTERVIEW LOOP ==========
        all_results = []
        session_violations = []
        
        # Session video recording setup
        session_video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        session_video_path = session_video_temp.name
        session_video_temp.close()
        
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = None
        
        for q_idx, question_data in enumerate(questions_list):
            question_text = question_data.get('question', 'No question text')
            question_tip = question_data.get('tip', 'Speak clearly and confidently')
            
            st.markdown("---")
            st.markdown(f"### 📝 Question {q_idx + 1} of {len(questions_list)}")
            st.markdown(f"**{question_text}**")
            st.caption(f"💡 Tip: {question_tip}")
            
            # Countdown
            for i in range(3, 0, -1):
                st.info(f"⏱️ Starting in {i}s...")
                time.sleep(1)
            
            # Start recording
            processor.start_recording(q_idx + 1, len(questions_list), duration_per_question)
            
            # WebRTC streamer for this question
            question_ctx = webrtc_streamer(
                key=f"question_{q_idx}",
                video_processor_factory=lambda: processor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": True},
                async_processing=True,
            )
            
            # Audio recording using streamlit-audio-recorder
            st.markdown("**🎤 Record your answer:**")
            from audio_recorder_streamlit import audio_recorder
            
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="2x",
                key=f"audio_{q_idx}"
            )
            
            # Recording loop
            question_start = time.time()
            question_violations = []
            
            status_placeholder = st.empty()
            timer_placeholder = st.empty()
            
            while (time.time() - question_start) < duration_per_question:
                elapsed = time.time() - question_start
                remaining = max(0, int(duration_per_question - elapsed))
                
                timer_placeholder.info(f"⏱️ Time remaining: **{remaining}s**")
                
                # Check for violations
                if question_ctx.video_processor and question_ctx.video_processor.violation_detected:
                    violation_reason = question_ctx.video_processor.violation_reason
                    violation_frame = question_ctx.video_processor.violation_frame
                    
                    if violation_frame is not None:
                        violation_img_path = self.save_violation_image(
                            violation_frame, q_idx + 1, violation_reason
                        )
                        
                        question_violations.append({
                            'reason': violation_reason,
                            'timestamp': elapsed,
                            'image_path': violation_img_path
                        })
                    
                    st.error(f"⚠️ Violation detected: {violation_reason}")
                    break
                
                # Update status
                if question_ctx.video_processor:
                    eye_contact = question_ctx.video_processor.eye_contact_frames
                    total = max(question_ctx.video_processor.total_frames, 1)
                    eye_pct = int((eye_contact / total) * 100)
                    
                    status_placeholder.markdown(f"""
                    **Status:**
                    - 👁️ Eye Contact: {eye_pct}%
                    - 😴 Blinks: {question_ctx.video_processor.blink_count}
                    - 💡 Lighting: {question_ctx.video_processor.lighting_status}
                    - ⚠️ Attention: {question_ctx.video_processor.attention_status}
                    """)
                
                time.sleep(0.1)
            
            # Stop recording
            frames = processor.stop_recording()
            
            # Initialize video writer if needed
            if video_writer is None and len(frames) > 0:
                h, w = frames[0].shape[:2]
                video_writer = cv2.VideoWriter(session_video_path, fourcc, 15.0, (w, h))
            
            # Write frames to video
            for frame in frames:
                if video_writer:
                    video_writer.write(frame)
            
            # Transcribe audio
            transcript = "[Could not understand audio]"
            if audio_bytes:
                transcript = self.transcribe_audio_from_bytes(audio_bytes)
                
                # Save audio to temp file
                audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_temp.write(audio_bytes)
                audio_path = audio_temp.name
                audio_temp.close()
            else:
                audio_path = ""
            
            # Store results
            if question_violations:
                session_violations.extend([f"Q{q_idx+1}: {v['reason']}" for v in question_violations])
            
            question_result = {
                'question_number': q_idx + 1,
                'question_text': question_text,
                'audio_path': audio_path,
                'frames': frames,
                'violations': question_violations,
                'violation_detected': len(question_violations) > 0,
                'eye_contact_pct': (processor.eye_contact_frames / max(processor.total_frames, 1)) * 100,
                'blink_count': processor.blink_count,
                'face_box': processor.face_box,
                'transcript': transcript,
                'lighting_status': processor.lighting_status
            }
            
            all_results.append(question_result)
            
            # Show completion message
            if question_violations:
                st.warning(f"⚠️ Violation detected in Q{q_idx + 1}! Continuing to next question in 3s...")
                time.sleep(3)
            elif q_idx < len(questions_list) - 1:
                st.success(f"✅ Question {q_idx + 1} complete! Next question in 3s...")
                time.sleep(3)
        
        # Cleanup
        if video_writer:
            video_writer.release()
        
        # Final message
        total_violations = sum(len(r.get('violations', [])) for r in all_results)
        
        if total_violations > 0:
            st.warning(f"⚠️ **TEST COMPLETED WITH {total_violations} VIOLATION(S)**")
        else:
            st.success("✅ **TEST COMPLETED SUCCESSFULLY!**")
        
        return {
            'questions_results': all_results,
            'session_video_path': session_video_path,
            'total_questions': len(questions_list),
            'completed_questions': len(all_results),
            'session_violations': session_violations,
            'total_violations': total_violations,
            'violation_images_dir': self.violation_images_dir,
            'session_duration': duration_per_question * len(questions_list)
        }


####