"""
Main Integration File - AI Interview System
WEBRTC VERSION - Works in Streamlit Cloud
"""

import streamlit as st
import warnings
import os
from PIL import Image, ImageDraw

# Import the WebRTC recording system
from Recording_system_webrtc import RecordingSystemWebRTC
from analysis_system import AnalysisSystem
from scoring_dashboard import ScoringDashboard

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Try importing optional modules
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
except:
    MP_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    DEEPFACE_AVAILABLE = False

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Interview Assessment Platform", layout="wide", page_icon="🎯")

# ==================== SIMPLE, CLEAN STYLES ====================
st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Simple body styling */
body { 
    background-color: #ffffff; 
    color: #333333; 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
}

/* Simple headers */
h1 { 
    color: #2c3e50; 
    font-weight: 600;
    margin-bottom: 0.5rem;
}

h2 { 
    color: #34495e; 
    font-weight: 500;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

h3 { 
    color: #555555; 
    font-weight: 500;
}

/* Simple boxes */
.info-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-left: 4px solid #28a745;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-left: 4px solid #ffc107;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-left: 4px solid #dc3545;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Simple question box */
.question-box {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    min-height: 200px;
}

.question-box h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #e9ecef;
}

/* Simple metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.metric-card h3 {
    color: #2c3e50;
    font-size: 1.5rem;
    margin: 0;
}

.metric-card p {
    color: #6c757d;
    font-size: 0.875rem;
    margin: 0.25rem 0 0 0;
}

/* Hide sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* Simple buttons */
.stButton > button {
    border-radius: 4px;
    border: 1px solid #dee2e6;
}

/* Simple progress bar */
.stProgress > div > div {
    background-color: #007bff;
}
</style>
""", unsafe_allow_html=True)

# ==================== QUESTIONS CONFIGURATION ====================
QUESTIONS = [
    {
        "question": "Tell me about yourself.",
        "type": "personal",
        "ideal_answer": "I'm a computer science postgraduate with a strong interest in AI and software development. I've worked on several projects involving Python, machine learning, and data analysis, which helped me improve both my technical and problem-solving skills. I enjoy learning new technologies and applying them to create practical solutions. Outside of academics, I like collaborating on team projects and continuously developing my professional skills.",
        "tip": "Focus on your background, skills, and personality"
    },
    {
        "question": "What are your strengths and weaknesses?",
        "type": "personal",
        "ideal_answer": "One of my key strengths is that I'm very detail-oriented and persistent – I make sure my work is accurate and well-tested. I also enjoy solving complex problems and learning new tools quickly. As for weaknesses, I used to spend too much time perfecting small details, which sometimes slowed me down. But I've been improving by prioritizing tasks better and focusing on overall impact.",
        "tip": "Be honest and show self-awareness"
    },
    {
        "question": "Where do you see yourself in the next 5 years?",
        "type": "personal",
        "ideal_answer": "In the next five years, I see myself growing into a more responsible and skilled professional, ideally in a role where I can contribute to meaningful projects involving AI and software development. I'd also like to take on leadership responsibilities and guide new team members as I gain experience.",
        "tip": "Show ambition aligned with career growth"
    }
]

# ==================== GENERATE DEMO IMAGES ====================
def create_frame_demo_image(is_correct=True):
    """Create demonstration image showing correct/incorrect positioning"""
    width, height = 500, 350
    img = Image.new('RGB', (width, height), color='#f8f9fa')
    draw = ImageDraw.Draw(img)
    
    margin = 40
    boundary_color = '#28a745' if is_correct else '#dc3545'
    
    # Draw boundaries
    draw.rectangle([margin, margin, width-margin, height-margin], outline=boundary_color, width=3)
    
    if is_correct:
        # Draw person inside
        head_x, head_y = width // 2, margin + 60
        draw.ellipse([head_x - 30, head_y - 30, head_x + 30, head_y + 30], fill='#ffc107', outline='#333333', width=2)
        
        body_y = head_y + 40
        draw.rectangle([head_x - 40, body_y, head_x + 40, body_y + 80], fill='#007bff', outline='#333333', width=2)
        
        draw.text((width//2 - 80, height - 30), "✓ Correct Position", fill='#28a745')
    else:
        # Draw person outside
        head_x, head_y = margin - 20, margin + 60
        draw.ellipse([head_x - 30, head_y - 30, head_x + 30, head_y + 30], fill='#ffc107', outline='#333333', width=2)
        
        draw.text((width//2 - 80, height - 30), "✗ Outside Bounds", fill='#dc3545')
    
    return img

# ==================== HOME PAGE ====================
def show_home_page():
    """Display clean home page"""
    
    st.title("Interview Assessment Platform")
    st.write("Professional evaluation system for video interviews")
    
    st.markdown("---")
    
    # Simple features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📋 Structured Assessment**
        
        Standardized evaluation with consistent criteria
        """)
    
    with col2:
        st.markdown("""
        **📊 Detailed Analytics**
        
        Comprehensive metrics and performance insights
        """)
    
    with col3:
        st.markdown("""
        **✅ Compliance Monitoring**
        
        Real-time monitoring ensures integrity
        """)
    
    st.markdown("---")
    
    # Introduction
    st.subheader("Before You Begin")
    st.write("""
    This platform evaluates candidates through structured video interviews. Please review 
    the camera positioning requirements below to ensure a smooth assessment.
    """)
    
    # Frame positioning
    st.subheader("Camera Positioning Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Correct Positioning**")
        correct_img = create_frame_demo_image(is_correct=True)
        st.image(correct_img, use_container_width=True)
        st.markdown("""
        - Center yourself in the frame
        - Keep entire face visible
        - Remain alone in the frame
        - Ensure adequate lighting
        - Maintain forward gaze
        """)
    
    with col2:
        st.markdown("**❌ Common Mistakes**")
        incorrect_img = create_frame_demo_image(is_correct=False)
        st.image(incorrect_img, use_container_width=True)
        st.markdown("""
        - Moving outside boundaries
        - Multiple people visible
        - Obstructed or partial view
        - Poor lighting conditions
        - Extended periods looking away
        """)
    
    st.markdown("---")
    
    # Assessment process
    st.subheader("Assessment Process")
    st.markdown(f"""
    1. **Initial Setup (60 seconds):** Position yourself within marked boundaries
    2. **Environment Scan:** System records baseline to detect changes
    3. **Interview Session:** Respond to {len(QUESTIONS)} questions (20 seconds each)
    4. **Continuous Monitoring:** System monitors compliance throughout
    5. **Results Analysis:** Receive comprehensive evaluation with feedback
    """)
    
    st.markdown("---")
    
    # Technical requirements
    st.subheader("Technical Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Hardware**
        - Functional webcam (720p recommended)
        - Clear microphone
        - Stable internet (5 Mbps minimum)
        - Desktop or laptop computer
        """)
    
    with col2:
        st.markdown("""
        **Environment**
        - Quiet, private space
        - Front-facing lighting
        - Neutral background
        - Comfortable seating
        """)
    
    st.markdown("---")
    
    # Browser permissions
    st.warning("""
    **🔒 Browser Permissions Required:**
    - Camera access
    - Microphone access
    
    You will be prompted to grant these permissions when the interview starts.
    """)
    
    st.markdown("---")
    
    # Confirmation
    st.subheader("Ready to Begin")
    
    if 'guidelines_accepted' not in st.session_state:
        st.session_state.guidelines_accepted = False
    
    st.session_state.guidelines_accepted = st.checkbox(
        f"I confirm that I have reviewed all guidelines and am prepared to complete {len(QUESTIONS)} interview questions.",
        value=st.session_state.guidelines_accepted,
        key="guidelines_checkbox"
    )
    
    if st.session_state.guidelines_accepted:
        st.success("✅ You are ready to proceed with the assessment.")
        if st.button("Begin Assessment", type="primary"):
            st.session_state.page = "interview"
            st.session_state.interview_started = False
            st.rerun()
    else:
        st.info("ℹ️ Please confirm that you have reviewed the guidelines to continue.")

# ==================== LOAD MODELS ====================
@st.cache_resource(show_spinner="Initializing assessment system...")
def load_all_models():
    """Load all AI models and return dictionary"""
    models = {}
    
    if DEEPFACE_AVAILABLE:
        try:
            _ = DeepFace.build_model("Facenet")
            models['face_loaded'] = True
        except:
            models['face_loaded'] = False
    else:
        models['face_loaded'] = False
    
    if SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            models['sentence_model'] = None
    else:
        models['sentence_model'] = None
    
    if MP_AVAILABLE:
        try:
            models['face_mesh'] = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            models['hands'] = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except:
            models['face_mesh'] = None
            models['hands'] = None
    else:
        models['face_mesh'] = None
        models['hands'] = None
    
    if YOLO_AVAILABLE:
        try:
            models['yolo'] = YOLO("yolov8n.pt")
            models['yolo_cls'] = YOLO("yolov8n-cls.pt")
        except:
            models['yolo'] = None
            models['yolo_cls'] = None
    else:
        models['yolo'] = None
        models['yolo_cls'] = None
    
    return models

models = load_all_models()

# ==================== INITIALIZE SYSTEMS ====================
recording_system = RecordingSystemWebRTC(models)
analysis_system = AnalysisSystem(models)
scoring_dashboard = ScoringDashboard()

# ==================== SESSION STATE ====================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "results" not in st.session_state:
    st.session_state.results = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

# ==================== MAIN ROUTING ====================
if st.session_state.page == "home":
    show_home_page()

else:  # Interview page
    st.title("Interview Assessment Session")
    st.write("Complete all questions to receive your evaluation")
    
    # Simple navigation
    if not st.session_state.interview_complete:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.session_state.interview_started = False
            st.session_state.interview_complete = False
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Home"):
                st.session_state.page = "home"
                st.session_state.interview_started = False
                st.session_state.interview_complete = False
                st.rerun()
        with col2:
            if st.button("🔄 New Assessment"):
                st.session_state.results = []
                st.session_state.interview_started = False
                st.session_state.interview_complete = False
                st.rerun()
    
    st.markdown("---")
    
    # ==================== MAIN CONTENT ====================
    
    if not st.session_state.interview_started and not st.session_state.interview_complete:
        st.subheader("Ready to Begin?")
        st.write(f"""
        - You will respond to **{len(QUESTIONS)} questions**
        - Each question allows **20 seconds** for your response
        - The system will monitor compliance throughout
        - **Browser permissions required:** Camera & Microphone
        """)
        
        if st.button("Begin Assessment", type="primary"):
            st.session_state.interview_started = True
            st.rerun()
    
    elif st.session_state.interview_started and not st.session_state.interview_complete:
        # UI callbacks (simplified for WebRTC)
        ui_callbacks = {
            'countdown_update': lambda msg: None,
            'video_update': lambda frame: None,
            'status_update': lambda text: None,
            'progress_update': lambda val: None,
            'timer_update': lambda text: None,
            'question_update': lambda q_num, q_text, q_tip="": None
        }
        
        st.info("🎬 Starting WebRTC interview session...")
        
        # Run WebRTC interview
        session_result = recording_system.record_continuous_interview_webrtc(
            QUESTIONS, 
            duration_per_question=20,
            ui_callbacks=ui_callbacks
        )
        
        if isinstance(session_result, dict) and 'questions_results' in session_result:
            st.session_state.results = []
            
            for q_result in session_result['questions_results']:
                question_data = QUESTIONS[q_result['question_number'] - 1]
                analysis_results = analysis_system.analyze_recording(q_result, question_data, 20)
                
                result = {
                    "question": question_data["question"],
                    "video_path": session_result.get('session_video_path', ''),
                    "audio_path": q_result.get('audio_path', ''),
                    "transcript": q_result.get('transcript', ''),
                    "violations": q_result.get('violations', []),
                    "violation_detected": q_result.get('violation_detected', False),
                    "fused_emotions": analysis_results.get('fused_emotions', {}),
                    "emotion_scores": analysis_results.get('emotion_scores', {}),
                    "accuracy": analysis_results.get('accuracy', 0),
                    "fluency": analysis_results.get('fluency', 0),
                    "wpm": analysis_results.get('wpm', 0),
                    "blink_count": q_result.get('blink_count', 0),
                    "outfit": analysis_results.get('outfit', 'Unknown'),
                    "has_valid_data": analysis_results.get('has_valid_data', False),
                    "fluency_detailed": analysis_results.get('fluency_detailed', {}),
                    "fluency_level": analysis_results.get('fluency_level', 'No Data'),
                    "grammar_errors": analysis_results.get('grammar_errors', 0),
                    "filler_count": analysis_results.get('filler_count', 0),
                    "filler_ratio": analysis_results.get('filler_ratio', 0),
                    "improvements_applied": analysis_results.get('improvements_applied', {})
                }
                
                decision, reasons = scoring_dashboard.decide_hire(result)
                result["hire_decision"] = decision
                result["hire_reasons"] = reasons
                
                st.session_state.results.append(result)
            
            st.session_state.interview_complete = True
            
            total_violations = session_result.get('total_violations', 0)
            if total_violations > 0:
                st.warning(f"⚠️ Assessment completed with {total_violations} compliance issue(s).")
            else:
                st.success("🎉 Assessment completed successfully!")
            
            import time
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Assessment failed. Please try again.")
            st.session_state.interview_started = False
    
    else:
        # Show results
        scoring_dashboard.render_dashboard(st.session_state.results)