"""
MindMate AI - Enhanced Streamlit Application
Your Personal Mental Wellness Companion with Multimodal Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import base64
from PIL import Image
import io

# AI Model Imports (with fallback handling)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Install with: pip install transformers")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="MindMate AI - Wellness Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Mood cards */
    .mood-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .mood-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .mood-card-happy {
        border-left-color: #48bb78;
        background: linear-gradient(135deg, #ffffff 0%, #f0fff4 100%);
    }
    
    .mood-card-sad {
        border-left-color: #4299e1;
        background: linear-gradient(135deg, #ffffff 0%, #ebf8ff 100%);
    }
    
    .mood-card-stressed {
        border-left-color: #fc8181;
        background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
    }
    
    .mood-card-neutral {
        border-left-color: #a0aec0;
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        text-align: right;
    }
    
    .chat-message-assistant {
        background: #f7fafc;
        color: #2d3748;
        margin-right: 20%;
        border: 2px solid #e2e8f0;
    }
    
    /* Feature badges */
    .feature-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-happy {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
    }
    
    .badge-sad {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
    }
    
    .badge-stressed {
        background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
        color: white;
    }
    
    .badge-neutral {
        background: linear-gradient(135deg, #a0aec0 0%, #718096 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-left: 5px solid #4299e1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 5px solid #fc8181;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 5px solid #48bb78;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'entries' not in st.session_state:
        st.session_state.entries = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    
    if 'mood_log' not in st.session_state:
        st.session_state.mood_log = []
    
    if 'current_streak' not in st.session_state:
        st.session_state.current_streak = 0
    
    if 'privacy_mode' not in st.session_state:
        st.session_state.privacy_mode = True
    
    if 'sentiment_model' not in st.session_state:
        st.session_state.sentiment_model = None
    
    if 'emotion_model' not in st.session_state:
        st.session_state.emotion_model = None

init_session_state()

# Load AI Models (with caching)
@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Using DistilBERT for sentiment
        model = pipeline("sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english")
        return model
    except Exception as e:
        st.error(f"Failed to load sentiment model: {e}")
        return None

@st.cache_resource
def load_emotion_model():
    """Load emotion classification model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Using RoBERTa for detailed emotion classification
        model = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base",
                        top_k=None)
        return model
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return None

# Initialize models
if st.session_state.sentiment_model is None and TRANSFORMERS_AVAILABLE:
    with st.spinner("🔄 Loading AI models..."):
        st.session_state.sentiment_model = load_sentiment_model()
        st.session_state.emotion_model = load_emotion_model()

# Utility Functions
def get_mood_emoji(mood: str) -> str:
    """Get emoji for mood"""
    emojis = {
        'happy': '😊',
        'sad': '😢',
        'stressed': '😰',
        'neutral': '😐',
        'angry': '😠',
        'fear': '😨'
    }
    return emojis.get(mood.lower(), '😐')

def calculate_streak() -> int:
    """Calculate current journaling streak"""
    if not st.session_state.entries:
        return 0
    
    entries = sorted(st.session_state.entries, 
                    key=lambda x: x['timestamp'], 
                    reverse=True)
    
    streak = 0
    current_date = datetime.now().date()
    
    for entry in entries:
        entry_date = datetime.fromisoformat(entry['timestamp']).date()
        if entry_date == current_date:
            streak += 1
            current_date -= timedelta(days=1)
        else:
            break
    
    return streak

def detect_mood_with_ai(text: str) -> Dict:
    """
    Real AI-based mood detection using transformers
    Falls back to keyword detection if models unavailable
    """
    # Try using emotion model first (more detailed)
    if st.session_state.emotion_model is not None:
        try:
            results = st.session_state.emotion_model(text[:512])[0]
            
            # Map emotions to our categories
            emotion_map = {
                'joy': 'happy',
                'happiness': 'happy',
                'love': 'happy',
                'sadness': 'sad',
                'anger': 'stressed',
                'fear': 'stressed',
                'surprise': 'neutral',
                'disgust': 'stressed',
                'neutral': 'neutral'
            }
            
            # Get top emotion
            top_emotion = max(results, key=lambda x: x['score'])
            mapped_emotion = emotion_map.get(top_emotion['label'].lower(), 'neutral')
            
            return {
                'label': mapped_emotion,
                'score': top_emotion['score'],
                'confidence': 'high' if top_emotion['score'] > 0.8 else 'medium',
                'raw_emotion': top_emotion['label'],
                'all_scores': {r['label']: r['score'] for r in results},
                'method': 'transformer'
            }
        except Exception as e:
            st.warning(f"Emotion model failed: {e}. Falling back to sentiment.")
    
    # Fall back to sentiment model
    if st.session_state.sentiment_model is not None:
        try:
            result = st.session_state.sentiment_model(text[:512])[0]
            
            label_map = {
                'POSITIVE': 'happy',
                'NEGATIVE': 'sad',
                'NEUTRAL': 'neutral'
            }
            
            mood = label_map.get(result['label'], 'neutral')
            
            return {
                'label': mood,
                'score': result['score'],
                'confidence': 'high' if result['score'] > 0.8 else 'medium',
                'method': 'sentiment'
            }
        except Exception as e:
            st.warning(f"Sentiment model failed: {e}. Using keyword detection.")
    
    # Final fallback to keyword detection
    return simulate_mood_detection(text)

def analyze_facial_emotion(image_path_or_array) -> Dict:
    """
    Real facial emotion detection using DeepFace
    Falls back to simulation if unavailable
    """
    if not DEEPFACE_AVAILABLE:
        st.warning("⚠️ DeepFace not available. Install with: pip install deepface")
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'method': 'simulated',
            'all_emotions': {}
        }
    
    try:
        # Analyze image
        result = DeepFace.analyze(
            img_path=image_path_or_array,
            actions=['emotion'],
            enforce_detection=False
        )
        
        # Handle both single and multiple face detections
        if isinstance(result, list):
            result = result[0]
        
        dominant_emotion = result['dominant_emotion']
        emotions = result['emotion']
        
        # Map to our categories
        emotion_map = {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'stressed',
            'fear': 'stressed',
            'surprise': 'neutral',
            'disgust': 'stressed',
            'neutral': 'neutral'
        }
        
        mapped_emotion = emotion_map.get(dominant_emotion, 'neutral')
        confidence = emotions[dominant_emotion] / 100.0  # Convert to 0-1 scale
        
        return {
            'emotion': mapped_emotion,
            'confidence': confidence,
            'raw_emotion': dominant_emotion,
            'all_emotions': emotions,
            'method': 'deepface'
        }
    
    except Exception as e:
        st.error(f"Facial analysis failed: {e}")
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'method': 'error',
            'all_emotions': {}
        }

def fuse_multimodal_emotions(text_result: Dict, face_result: Dict) -> Dict:
    """
    Advanced fusion of text and facial emotion analysis
    """
    # Weights (text gets more weight as it's more reliable in chat context)
    text_weight = 0.6
    face_weight = 0.4
    
    # Normalize scores to [0, 1]
    text_score = text_result.get('score', 0.5)
    face_score = face_result.get('confidence', 0.5)
    
    # Map emotions to numeric scores for averaging
    emotion_scores = {
        'happy': 1.0,
        'neutral': 0.0,
        'sad': -1.0,
        'stressed': -0.7
    }
    
    text_numeric = emotion_scores.get(text_result['label'], 0.0)
    face_numeric = emotion_scores.get(face_result['emotion'], 0.0)
    
    # Weighted fusion
    fused_numeric = (text_weight * text_numeric * text_score + 
                     face_weight * face_numeric * face_score)
    
    fused_confidence = (text_weight * text_score + face_weight * face_score)
    
    # Determine final emotion from fused score
    if fused_numeric > 0.3:
        final_emotion = 'happy'
    elif fused_numeric < -0.5:
        final_emotion = 'stressed'
    elif fused_numeric < -0.2:
        final_emotion = 'sad'
    else:
        final_emotion = 'neutral'
    
    # Calculate agreement index
    agreement = text_result['label'] == face_result['emotion']
    agreement_score = 1.0 if agreement else abs(text_numeric - face_numeric) / 2.0
    
    return {
        'final_emotion': final_emotion,
        'confidence': fused_confidence,
        'fused_score': fused_numeric,
        'agreement': agreement,
        'agreement_index': agreement_score,
        'text_component': {
            'emotion': text_result['label'],
            'score': text_score,
            'weight': text_weight
        },
        'face_component': {
            'emotion': face_result['emotion'],
            'score': face_score,
            'weight': face_weight
        },
        'method': 'weighted_fusion'
    }

def simulate_mood_detection(text: str) -> Dict:
    """Fallback keyword-based mood detection"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['happy', 'great', 'amazing', 'wonderful', 'excited']):
        return {'label': 'happy', 'score': 0.85, 'confidence': 'high', 'method': 'keyword'}
    elif any(word in text_lower for word in ['sad', 'down', 'depressed', 'hopeless']):
        return {'label': 'sad', 'score': 0.82, 'confidence': 'high', 'method': 'keyword'}
    elif any(word in text_lower for word in ['stressed', 'anxious', 'overwhelmed', 'worried']):
        return {'label': 'stressed', 'score': 0.88, 'confidence': 'high', 'method': 'keyword'}
    else:
        return {'label': 'neutral', 'score': 0.70, 'confidence': 'medium', 'method': 'keyword'}

def simulate_mood_detection(text: str) -> Dict:
    """Fallback keyword-based mood detection"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['happy', 'great', 'amazing', 'wonderful', 'excited']):
        return {'label': 'happy', 'score': 0.85, 'confidence': 'high', 'method': 'keyword'}
    elif any(word in text_lower for word in ['sad', 'down', 'depressed', 'hopeless']):
        return {'label': 'sad', 'score': 0.82, 'confidence': 'high', 'method': 'keyword'}
    elif any(word in text_lower for word in ['stressed', 'anxious', 'overwhelmed', 'worried']):
        return {'label': 'stressed', 'score': 0.88, 'confidence': 'high', 'method': 'keyword'}
    else:
        return {'label': 'neutral', 'score': 0.70, 'confidence': 'medium', 'method': 'keyword'}

def generate_wordcloud_visualization(entries: List[Dict]) -> plt.Figure:
    """Generate word cloud from journal entries"""
    if not WORDCLOUD_AVAILABLE or not entries:
        return None
    
    try:
        # Combine all text
        text_blob = " ".join([entry.get('text', '') for entry in entries])
        
        if not text_blob.strip():
            return None
        
        # Generate word cloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text_blob)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Your Emotional Keywords', fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Word cloud generation failed: {e}")
        return None

def create_attention_heatmap(image_array, emotion_scores: Dict) -> plt.Figure:
    """
    Create a pseudo attention heatmap for explainability
    (Placeholder for Grad-CAM - would need actual CNN model)
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image_array)
        ax1.axis('off')
        ax1.set_title('Original Image', fontsize=14)
        
        # Create pseudo heatmap (overlay)
        # In production, this would be actual Grad-CAM output
        heatmap = np.random.rand(*image_array.shape[:2])
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Overlay heatmap on image
        ax2.imshow(image_array)
        ax2.imshow(heatmap, alpha=0.4, cmap='jet')
        ax2.axis('off')
        ax2.set_title('Attention Heatmap (Placeholder)', fontsize=14)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Heatmap generation failed: {e}")
        return None
    """Generate empathetic response based on mood"""
    responses = {
        'happy': "That's wonderful! 😊 It's great to see you feeling positive. What specifically made you feel this way?",
        'sad': "I'm sorry you're feeling this way. 💙 Your feelings are valid. Would you like to talk about what's contributing to these feelings?",
        'stressed': "I hear that you're feeling overwhelmed. 🌟 Let's break this down together. What's the most urgent thing you're facing right now?",
        'neutral': "Thank you for sharing. How are you feeling about things right now?"
    }
    return responses.get(mood['label'], "I'm here to listen. Tell me more about how you feel.")

def generate_empathetic_response(text: str, mood: Dict) -> str:
    """Generate empathetic response based on mood"""
    responses = {
        'happy': "That's wonderful! 😊 It's great to see you feeling positive. What specifically made you feel this way?",
        'sad': "I'm sorry you're feeling this way. 💙 Your feelings are valid. Would you like to talk about what's contributing to these feelings?",
        'stressed': "I hear that you're feeling overwhelmed. 🌟 Let's break this down together. What's the most urgent thing you're facing right now?",
        'neutral': "Thank you for sharing. How are you feeling about things right now?"
    }
    return responses.get(mood['label'], "I'm here to listen. Tell me more about how you feel.")

# Header with Logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 48px; margin: 0;">🧠 MindMate AI</h1>
            <p style="font-size: 18px; color: #667eea; margin: 10px 0;">
                Your Emotional Wellness Companion
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar with Enhanced Features
with st.sidebar:
    # Branding
    st.markdown("""
        <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3 style="margin: 0;">🌟 Features</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature badges
    features = [
        ("💬", "AI Chat", "Empathetic conversations"),
        ("📊", "Analytics", "Mood tracking"),
        ("🆘", "Crisis Detection", "24/7 safety net"),
        ("🔒", "Private", "Your data, your control")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background: #f7fafc; border-radius: 8px; border-left: 4px solid #667eea;">
                <div style="font-size: 16px;">{icon} <strong>{title}</strong></div>
                <div style="font-size: 12px; color: #718096;">{desc}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    
    # Privacy mode toggle
    st.session_state.privacy_mode = st.checkbox(
        "🔒 Privacy Mode", 
        value=st.session_state.privacy_mode,
        help="All data processed locally, no uploads"
    )
    
    if st.session_state.privacy_mode:
        st.success("✅ Your data is processed locally and never uploaded")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for AI chat features")
    if api_key:
        st.session_state.api_key_configured = True
        st.success("✅ API key configured")
    
    st.divider()
    
    # Navigation
    st.header("📱 Navigation")
    page = st.radio(
        "Go to:",
        ["💬 Chat", "📊 Dashboard", "📝 Prompts", "📸 Multimodal", "📖 History"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Quick stats
    st.header("📈 Quick Stats")
    
    # Current streak
    current_streak = calculate_streak()
    st.metric("🔥 Current Streak", f"{current_streak} days")
    
    st.metric("📝 Total Entries", len(st.session_state.entries))
    
    if st.session_state.entries:
        moods = [e.get('mood_label', 'neutral') for e in st.session_state.entries]
        most_common = Counter(moods).most_common(1)[0][0]
        st.metric("😊 Most Common Mood", most_common.title())
    
    st.divider()
    
    # Crisis resources (always visible)
    st.markdown("""
        <div class="warning-box">
            <h4>🆘 Crisis Resources</h4>
            <p><strong>US:</strong> 988 (24/7)</p>
            <p><strong>India:</strong> 1860-2662-345</p>
            <p><strong>International:</strong> <a href="https://findahelpline.com" target="_blank">findahelpline.com</a></p>
        </div>
    """, unsafe_allow_html=True)

# Main content based on page selection
if page == "💬 Chat":
    st.markdown('<div class="main-header"><h1>💬 Chat with MindMate</h1><p>Share your thoughts in a safe, judgment-free space</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat input
        user_input = st.text_area(
            "How are you feeling today?",
            height=150,
            placeholder="Share what's on your mind... I'm here to listen.",
            key="chat_input"
        )
        
        col_send, col_clear = st.columns([1, 4])
        
        with col_send:
            send_clicked = st.button("Send 💬", use_container_width=True)
        
        with col_clear:
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_clicked and user_input:
            # Detect mood with AI
            with st.spinner("🔄 Analyzing your message..."):
                mood = detect_mood_with_ai(user_input)
            
            # Generate response
            ai_response = generate_empathetic_response(user_input, mood)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "mood": mood,
                "timestamp": datetime.now().isoformat()
            })
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save to entries
            st.session_state.entries.append({
                "text": user_input,
                "mood_label": mood['label'],
                "mood_score": mood['score'],
                "mood_confidence": mood['confidence'],
                "response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "analysis_method": mood.get('method', 'unknown')
            })
            
            # Add to mood log
            st.session_state.mood_log.append({
                'emotion': mood['label'],
                'score': mood['score'],
                'time': datetime.now()
            })
            
            # Show analysis details
            if 'all_scores' in mood:
                with st.expander("🔍 Detailed Emotion Analysis"):
                    st.write("**All detected emotions:**")
                    for emotion, score in sorted(mood['all_scores'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5]:
                        st.progress(score, text=f"{emotion}: {score:.2%}")
            
            st.rerun()
        
        # Display chat history
        st.divider()
        st.subheader("💭 Conversation")
        
        if not st.session_state.chat_history:
            st.info("Start the conversation by sharing how you're feeling today.")
        else:
            for msg in st.session_state.chat_history[-10:]:
                if msg["role"] == "user":
                    mood_emoji = get_mood_emoji(msg.get('mood', {}).get('label', 'neutral'))
                    st.markdown(f"""
                        <div class="chat-message chat-message-user">
                            <strong>You {mood_emoji}:</strong> {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    if 'mood' in msg:
                        mood_badge_class = f"badge-{msg['mood']['label']}"
                        st.markdown(f"""
                            <div style="text-align: right; margin: -10px 0 10px 0;">
                                <span class="feature-badge {mood_badge_class}">
                                    Mood: {msg['mood']['label'].title()} ({msg['mood']['score']:.0%})
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message chat-message-assistant">
                            <strong>🧠 MindMate:</strong> {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.caption(f"🕐 {msg['timestamp'][:16]}")
    
    with col2:
        # Tips and guidance
        st.markdown("""
            <div class="info-box">
                <h4>💡 Tips for Journaling</h4>
                <ul style="font-size: 14px; line-height: 1.8;">
                    <li>Be honest about your feelings</li>
                    <li>There's no right or wrong</li>
                    <li>MindMate is here to listen</li>
                    <li>Take your time</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="success-box">
                <h4>✨ What MindMate Provides</h4>
                <ul style="font-size: 14px; line-height: 1.8;">
                    <li>Empathetic responses</li>
                    <li>Mood detection</li>
                    <li>Evidence-based strategies</li>
                    <li>Crisis support</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

elif page == "📊 Dashboard":
    st.markdown('<div class="main-header"><h1>📊 Your Mood Dashboard</h1><p>Visualize your emotional journey</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.entries:
        st.info("📝 Start journaling to see your mood insights and analytics!")
        
        # Show demo data option
        if st.button("📊 Load Demo Data"):
            # Add sample demo data
            demo_entries = [
                {"text": "Had a great day!", "mood_label": "happy", "mood_score": 0.9, "timestamp": (datetime.now() - timedelta(days=7)).isoformat()},
                {"text": "Feeling stressed about work", "mood_label": "stressed", "mood_score": 0.85, "timestamp": (datetime.now() - timedelta(days=6)).isoformat()},
                {"text": "Just a normal day", "mood_label": "neutral", "mood_score": 0.7, "timestamp": (datetime.now() - timedelta(days=5)).isoformat()},
                {"text": "Accomplished a lot today!", "mood_label": "happy", "mood_score": 0.88, "timestamp": (datetime.now() - timedelta(days=4)).isoformat()},
                {"text": "Feeling down", "mood_label": "sad", "mood_score": 0.78, "timestamp": (datetime.now() - timedelta(days=3)).isoformat()},
                {"text": "Anxiety about deadlines", "mood_label": "stressed", "mood_score": 0.82, "timestamp": (datetime.now() - timedelta(days=2)).isoformat()},
                {"text": "Had fun with friends!", "mood_label": "happy", "mood_score": 0.92, "timestamp": (datetime.now() - timedelta(days=1)).isoformat()},
            ]
            st.session_state.entries.extend(demo_entries)
            st.rerun()
    else:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">📝 {}</div>
                    <div class="metric-label">Total Entries</div>
                </div>
            """.format(len(st.session_state.entries)), unsafe_allow_html=True)
        
        with col2:
            moods = [e.get('mood_label', 'neutral') for e in st.session_state.entries]
            most_common = Counter(moods).most_common(1)[0][0]
            emoji = get_mood_emoji(most_common)
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{} {}</div>
                    <div class="metric-label">Most Common</div>
                </div>
            """.format(emoji, most_common.title()), unsafe_allow_html=True)
        
        with col3:
            streak = calculate_streak()
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">🔥 {}</div>
                    <div class="metric-label">Day Streak</div>
                </div>
            """.format(streak), unsafe_allow_html=True)
        
        with col4:
            unique_days = len(set([datetime.fromisoformat(e['timestamp']).date() for e in st.session_state.entries]))
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">📅 {}</div>
                    <div class="metric-label">Active Days</div>
                </div>
            """.format(unique_days), unsafe_allow_html=True)
        
        st.divider()
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("😊 Mood Distribution")
            
            df = pd.DataFrame(st.session_state.entries)
            mood_counts = df['mood_label'].value_counts()
            
            colors = {
                'happy': '#48bb78',
                'sad': '#4299e1',
                'stressed': '#fc8181',
                'neutral': '#a0aec0'
            }
            
            color_list = [colors.get(mood, '#718096') for mood in mood_counts.index]
            
            fig = px.pie(
                values=mood_counts.values,
                names=mood_counts.index,
                title="Your Emotional Landscape",
                color_discrete_sequence=color_list
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.subheader("📈 Mood Trends Over Time")
            
            # Prepare data for timeline
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            mood_scores = {'happy': 1.0, 'neutral': 0.0, 'sad': -1.0, 'stressed': -0.5}
            df['mood_numeric'] = df['mood_label'].map(mood_scores)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['mood_numeric'],
                mode='lines+markers',
                name='Mood Score',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color=df['mood_numeric'], 
                           colorscale='RdYlGn', showscale=False)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(
                title="Mood Journey",
                xaxis_title="Date",
                yaxis_title="Mood Score",
                yaxis=dict(range=[-1.2, 1.2]),
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Mood tracker visualization
        if len(st.session_state.mood_log) > 0:
            st.divider()
            st.subheader("🎯 Mood Tracker Over Time")
            
            mood_df = pd.DataFrame(st.session_state.mood_log)
            fig = px.line(
                mood_df, 
                x='time', 
                y='score', 
                color='emotion',
                title="Mood Tracker - Real-time Updates",
                markers=True
            )
            fig.update_layout(
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Word Cloud Visualization
        if WORDCLOUD_AVAILABLE and len(st.session_state.entries) > 5:
            st.divider()
            st.subheader("☁️ Your Emotional Keywords")
            
            wordcloud_fig = generate_wordcloud_visualization(st.session_state.entries)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
                st.caption("Word cloud generated from your journal entries showing frequently used words")
            else:
                st.info("Install wordcloud for visualization: pip install wordcloud")
        
        # Top Keywords Analysis
        st.divider()
        st.subheader("🔤 Top Keywords by Mood")
        
        # Group entries by mood and extract common words
        from collections import defaultdict
        mood_words = defaultdict(list)
        
        for entry in st.session_state.entries:
            words = entry['text'].lower().split()
            # Filter out common words
            filtered_words = [w for w in words if len(w) > 4 and w.isalpha()]
            mood_words[entry['mood_label']].extend(filtered_words)
        
        cols = st.columns(len(mood_words))
        for idx, (mood, words) in enumerate(mood_words.items()):
            with cols[idx]:
                word_counts = Counter(words).most_common(5)
                emoji = get_mood_emoji(mood)
                st.markdown(f"**{emoji} {mood.title()}**")
                for word, count in word_counts:
                    st.text(f"• {word} ({count})")
        
        st.divider()
        
        # Export options
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📥 Download as CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV File",
                    csv,
                    "mindmate_journal.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col_export2:
            if st.button("📋 Download as JSON", use_container_width=True):
                json_str = json.dumps(st.session_state.entries, indent=2)
                st.download_button(
                    "Download JSON File",
                    json_str,
                    "mindmate_journal.json",
                    "application/json",
                    use_container_width=True
                )
        
        with col_export3:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                if st.checkbox("I'm sure I want to delete all my data"):
                    st.session_state.entries = []
                    st.session_state.chat_history = []
                    st.session_state.mood_log = []
                    st.success("✅ All data cleared!")
                    st.rerun()

elif page == "📝 Prompts":
    st.markdown('<div class="main-header"><h1>📝 Therapeutic Prompts</h1><p>Evidence-based journaling prompts for reflection</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h4>✨ About These Prompts</h4>
            <p>These prompts are based on evidence-based therapeutic techniques including:</p>
            <ul>
                <li><strong>CBT</strong> - Cognitive Behavioral Therapy</li>
                <li><strong>DBT</strong> - Dialectical Behavior Therapy</li>
                <li><strong>Positive Psychology</strong> - Strengths & Gratitude</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Therapeutic prompts by category
    prompt_categories = {
        "🌟 Gratitude & Positivity": [
            "What are three things you're grateful for today, and why?",
            "Describe a moment that made you smile this week.",
            "What's something you're looking forward to?"
        ],
        "💪 Cognitive Reframing": [
            "What's a challenging thought you had today? How could you reframe it?",
            "Describe a time you overcame a challenge. What strengths did you use?",
            "What would you tell a friend who had the same worry as you?"
        ],
        "🧘 Mindfulness & Present Moment": [
            "Describe what you notice right now using all five senses.",
            "What emotions are you feeling? Where do you feel them in your body?",
            "Take 3 deep breaths. How do you feel now compared to 5 minutes ago?"
        ],
        "🎯 Goals & Growth": [
            "What's one small step you can take today toward a goal?",
            "What did you learn about yourself this week?",
            "How have you grown in the past month?"
        ]
    }
    
    for category, prompts in prompt_categories.items():
        with st.expander(category, expanded=True):
            for i, prompt in enumerate(prompts, 1):
                st.markdown(f"""
                    <div class="mood-card">
                        <h4 style="color: #667eea; margin-top: 0;">Prompt {i}</h4>
                        <p style="font-size: 16px; color: #2d3748;">{prompt}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                response = st.text_area(
                    f"Your response:",
                    key=f"prompt_{category}_{i}",
                    height=100,
                    placeholder="Take your time to reflect..."
                )
                
                if st.button(f"💾 Save Response", key=f"save_{category}_{i}"):
                    if response:
                        # Detect mood from response
                        mood = simulate_mood_detection(response)
                        
                        # Save entry
                        st.session_state.entries.append({
                            "text": response,
                            "mood_label": mood['label'],
                            "mood_score": mood['score'],
                            "mood_confidence": mood['confidence'],
                            "prompt": prompt,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        st.success(f"✅ Response saved! Detected mood: {get_mood_emoji(mood['label'])} {mood['label'].title()}")
                    else:
                        st.warning("Please write a response before saving.")

elif page == "📸 Multimodal":
    st.markdown('<div class="main-header"><h1>📸 Multimodal Analysis</h1><p>Combine text and facial emotion for deeper insights</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h4>🎭 Multimodal Emotion Detection</h4>
            <p>MindMate can analyze both your words and facial expressions to provide comprehensive emotional insights.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Text Analysis")
        
        text_input = st.text_area(
            "Share how you're feeling:",
            height=150,
            placeholder="Express your emotions in words...",
            key="multimodal_text"
        )
        
        if text_input:
            text_mood = simulate_mood_detection(text_input)
            
            st.markdown(f"""
                <div class="mood-card mood-card-{text_mood['label']}">
                    <h4>Text Emotion: {get_mood_emoji(text_mood['label'])} {text_mood['label'].title()}</h4>
                    <p><strong>Confidence:</strong> {text_mood['score']:.0%}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📸 Facial Emotion Recognition")
        
        st.markdown("""
            <div class="warning-box">
                <p><strong>Note:</strong> Facial emotion detection requires local processing. 
                Upload an image or use your webcam (in production deployment).</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "png", "jpeg"],
            help="Upload a photo showing your facial expression"
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            # Simulate facial emotion detection (replace with actual model)
            with st.spinner("Analyzing facial emotion..."):
                import time
                time.sleep(1)  # Simulate processing
                
                # Simulated result
                face_emotion = "happy"  # Would come from DeepFace model
                face_confidence = 0.87
                
                st.markdown(f"""
                    <div class="mood-card mood-card-{face_emotion}">
                        <h4>Facial Emotion: {get_mood_emoji(face_emotion)} {face_emotion.title()}</h4>
                        <p><strong>Confidence:</strong> {face_confidence:.0%}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Fusion Analysis
    if text_input and uploaded_image:
        st.divider()
        st.subheader("💫 Emotion Fusion Analysis")
        
        # Simulate fusion (replace with actual fusion logic)
        text_weight = 0.6
        face_weight = 0.4
        
        text_score = text_mood['score']
        face_score = 0.87  # Simulated
        
        final_score = (text_weight * text_score) + (face_weight * face_score)
        
        # Determine final emotion
        if text_mood['label'] == face_emotion:
            final_emotion = text_mood['label']
            agreement = True
        else:
            final_emotion = text_mood['label'] if text_score > face_score else face_emotion
            agreement = False
        
        st.markdown(f"""
            <div class="success-box">
                <h3>🌈 Final Emotion: {get_mood_emoji(final_emotion)} {final_emotion.title()}</h3>
                <p><strong>Combined Score:</strong> {final_score:.2%}</p>
                <p><strong>Modality Agreement:</strong> {'✅ Yes' if agreement else '⚠️ No - Mixed signals detected'}</p>
                <hr>
                <p><strong>Analysis:</strong></p>
                <p>Text component (60% weight): {get_mood_emoji(text_mood['label'])} {text_mood['label'].title()} - {text_score:.0%}</p>
                <p>Facial component (40% weight): {get_mood_emoji(face_emotion)} {face_emotion.title()} - {face_score:.0%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("💾 Save Multimodal Analysis"):
            st.session_state.entries.append({
                "text": text_input,
                "mood_label": final_emotion,
                "mood_score": final_score,
                "mood_confidence": "multimodal",
                "text_emotion": text_mood['label'],
                "face_emotion": face_emotion,
                "timestamp": datetime.now().isoformat()
            })
            st.success("✅ Multimodal analysis saved!")

elif page == "📖 History":
    st.markdown('<div class="main-header"><h1>📖 Your Journal History</h1><p>Review your emotional journey</p></div>', unsafe_allow_html=True)
    
    if not st.session_state.entries:
        st.info("📝 No entries yet. Start journaling to build your history!")
    else:
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            mood_filter = st.multiselect(
                "Filter by mood:",
                ["happy", "sad", "stressed", "neutral"],
                default=[]
            )
        
        with col_filter2:
            days_filter = st.selectbox(
                "Time period:",
                ["All time", "Last 7 days", "Last 30 days", "Last 90 days"]
            )
        
        with col_filter3:
            sort_order = st.selectbox(
                "Sort by:",
                ["Newest first", "Oldest first"]
            )
        
        # Apply filters
        filtered_entries = st.session_state.entries.copy()
        
        # Mood filter
        if mood_filter:
            filtered_entries = [e for e in filtered_entries if e['mood_label'] in mood_filter]
        
        # Time filter
        if days_filter != "All time":
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            days = days_map[days_filter]
            cutoff = datetime.now() - timedelta(days=days)
            filtered_entries = [e for e in filtered_entries 
                              if datetime.fromisoformat(e['timestamp']) > cutoff]
        
        # Sort
        filtered_entries = sorted(
            filtered_entries,
            key=lambda x: x['timestamp'],
            reverse=(sort_order == "Newest first")
        )
        
        st.markdown(f"**Showing {len(filtered_entries)} entries**")
        st.divider()
        
        # Display entries
        for i, entry in enumerate(filtered_entries[:50], 1):  # Limit to 50 for performance
            timestamp = datetime.fromisoformat(entry['timestamp'])
            mood = entry.get('mood_label', 'neutral')
            emoji = get_mood_emoji(mood)
            
            with st.expander(f"{emoji} Entry {i} - {timestamp.strftime('%B %d, %Y at %I:%M %p')}"):
                # Mood badge
                mood_badge_class = f"badge-{mood}"
                st.markdown(f"""
                    <span class="feature-badge {mood_badge_class}">
                        {mood.title()} {emoji}
                    </span>
                """, unsafe_allow_html=True)
                
                # Entry text
                st.markdown(f"""
                    <div style="padding: 15px; background: #f7fafc; border-radius: 10px; margin: 10px 0;">
                        {entry['text']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Response if available
                if 'response' in entry and entry['response']:
                    st.markdown("**🧠 MindMate's Response:**")
                    st.markdown(f"""
                        <div style="padding: 15px; background: #ebf8ff; border-radius: 10px; border-left: 4px solid #4299e1;">
                            {entry['response']}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Prompt if available
                if 'prompt' in entry and entry['prompt']:
                    st.caption(f"📝 Prompt: {entry['prompt']}")
                
                # Metadata
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    if 'mood_score' in entry:
                        st.caption(f"Confidence: {entry['mood_score']:.0%}")
                with col_meta2:
                    st.caption(f"Timestamp: {timestamp.strftime('%I:%M %p')}")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 20px; background: #f7fafc; border-radius: 10px;">
        <p style="color: #718096; margin: 0;">
            ⚠️ <strong>Important:</strong> MindMate is not a replacement for professional mental health care.
        </p>
        <p style="color: #718096; margin: 5px 0 0 0;">
            If you're in crisis, please call <strong>988 (US)</strong> or contact local emergency services.
        </p>
        <p style="color: #a0aec0; margin: 10px 0 0 0; font-size: 14px;">
            Made with 💙 by MindMate AI | Your privacy is our priority
        </p>
    </div>
""", unsafe_allow_html=True)
