import pandas as pd  # type: ignore
import psycopg2  # type: ignore
from datetime import datetime, timezone
from openai import OpenAI
import streamlit as st
from typing import Dict, Any, Optional


# Konfiguracja strony - musi być na początku
st.set_page_config(
    page_title="LinguaTales",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicjalizacja OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["api_keys"]["OPENAI_API_KEY"])

openai_client = get_openai_client()

# Centralna inicjalizacja session state
def init_session_state():
    """Inicjalizuje wszystkie potrzebne zmienne session state"""
    defaults = {
        'audio_bytes': None,
        'user_input': "",
        'story': None,
        'language': "English",
        'level': "Intermediate",
        'flashcards': [],
        'generation_in_progress': False,
        'audio_generation_in_progress': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Stałe konfiguracyjne
LANGUAGE_CONFIGS = {
    "English": {
        "code": "english",
        "hint": "💬 Please enter the story idea in English to get a natural result.",
        "tts_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    },
    "Spanish": {
        "code": "spanish", 
        "hint": "💬 Por favor, escribe el tema en español para obtener una historia coherente.",
        "tts_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    }
}

DIFFICULTY_CONFIGS = {
    "Beginner": {
        "level": "Beginner (A1-A2)",
        "instruction": "Use very simple vocabulary and short sentences. Avoid complex grammar structures. Use present tense most often. Define any words that might be challenging."
    },
    "Intermediate": {
        "level": "Intermediate (B1-B2)", 
        "instruction": "Use a mix of simple and more advanced vocabulary. Include some compound sentences and varied grammar structures."
    },
    "Advanced": {
        "level": "Advanced (C1-C2)",
        "instruction": "Use advanced vocabulary, idioms, and varied complex sentence structures. Don't shy away from sophisticated language."
    }
}

# Database functions
#@st.cache_resource
# def get_connection():
#     """Cached database connection"""
#     return psycopg2.connect(
#         dbname=st.secrets["database"]["database"],
#         user=st.secrets["database"]["username"],
#         password=st.secrets["database"]["password"],
#         host=st.secrets["database"]["host"],
#         port=st.secrets["database"]["port"],
#         sslmode=st.secrets["database"]["sslmode"]
#     )

# def insert_usage(email: str, output_tokens: int, input_tokens: int, input_text: str) -> None:
#     """Zapisuje informacje o użyciu do bazy danych"""
#     try:
#         with get_connection() as conn:
#             with conn.cursor() as cur:
#                 cur.execute("""
#                     INSERT INTO usages (google_user_email, output_tokens, input_tokens, input_text) 
#                     VALUES (%s, %s, %s, %s)
#                 """, (email, output_tokens, input_tokens, input_text))
#                 conn.commit()
#     except Exception as e:
#         st.error(f"Database error: {str(e)}")

# def get_current_month_usage_df(email: str) -> pd.DataFrame:
#     """Pobiera statystyki użycia dla bieżącego miesiąca"""
#     try:
#         with get_connection() as conn:
#             now = datetime.now(timezone.utc)
#             start_date = datetime(now.year, now.month, 1)
#             with conn.cursor() as cur:
#                 cur.execute(
#                     "SELECT * FROM usages WHERE google_user_email = %s AND created_at >= %s", 
#                     (email, start_date)
#                 )
#                 rows = cur.fetchall()
#                 columns = [desc[0] for desc in cur.description]
#                 return pd.DataFrame(rows, columns=columns)
#     except Exception as e:
#         st.error(f"Database error: {str(e)}")
#         return pd.DataFrame()

# Story generation functions
def validate_story_input(story_prompt: str) -> None:
    """Waliduje dane wejściowe dla generowania opowiadania"""
    if not story_prompt or len(story_prompt.strip()) < 2:
        raise ValueError("Story idea is too short. Please provide at least 2 characters.")
    
    if len(story_prompt.strip()) > 1000:
        raise ValueError("Story idea is too long. Please keep it under 1000 characters.")

def create_story(story_prompt: str) -> Dict[str, Any]:
    """Generuje opowiadanie przy użyciu OpenAI GPT"""
    
    # Walidacja
    validate_story_input(story_prompt)
    
    # Pobieranie konfiguracji
    language = st.session_state.get("language", "English")
    difficulty = st.session_state.get("level", "Intermediate")
    
    lang_config = LANGUAGE_CONFIGS[language]
    diff_config = DIFFICULTY_CONFIGS[difficulty]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You are a professional storyteller.
                        Create engaging short stories based on the provided text.
                        Stories should be surprising and well-structured with:
                        - Clear introduction
                        - Engaging body with conflict/tension
                        - Satisfying conclusion
                        
                        Story language: {lang_config['code']}
                        Difficulty level: {diff_config['level']}
                        {diff_config['instruction']}
                        
                        The story should be 150-300 words long.
                        Make it memorable and educational.
                    """
                },
                {"role": "user", "content": story_prompt.strip()}
            ],
            max_tokens=500,
            temperature=0.8
        )
        
        # Przetwarzanie odpowiedzi
        usage = {}
        if response.usage:
            usage = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        # # Zapisywanie do bazy danych
        # if hasattr(st, 'user') and st.user.email:
        #     insert_usage(
        #         email=st.user.email,
        #         output_tokens=usage.get('completion_tokens', 0),
        #         input_tokens=usage.get('prompt_tokens', 0),
        #         input_text=story_prompt,
        #     )
        
        return {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "usage": usage,
        }
        
    except Exception as e:
        raise Exception(f"Error generating story: {str(e)}")

def generate_speech(text: str, voice: str) -> Optional[bytes]:
    """Generuje mowę z tekstu przy użyciu OpenAI TTS"""
    try:
        if not text.strip():
            return None
            
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text[:4000]  # Limit długości tekstu dla TTS
        )
        return response.content
        
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# UI Helper functions
def render_header():
    """Renderuje nagłówek aplikacji"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        try:
            st.image("logo.png", width=110)
        except:
            st.markdown("📚")  # Fallback jeśli brak logo
    
    with col2:
        st.markdown("<h1 style='margin-top: 0px;'>LinguaTales</h1>", unsafe_allow_html=True)
    
    st.markdown("#### _Because language is best learned in context!_")
    
    st.write("""
    LinguaTales is an innovative application that supports learning English and Spanish 
    through engaging, short stories created by artificial intelligence.

    **How it works:**
             
    🔤 Choose a language  
    🎯 Set the difficulty level  
    💡 Enter a story idea – You can type a full sentence, a few keywords, or just a topic  
    🗣️ Pick a narrator's voice – AI will read the story aloud for you  
    📇 Manually collect tricky words while reading  
    ⬇️ Download the story, flashcards and audio to study offline
    """)

def render_story_generation():
    """Renderuje sekcję generowania opowiadań"""
    st.markdown("---")
    st.subheader("✍️ Create Your Story")
    
    # Wybór języka
    language = st.selectbox(
        "Select language:", 
        list(LANGUAGE_CONFIGS.keys()),
        index=list(LANGUAGE_CONFIGS.keys()).index(st.session_state["language"])
    )
    st.session_state["language"] = language
    
    # Wyświetlenie wskazówki dla wybranego języka
    st.info(LANGUAGE_CONFIGS[language]["hint"])
    
    # Wybór poziomu
    level = st.selectbox(
        "Select the level of difficulty:", 
        list(DIFFICULTY_CONFIGS.keys()),
        index=list(DIFFICULTY_CONFIGS.keys()).index(st.session_state["level"])
    )
    st.session_state["level"] = level
    
    # Input dla pomysłu na opowiadanie
    user_input = st.text_area(
        "What's your story idea?", 
        value=st.session_state['user_input'],
        max_chars=1000,
        help="Enter keywords, phrases, or a topic for your story"
    )
    st.session_state['user_input'] = user_input
    
    # Przycisk generowania
    generate_disabled = (
        not user_input.strip() or 
        st.session_state.get('generation_in_progress', False)
    )
    
    if st.button(
        "✍️ Generate story", 
        disabled=generate_disabled,
        use_container_width=True
    ):
        st.session_state['generation_in_progress'] = True
        
        with st.spinner("Generating your story..."):
            try:
                st.session_state['story'] = create_story(user_input)
                st.success("✅ Story generated successfully!")
                # Reset audio gdy wygenerowano nowe opowiadanie
                st.session_state['audio_bytes'] = None
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                st.session_state['generation_in_progress'] = False
                st.rerun()

def render_story_display():
    """Renderuje wygenerowane opowiadanie"""
    if st.session_state.get('story'):
        st.markdown("### 📖 Your Generated Story")
        
        # Wyświetlenie opowiadania w ładnym kontenerze
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #ff6b6b;
                margin: 10px 0;
            ">
                {st.session_state['story']['content']}
            </div>
            """, unsafe_allow_html=True)
        
        # Przycisk pobierania
        st.download_button(
            label="⬇️ Download story",
            data=st.session_state['story']['content'],
            file_name=f"linguatales_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

def render_audio_section():
    """Renderuje sekcję audio"""
    st.markdown("---")
    st.subheader("🎧 Listen to Your Story")
    
    if not st.session_state.get('story'):
        st.info("👆 Generate a story first to enable audio features")
        return
    
    # Wybór głosu
    current_language = st.session_state.get("language", "English")
    available_voices = LANGUAGE_CONFIGS[current_language]["tts_voices"]
    
    voice = st.selectbox(
        "Select narrator voice:",
        available_voices,
        help="Different voices have different characteristics - try them out!"
    )
    
    # Przycisk generowania audio
    audio_disabled = (
        not st.session_state.get('story') or
        st.session_state.get('audio_generation_in_progress', False)
    )
    
    if st.button(
        "🎧 Generate audio", 
        disabled=audio_disabled,
        use_container_width=True
    ):
        st.session_state['audio_generation_in_progress'] = True
        
        with st.spinner("Generating audio..."):
            story_text = st.session_state['story']['content']
            audio_bytes = generate_speech(story_text, voice)
            
            if audio_bytes:
                st.session_state['audio_bytes'] = audio_bytes
                st.success("✅ Audio generated successfully!")
            else:
                st.error("❌ Failed to generate audio. Please try again.")
                
            st.session_state['audio_generation_in_progress'] = False
            st.rerun()

def render_audio_player():
    """Renderuje odtwarzacz audio"""
    if st.session_state.get('audio_bytes'):
        st.audio(st.session_state['audio_bytes'], format="audio/mp3")
        
        st.download_button(
            label="⬇️ Download audio",
            data=st.session_state['audio_bytes'],
            file_name=f"linguatales_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mp3",
            use_container_width=True,
        )

def render_flashcards_sidebar():
    """Renderuje sekcję flashcards w sidebarze"""
    with st.sidebar:
        try:
            st.image("logo.png", width=110)
            st.markdown("Learn languages ​​with AI-generated stories.")
        except:
            st.markdown("# 📚")
        
        st.markdown("---")
        st.header("📇 Flashcards")
        
        # Input dla nowego słowa
        new_word = st.text_input(
            "Add a word to flashcards:",
            help="Add difficult words from the story for later review"
        )
        
        # Przycisk dodawania
        if st.button("➕ Add to flashcards") and new_word.strip():
            word = new_word.strip().lower()
            if word not in [w.lower() for w in st.session_state.flashcards]:
                st.session_state.flashcards.append(new_word.strip())
                st.success(f"✅ Added: {new_word}")
                st.rerun()
            else:
                st.warning("⚠️ Word already in flashcards!")
        
        # Wyświetlenie flashcards
        if st.session_state.flashcards:
            st.markdown("### 📚 Your Words:")
            
            for i, word in enumerate(st.session_state.flashcards):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{i+1}.** {word}")
                with col2:
                    if st.button("🗑️", key=f"del_{i}", help="Remove word"):
                        st.session_state.flashcards.pop(i)
                        st.rerun()
            
            # Przycisk pobierania flashcards
            flashcards_content = "\n".join([f"{i+1}. {word}" for i, word in enumerate(st.session_state.flashcards)])
            st.download_button(
                "⬇️ Download Flashcards",
                data=flashcards_content,
                file_name=f"linguatales_flashcards_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Przycisk czyszczenia wszystkich flashcards
            if st.button("🗑️ Clear all", help="Remove all flashcards"):
                st.session_state.flashcards = []
                st.rerun()
        else:
            st.info("No flashcards yet. Add some words while reading!")

def render_sidebar_footer():
    """Renderuje stopkę sidebara"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ☕ Support the Project")
        st.markdown("If you like the app, you can support me. Thank you!")
        st.link_button(
            "☕ Buy me a coffee", 
            "https://buymeacoffee.com/kgustak93u", 
            use_container_width=True
        )
        
        st.markdown("---")
        st.link_button(
            "🔒 Privacy Policy", 
            "https://opowiadania-assets.fra1.cdn.digitaloceanspaces.com/privacy_policy.pdf",
            use_container_width=True
        )
        st.link_button(
            "📋 Terms of Service", 
            "https://opowiadania-assets.fra1.cdn.digitaloceanspaces.com/regulations.pdf",
            use_container_width=True
        )

def render_footer():
    """Renderuje stopkę aplikacji"""
    st.markdown("---")
    st.markdown(
        " 📚 **LinguaTales** — An AI-powered story generator for language learning through engaging stories!"
    )

# MAIN APPLICATION
def main():
    """Główna funkcja aplikacji"""
    
    # Renderowanie komponentów
    render_header()
    render_story_generation()
    render_story_display()
    render_audio_section()
    render_audio_player()
    render_flashcards_sidebar()
    render_sidebar_footer()
    render_footer()

if __name__ == "__main__":
    main()