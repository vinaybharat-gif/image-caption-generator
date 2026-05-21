import os
from gtts import gTTS
from googletrans import Translator

def translate_caption(caption, target_lang):
    """
    Translates English text into Hindi ('hi') or Telugu ('te') using Googletrans.
    """
    try:
        translator = Translator()
        translation = translator.translate(caption, dest=target_lang)
        return translation.text
    except Exception as e:
        return f"[Translation Error: Please check internet connection. {e}]"

def text_to_speech(text, lang='en'):
    """
    Converts caption text into an audio file using gTTS.
    Returns the path to the saved audio file.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_path = "assets/caption_audio.mp3"
        
        # Ensure assets directory exists
        os.makedirs("assets", exist_ok=True)
        tts.save(audio_path)
        return audio_path
    except Exception:
        return None