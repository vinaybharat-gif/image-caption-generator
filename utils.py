import os
import time
from gtts import gTTS
from deep_translator import GoogleTranslator

def translate_caption(caption, target_lang):
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(caption)
        return translated
    except Exception as e:
        return f"[Translation Error: {e}]"

def text_to_speech(text, lang='en'):
    try:
        os.makedirs("assets", exist_ok=True)
        timestamp = int(time.time())
        audio_path = f"assets/caption_{timestamp}.mp3"
        
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(audio_path)
        return audio_path
    except Exception:
        return None