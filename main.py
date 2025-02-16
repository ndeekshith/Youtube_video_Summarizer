import speech_recognition as sr
import yt_dlp
import os
from typing import Optional, Tuple
import logging
import time
from pydub import AudioSegment
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging as transformers_logging
import torch

class TranscriptionError(Exception):
    pass

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    transformers_logging.set_verbosity_error()

def download_audio(youtube_url: str) -> Tuple[str, str]:
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': True,
            'ffmpeg_location': 'C:\\Project3\\ffmpeg\\bin',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info("Downloading audio...")
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'Unknown Title')
            video_id = info.get('id', 'unknown')
            audio_file = f"{video_id}.wav"

            if not os.path.exists(audio_file):
                raise TranscriptionError("Download failed - output file not found")

            return audio_file, video_title

    except Exception as e:
        raise TranscriptionError(f"Failed to download audio: {str(e)}")

def split_audio(audio_file: str, chunk_duration: int = 60) -> list[str]:
    sound = AudioSegment.from_file(audio_file)
    chunk_length_ms = chunk_duration * 1000
    chunks = []
    for i in range(0, len(sound), chunk_length_ms):
        chunk = sound[i:i + chunk_length_ms]
        chunk_file = f"{os.path.splitext(audio_file)[0]}_chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

def transcribe_audio_google(audio_file: str, retries: int = 3, backoff_factor: int = 2, max_backoff_time: int = 60) -> str:
    recognizer = sr.Recognizer()
    full_text = ""
    chunks = split_audio(audio_file)
    problematic_chunks = []

    for chunk_file in chunks:
        for attempt in range(retries):
            try:
                with sr.AudioFile(chunk_file) as source:
                    logging.info(f"Processing chunk: {chunk_file} (attempt {attempt + 1})...")
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    full_text += text + " "
                    break

            except sr.UnknownValueError:
                if attempt == retries - 1:
                    logging.warning(f"Could not understand audio: {chunk_file}")
                    problematic_chunks.append(chunk_file)

            except sr.RequestError as e:
                if attempt < retries - 1:
                    backoff_time = min(backoff_factor ** attempt, max_backoff_time)
                    logging.warning(f"Request error: {e}. Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    raise TranscriptionError(f"Speech Recognition service failed: {e}")

            except Exception as e:
                raise TranscriptionError(f"Transcription failed: {e}")

        if chunk_file not in problematic_chunks:
            try:
                os.remove(chunk_file)
            except Exception as e:
                logging.error(f"Failed to clean up {chunk_file}: {str(e)}")

    if problematic_chunks:
        logging.warning(f"Skipped {len(problematic_chunks)} problematic chunks.")
    return full_text.strip()

def load_summarizer(model_name="microsoft/phi-2", save_directory="phi2_model"):
    try:
        model = AutoModelForCausalLM.from_pretrained(save_directory, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        logging.info(f"Model loaded from {save_directory}")
        
        summarizer = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="cuda"
        )
        return summarizer
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def chunk_text(text: str, max_chunk_size: int = 1024) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # Add 1 for space
        if current_size + word_size > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(summarizer, text: str, max_length: int = 150, min_length: int = 30) -> str:
    prompt = f"""Instruction: Provide a concise summary of the following text.

Text: {text}

Summary:"""

    try:
        result = summarizer(
            prompt,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=summarizer.tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        summary_start = generated_text.rfind("Summary:")
        
        if summary_start != -1:
            summary = generated_text[summary_start + 8:].strip()
            logging.info(f"Generated summary length: {len(summary)}")
            logging.info(f"Summary preview: {summary[:100]}...")
            return summary
        else:
            logging.warning("Summary marker not found in output")
            return generated_text.strip()
            
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        raise

def summarize_long_text(summarizer, text: str, max_chunk_size: int = 1024) -> str:
    chunks = chunk_text(text, max_chunk_size)
    summaries = []
    
    for i, chunk in enumerate(chunks):
        logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_text(summarizer, chunk)
        summaries.append(summary)
    
    return " ".join(summaries)

def youtube_audio_to_text(youtube_url: str) -> Optional[Tuple[str, str]]:
    setup_logging()
    audio_file = None

    try:
        audio_file, video_title = download_audio(youtube_url)
        transcription = transcribe_audio_google(audio_file)
        return transcription, video_title

    except TranscriptionError as e:
        logging.error(str(e))
        return None
    finally:
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception as e:
                logging.error(f"Failed to clean up file: {str(e)}")

def save_transcription_and_summary(transcription: str, summary: str, video_title: str) -> Tuple[str, str]:
    safe_title = "".join(c for c in video_title[:30] if c.isalnum() or c in (' ', '-', '_'))
    transcription_filename = f"transcription_{safe_title}.txt"
    summary_filename = f"summary_{safe_title}.txt"

    with open(transcription_filename, 'w', encoding='utf-8') as f:
        f.write(transcription)
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(summary)

    return transcription_filename, summary_filename

def main():
    try:
        youtube_url = input("Enter YouTube URL: ")
        print("\nProcessing video...")

        result = youtube_audio_to_text(youtube_url)
        if not result:
            print("Transcription failed. Check logs for details.")
            return

        transcription, video_title = result
        print(f"\nTranscription for: {video_title}")
        print("-" * 50)
        print(transcription)

        summarizer = load_summarizer()
        summary = summarize_long_text(summarizer, transcription)

        print(f"\nSummary for: {video_title}")
        print("-" * 50)
        print(summary)

        trans_file, sum_file = save_transcription_and_summary(transcription, summary, video_title)
        print(f"\nFiles saved:\n{trans_file}\n{sum_file}")

    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        print("An error occurred. Check logs for details.")

if __name__ == "__main__":
    main()