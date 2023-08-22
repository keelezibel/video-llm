import os
import whisper
import requests
import gradio as gr
from dotenv import load_dotenv
from pytube import YouTube
from yt_dlp import YoutubeDL

load_dotenv()

class GradioInference():
    def __init__(self):
        self.sizes = list(whisper._MODELS.keys())
        self.langs = ["none"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.current_size = "small.en"
        self.loaded_model = whisper.load_model(self.current_size)
        self.yt = None
    
    def download_videos(self, link):
        """Specify the yt-dlp parameters
        Args:
            url (str): URL to retrieve videl
            name (str): speaker name
        """
        ydl_opts = {
            # "format": "bestvideo[ext=mp4]",
            "format": "m4a/bestaudio/best",
            "postprocessors": [
                {  # Extract audio using ffmpeg
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": [
                '-ar', '16000'
            ],
            "outtmpl": f"{os.getenv('TMP_FOLDER')}/tmp.%(ext)s",
            # "keepvideo": True
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download(link)

        return f"{os.getenv('TMP_FOLDER')}/tmp.wav"

    def detect_lang(self, path):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(path)
        audio_segment = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio_segment).to(self.loaded_model.device)

        # detect the spoken language
        _, probs = self.loaded_model.detect_language(mel)
        language = max(probs, key=probs.get)

        return language

    def summarize(self, query: dict):
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"{os.getenv("HF_AUTHORIZATION")}"}

        response = requests.post(API_URL, headers=headers, json=query)
        return response.json()

    def __call__(self, link, lang, size, path=''):
        if self.yt is None and link != '':
            self.yt = YouTube(link)
        if not path:
            path = self.download_videos(link)

        if size != self.current_size:
            self.loaded_model = whisper.load_model(size)
            self.current_size = size
        
        if lang == "none":
            lang = self.detect_lang(path)

        options = whisper.DecodingOptions().__dict__.copy()
        options["language"] = lang
        options["beam_size"] = 5
        options["best_of"] = 5
        del options["task"]
        transcribe_options = dict(task="transcribe", **options)
        # translate_options = dict(task="translate", **options)
        # translation_txt = self.loaded_model.transcribe(path, **translate_options)["text"]
        transcription_txt = self.loaded_model.transcribe(path, **transcribe_options)["text"]

        summary = self.summarize({
            "inputs": f"{transcription_txt}",
            "wait_for_model": True
        })[0]["summary_text"]


        return transcription_txt, summary
   
    def populate_metadata(self, link):
        self.yt = YouTube(link)
        return self.yt.thumbnail_url, self.yt.title

gio = GradioInference()
examples_folder = os.getenv("EXAMPLES_FOLDER")

with gr.Blocks() as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 500px; margin: 0 auto;">
              <div>
                <h1>Video LLM</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Speech to text transcription of Youtube videos using OpenAI's Whisper
              </p>
            </div>
        """
    )

    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                sz = gr.Dropdown(label="Model Size", choices=gio.sizes, value='small.en')
                lang = gr.Dropdown(label="Language (Optional)", choices=gio.langs, value="english")
            link = gr.Textbox(label="YouTube Link", placeholder="")
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    title = gr.Textbox(label="Title")
                    image = gr.Image(label="Thumbnail")
                translate = gr.Textbox(label="Transcription", placeholder="Transcription Output", lines=15, show_copy_button=True)
                summary = gr.Textbox(label="Summary (facebook/bart-large-cnn)", placeholder="Summary", lines=5, show_copy_button=True)
            with gr.Row().style(equal_height=True): 
                btn = gr.Button("Generate")       
        
        btn.click(gio, inputs=[link, lang, sz], outputs=[translate, summary])
        link.change(gio.populate_metadata, inputs=[link], outputs=[image, title])
    
demo.launch()