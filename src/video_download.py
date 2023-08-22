##############################################################################################
"""
This script will read the CSV containing the YT links and download only the audio related to 
each video. The vdieos and audios are saved in data/videos/<poi>

python3 src/video_scrapper.py
"""
##############################################################################################

import os
import re
import argparse
import ast
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from yt_dlp import YoutubeDL

load_dotenv()


class YTLogger:
    def debug(self):
        # For compatibility with youtube-dl, both debug and info are passed into debug
        # You can distinguish them by the prefix '[debug] '
        pass

    def info(self):
        pass

    def warning(self):
        pass

    def error(self):
        pass


class VideoScrapper:
    def __init__(self, links_file, audio_dir):
        """Initialize a class to scrape the actual audios

        Args:
            links_file (str): filepath to the txt with URLs
            audio_dir(str): Directory to store the audio files
        """
        self.links_file = links_file
        self.df = None
        self.audio_dir = audio_dir

    def download_videos(self, url):
        """Specify the yt-dlp parameters

        Args:
            url (str): URL to retrieve videl
        """
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "postprocessors": [
                {  # Extract audio using ffmpeg
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
            "outtmpl": f"{self.audio_dir}/%(id)s.%(ext)s",
            "keepvideo": False,
            "logger": YTLogger,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download(url)
        except Exception as e:
            print(e)
            return

    def process_urls(self):
        """Iterate through each name and download videos into each speaker folder"""
        with open(self.links_file, "r") as urls:
            for url in tqdm(urls):
                self.download_videos(url)


if __name__ == "__main__":
    links_file = os.path.join(os.getenv("DATA_FOLDER"), os.getenv("URL_LINKS_FILE"))
    clsObj = VideoScrapper(
        links_file,
        audio_dir=os.path.join(os.getenv("DATA_FOLDER"), os.getenv("AUDIO_FOLDER")),
    )
    clsObj.process_urls()
