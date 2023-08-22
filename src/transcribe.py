import os
import re
import time
import torch
import ffmpeg
import srt
import shutil
import numpy as np
from numpy.typing import NDArray
import datetime
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from dotenv import load_dotenv

load_dotenv()


class Transcribe:
    def __init__(self, use_gpu=False) -> None:
        # Configuration
        assert int(os.getenv("MAX_ATTEMPTS")) >= 1
        assert float(os.getenv("VAD_THRESHOLD")) >= 0.01
        assert float(os.getenv("CHUNK_THRESHOLD")) >= 0.1

        self.max_attempts = int(os.getenv("MAX_ATTEMPTS"))
        self.vad_threshold = float(os.getenv("VAD_THRESHOLD"))
        self.chunk_threshold = float(os.getenv("CHUNK_THRESHOLD"))

        self.use_gpu = use_gpu

        if os.getenv("TRANSLATION_MODE") == "Transcribe & Translate":
            self.task = "translate"
        elif os.getenv("TRANSLATION_MODE") == "Transcribe Only":
            self.task = "transcribe"
        else:
            raise ValueError("Invalid translation mode")

        self.source_separation = os.getenv("SOURCE_SEPARATION")
        self.data_folder = os.getenv("DATA_FOLDER")
        self.audio_files_folder = os.path.join(
            self.data_folder, os.getenv("AUDIO_FOLDER")
        )
        self.output_folder = os.path.join(self.data_folder, os.getenv("OUTPUT_FOLDER"))
        self.processed_folder = os.path.join(
            self.data_folder, os.getenv("PROCESSED_FOLDER")
        )
        self.init_vad_model()

        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE")
        self.init_whisper_model()

    def init_vad_model(self):
        print("Initializing VAD Model")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
        )
        print("VAD Model loaded")

    def init_whisper_model(self):
        if self.use_gpu:
            import whisper

            print("Initializing Whisper Model")
            self.whisper_model = whisper.load_model(
                f"/models/whisper/{self.whisper_model_size}.pt"
            )
            print("Whisper Model loaded with GPU")
        else:
            from whispercpp import Whisper

            def to_time(n):
                return time.strftime("%H:%M:%S", time.gmtime(int(n / 100)))

            def transcribe_with_time(
                self, data: NDArray[np.float32], num_proc: int = 1, strict: bool = False
            ):
                if strict:
                    assert (
                        self.context.is_initialized
                    ), "strict=True and context is not initialized. Make sure to call 'context.init_state()' before."
                else:
                    if (
                        not self.context.is_initialized
                        and not self._context_initialized
                    ):
                        self.context.init_state()
                        self._context_initialized = True

                self.context.full_parallel(self.params, data, num_proc)
                return [
                    (
                        self.context.full_get_segment_start(i),
                        self.context.full_get_segment_end(i),
                        # to_time(self.context.full_get_segment_start(i)),
                        # to_time(self.context.full_get_segment_end(i)),
                        self.context.full_get_segment_text(i),
                    )
                    for i in range(self.context.full_n_segments())
                ]

            Whisper.transcribe_with_time = transcribe_with_time

            self.whisper_model = Whisper.from_pretrained(
                os.getenv("WHISPERCPP_MODEL_SIZE")
            )
            self.whisper_model.params.with_token_timestamps(True)

            print("Whisper Model loaded without GPU")

    def source_separate(self, audio_path):
        if self.source_separation:
            print("Separating vocals...")
            input_len_path = f"{os.getenv('DATA_FOLDER')}/input_length"
            os.system(
                f"ffprobe -i '{audio_path}' -show_entries format=duration -v quiet -of csv='p=0' > '{input_len_path}'"
            )
            with open(input_len_path) as f:
                input_length = int(float(f.read())) + 1

            if input_length > float(
                os.getenv("MAX_AUDIO_LEN_SEPARATION")
            ):  # Library supports up to 10mins of audio
                proc_audio_path = audio_path
            else:
                os.system(
                    f"spleeter separate -d {input_length} -p spleeter:2stems -o '{self.data_folder}/outputs' '{audio_path}'"
                )
                spleeter_dir = os.path.basename(os.path.splitext(audio_path)[0])
                proc_audio_path = (
                    os.path.join(self.output_folder, spleeter_dir)
                    + os.path.sep
                    + "vocals.wav"
                )
            return proc_audio_path

    def encode_audio(self, audio_path):
        print("Encoding audio...")
        vad_chunks_path = f"{self.data_folder}/vad_chunks"
        if not os.path.exists(vad_chunks_path):
            os.mkdir(vad_chunks_path)
        try:
            ffmpeg.input(audio_path).output(
                f"{vad_chunks_path}/silero_temp.wav",
                ar="16000",
                ac="1",
                acodec="pcm_s16le",
                map_metadata="-1",
                fflags="+bitexact",
            ).overwrite_output().run(
                quiet=True, capture_stdout=True, capture_stderr=True
            )
        except ffmpeg.Error as e:
            print("stdout:", e.stdout.decode("utf8"))
            print("stderr:", e.stderr.decode("utf8"))
            raise e

    def run_whisper_cpu(self, out_path, num_proc: int = 1, strict: bool = False):
        print("Running VAD...")

        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = self.vad_utils

        # Generate VAD timestamps
        VAD_SR = int(os.getenv("SAMPLING_FREQUENCY"))
        vad_chunks_path = f"{self.data_folder}/vad_chunks"
        wav = read_audio(f"{vad_chunks_path}/silero_temp.wav", sampling_rate=VAD_SR)
        t = get_speech_timestamps(
            wav, self.vad_model, sampling_rate=VAD_SR, threshold=self.vad_threshold
        )

        # Add a bit of padding, and remove small gaps
        for i in range(len(t)):
            t[i]["start"] = max(0, t[i]["start"] - 3200)  # 0.2s head
            t[i]["end"] = min(wav.shape[0] - 16, t[i]["end"] + 20800)  # 1.3s tail
            if i > 0 and t[i]["start"] < t[i - 1]["end"]:
                t[i]["start"] = t[i - 1]["end"]  # Remove overlap

        # If breaks are longer than chunk_threshold seconds, split into a new audio file
        # This'll effectively turn long transcriptions into many shorter ones
        u = [[]]
        for i in range(len(t)):
            if i > 0 and t[i]["start"] > t[i - 1]["end"] + (
                self.chunk_threshold * VAD_SR
            ):
                u.append([])
            u[-1].append(t[i])

        # Merge speech chunks
        for i in range(len(u)):
            save_audio(
                f"{vad_chunks_path}/" + str(i) + ".wav",
                collect_chunks(u[i], wav),
                sampling_rate=VAD_SR,
            )
        os.remove(f"{vad_chunks_path}/silero_temp.wav")

        # Convert timestamps to seconds
        for i in range(len(u)):
            time = 0.0
            offset = 0.0
            for j in range(len(u[i])):
                u[i][j]["start"] /= VAD_SR
                u[i][j]["end"] /= VAD_SR
                u[i][j]["chunk_start"] = time
                time += u[i][j]["end"] - u[i][j]["start"]
                u[i][j]["chunk_end"] = time
                if j == 0:
                    offset += u[i][j]["start"]
                else:
                    offset += u[i][j]["start"] - u[i][j - 1]["end"]
                u[i][j]["offset"] = offset

        # Run Whisper on each audio chunk
        print("Running Whisper...")
        subs = []
        sub_index = 1
        for i in range(len(u)):
            for x in range(self.max_attempts):
                try:
                    y, _ = (
                        ffmpeg.input(f"{vad_chunks_path}/" + str(i) + ".wav", threads=0)
                        .output(
                            "-",
                            format="s16le",
                            acodec="pcm_s16le",
                            ac=1,
                            ar=os.getenv("SAMPLING_FREQUENCY"),
                        )
                        .run(
                            cmd=["ffmpeg", "-nostdin"],
                            capture_stdout=True,
                            capture_stderr=True,
                        )
                    )
                except ffmpeg.Error as e:
                    raise RuntimeError(
                        f"Failed to load audio: {e.stderr.decode()}"
                    ) from e

                arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0
                result = self.whisper_model.transcribe_with_time(arr)

                if x + 1 < self.max_attempts:
                    print("Retrying chunk", i)
            for r in result:
                # Add to SRT list
                subs.append(
                    srt.Subtitle(
                        index=sub_index,
                        start=datetime.timedelta(seconds=r[0]),
                        end=datetime.timedelta(seconds=r[1]),
                        content=r[2].strip(),
                    )
                )
                sub_index += 1

        # Write file
        # Removal of garbage lines
        garbage_list = [
            "a",
            "aa",
            "ah",
            "ahh",
            "ha",
            "haa",
            "hah",
            "haha",
            "hahaha",
            "mmm",
            "mm",
            "m",
            "h",
            "o",
            "mh",
            "mmh",
            "hm",
            "hmm",
            "huh",
            "oh",
        ]
        need_context_lines = [
            "feelsgod",
            "godbye",
            "godnight",
            "thankyou",
        ]
        clean_subs = list()
        last_line_garbage = False
        for i in range(len(subs)):
            c = subs[i].content

            c = (
                c.replace(".", "")
                .replace(",", "")
                .replace(":", "")
                .replace(";", "")
                .replace("!", "")
                .replace("?", "")
                .replace("-", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .lower()
                .replace("that feels", "feels")
                .replace("it feels", "feels")
                .replace("feels good", "feelsgood")
                .replace("good bye", "goodbye")
                .replace("good night", "goodnight")
                .replace("thank you", "thankyou")
                .replace("aaaaaa", "a")
                .replace("aaaa", "a")
                .replace("aa", "a")
                .replace("aa", "a")
                .replace("mmmmmm", "m")
                .replace("mmmm", "m")
                .replace("mm", "m")
                .replace("mm", "m")
                .replace("hhhhhh", "h")
                .replace("hhhh", "h")
                .replace("hh", "h")
                .replace("hh", "h")
                .replace("oooooo", "o")
                .replace("oooo", "o")
                .replace("oo", "o")
                .replace("oo", "o")
                .replace("[BLANK_AUDIO]", "")
            )
            is_garbage = True
            for w in c.split(" "):
                if w.strip() == "":
                    continue
                if w.strip() in garbage_list:
                    continue
                elif w.strip() in need_context_lines and last_line_garbage:
                    continue
                else:
                    is_garbage = False
                    break
            if not is_garbage:
                clean_subs.append(subs[i])
            last_line_garbage = is_garbage
        with open(out_path, "w", encoding="utf8") as f:
            f.write(srt.compose(clean_subs))
        print("\nDone! Subs written to", out_path)

    def run_whisper(self, out_path):
        print("Running VAD...")

        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = self.vad_utils

        # Generate VAD timestamps
        VAD_SR = int(os.getenv("SAMPLING_FREQUENCY"))
        vad_chunks_path = f"{self.data_folder}/vad_chunks"
        wav = read_audio(f"{vad_chunks_path}/silero_temp.wav", sampling_rate=VAD_SR)
        t = get_speech_timestamps(
            wav, self.vad_model, sampling_rate=VAD_SR, threshold=self.vad_threshold
        )

        # Add a bit of padding, and remove small gaps
        for i in range(len(t)):
            t[i]["start"] = max(0, t[i]["start"] - 3200)  # 0.2s head
            t[i]["end"] = min(wav.shape[0] - 16, t[i]["end"] + 20800)  # 1.3s tail
            if i > 0 and t[i]["start"] < t[i - 1]["end"]:
                t[i]["start"] = t[i - 1]["end"]  # Remove overlap

        # If breaks are longer than chunk_threshold seconds, split into a new audio file
        # This'll effectively turn long transcriptions into many shorter ones
        u = [[]]
        for i in range(len(t)):
            if i > 0 and t[i]["start"] > t[i - 1]["end"] + (
                self.chunk_threshold * VAD_SR
            ):
                u.append([])
            u[-1].append(t[i])

        # Merge speech chunks
        for i in range(len(u)):
            save_audio(
                f"{vad_chunks_path}/" + str(i) + ".wav",
                collect_chunks(u[i], wav),
                sampling_rate=VAD_SR,
            )
        os.remove(f"{vad_chunks_path}/silero_temp.wav")

        # Convert timestamps to seconds
        for i in range(len(u)):
            time = 0.0
            offset = 0.0
            for j in range(len(u[i])):
                u[i][j]["start"] /= VAD_SR
                u[i][j]["end"] /= VAD_SR
                u[i][j]["chunk_start"] = time
                time += u[i][j]["end"] - u[i][j]["start"]
                u[i][j]["chunk_end"] = time
                if j == 0:
                    offset += u[i][j]["start"]
                else:
                    offset += u[i][j]["start"] - u[i][j - 1]["end"]
                u[i][j]["offset"] = offset

        # Run Whisper on each audio chunk
        print("Running Whisper...")
        subs = []
        sub_index = 1
        suppress_low = [
            "Thank you",
            "Thanks for",
            "ike and ",
            "Bye.",
            "Bye!",
            "Bye bye!",
            "lease sub",
            "The end.",
            "視聴",
        ]
        suppress_high = [
            "ubscribe",
            "my channel",
            "the channel",
            "our channel",
            "ollow me on",
            "for watching",
            "hank you for watching",
            "for your viewing",
            "r viewing",
            "Amara",
            "next video",
            "full video",
            "ranslation by",
            "ranslated by",
            "ee you next week",
            "ご視聴",
            "視聴ありがとうございました",
        ]
        for i in range(len(u)):
            for x in range(self.max_attempts):
                result = self.whisper_model.transcribe(
                    f"{vad_chunks_path}/" + str(i) + ".wav",
                    task=self.task,
                )
                # Break if result doesn't end with severe hallucinations
                if len(result["segments"]) == 0:
                    break
                elif result["segments"][-1]["end"] < u[i][-1]["chunk_end"] + 10.0:
                    break
                elif x + 1 < self.max_attempts:
                    print("Retrying chunk", i)
            for r in result["segments"]:
                # Skip audio timestamped after the chunk has ended
                if r["start"] > u[i][-1]["chunk_end"]:
                    continue
                # Reduce log probability for certain words/phrases
                for s in suppress_low:
                    if s in r["text"]:
                        r["avg_logprob"] -= 0.15
                for s in suppress_high:
                    if s in r["text"]:
                        r["avg_logprob"] -= 0.35
                # Keep segment info for debugging
                del r["tokens"]
                # Skip if log prob is low or no speech prob is high
                if r["avg_logprob"] < -1.0 or r["no_speech_prob"] > 0.85:
                    continue
                # Set start timestamp
                start = r["start"] + u[i][0]["offset"]
                for j in range(len(u[i])):
                    if (
                        r["start"] >= u[i][j]["chunk_start"]
                        and r["start"] <= u[i][j]["chunk_end"]
                    ):
                        start = r["start"] + u[i][j]["offset"]
                        break
                # Prevent overlapping subs
                if len(subs) > 0:
                    last_end = datetime.timedelta.total_seconds(subs[-1].end)
                    if last_end > start:
                        subs[-1].end = datetime.timedelta(seconds=start)
                # Set end timestamp
                end = u[i][-1]["end"] + 0.5
                for j in range(len(u[i])):
                    if (
                        r["end"] >= u[i][j]["chunk_start"]
                        and r["end"] <= u[i][j]["chunk_end"]
                    ):
                        end = r["end"] + u[i][j]["offset"]
                        break
                # Add to SRT list
                subs.append(
                    srt.Subtitle(
                        index=sub_index,
                        start=datetime.timedelta(seconds=start),
                        end=datetime.timedelta(seconds=end),
                        content=r["text"].strip(),
                    )
                )
                sub_index += 1

        # Write SRT file
        # Removal of garbage lines
        garbage_list = [
            "a",
            "aa",
            "ah",
            "ahh",
            "ha",
            "haa",
            "hah",
            "haha",
            "hahaha",
            "mmm",
            "mm",
            "m",
            "h",
            "o",
            "mh",
            "mmh",
            "hm",
            "hmm",
            "huh",
            "oh",
        ]
        need_context_lines = [
            "feelsgod",
            "godbye",
            "godnight",
            "thankyou",
        ]
        clean_subs = list()
        last_line_garbage = False
        for i in range(len(subs)):
            c = subs[i].content
            c = (
                c.replace(".", "")
                .replace(",", "")
                .replace(":", "")
                .replace(";", "")
                .replace("!", "")
                .replace("?", "")
                .replace("-", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .lower()
                .replace("that feels", "feels")
                .replace("it feels", "feels")
                .replace("feels good", "feelsgood")
                .replace("good bye", "goodbye")
                .replace("good night", "goodnight")
                .replace("thank you", "thankyou")
                .replace("aaaaaa", "a")
                .replace("aaaa", "a")
                .replace("aa", "a")
                .replace("aa", "a")
                .replace("mmmmmm", "m")
                .replace("mmmm", "m")
                .replace("mm", "m")
                .replace("mm", "m")
                .replace("hhhhhh", "h")
                .replace("hhhh", "h")
                .replace("hh", "h")
                .replace("hh", "h")
                .replace("oooooo", "o")
                .replace("oooo", "o")
                .replace("oo", "o")
                .replace("oo", "o")
                .replace("[BLANK_AUDIO]", "")
            )
            is_garbage = True
            for w in c.split(" "):
                if w.strip() == "":
                    continue
                if w.strip() in garbage_list:
                    continue
                elif w.strip() in need_context_lines and last_line_garbage:
                    continue
                else:
                    is_garbage = False
                    break
            if not is_garbage:
                clean_subs.append(subs[i])
            last_line_garbage = is_garbage
        with open(out_path, "w", encoding="utf8") as f:
            f.write(srt.compose(clean_subs))
        print("\nDone! Subs written to", out_path)

    def convert_srt_txt(self, srt_file, output_txt_file):
        # read file line by line
        file = open(srt_file, "r")
        lines = file.readlines()
        file.close()

        text = ""
        for line in lines:
            if (
                re.search("^[0-9]+$", line) is None
                and re.search("^[0-9]{2}:[0-9]{2}:[0-9]{2}", line) is None
                and re.search("^$", line) is None
            ):
                text += " " + line.rstrip("\n")
            text = text.lstrip()

        with open(output_txt_file, "w") as f:
            f.write(text)

    def iterate_files(self):
        wav_files_path = self.audio_files_folder + os.path.sep + "*.wav"
        wav_files = glob(wav_files_path)

        for audio_path in tqdm(wav_files):
            filename = os.path.basename(audio_path).split(".")[0]
            out_path_folder = os.path.join(self.output_folder, f"{filename}")
            
            # Create Folder if not exist
            os.makedirs(out_path_folder, exist_ok=True)

            if self.task == "transcribe":
                out_path = os.path.join(out_path_folder, f"{filename}_orig.srt")
                output_txt_file = os.path.join(
                    self.output_folder, f"{filename}/{filename}_orig.txt"
                )
            elif self.task == "translate":
                out_path = os.path.join(out_path_folder, f"{filename}_trans.srt")
                output_txt_file = os.path.join(
                    self.output_folder, f"{filename}/{filename}_trans.txt"
                )
            proc_audio_path = self.source_separate(audio_path)
            self.encode_audio(proc_audio_path)
            if self.use_gpu:
                out_path = os.path.join(out_path_folder, f"{filename}.srt")
                self.run_whisper(out_path)
                self.convert_srt_txt(out_path, output_txt_file)
            else:
                out_path = os.path.join(out_path_folder, f"{filename}.srt")
                self.run_whisper_cpu(out_path)
                self.convert_srt_txt(out_path, output_txt_file)

            # Move file to processed
            shutil.move(
                audio_path, os.path.join(self.processed_folder, f"{filename}.wav")
            )

            # Clean up vad_chunks folder
            vad_chunks_dir = os.path.join(self.data_folder, "vad_chunks")
            for f in os.listdir(vad_chunks_dir):
                if f != ".keep":
                    os.remove(os.path.join(vad_chunks_dir, f))


if __name__ == "__main__":
    if torch.cuda.is_available():
        transcribe_obj = Transcribe(use_gpu=True)
    else:
        transcribe_obj = Transcribe(use_gpu=False)
    transcribe_obj.iterate_files()
