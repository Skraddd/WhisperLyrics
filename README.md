# WhisperLyrics
A simple GUI for openai-whisper that let's you generate song's lyrics and more 

Tested only on python 3.11.11 on Arch / EndeavourOS / Windows

This project provides a graphical interface for the [OpenAI Whisper](https://github.com/openai/whisper) transcription system. The interface is built using [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap).

## Features

- **Advanced Options:** Adjust beam size, best-of candidates, temperature, length penalty, and more.
- **Tooltips:** Hover over the green question marks (placed immediately to the right of each option label) to see detailed explanations.
- **Model Download Progress:** The log window shows progress messages during model download (if the model is being downloaded for the first time).
- **GPU/CPU Selection:** Easily switch between using CUDA (GPU) or CPU for transcription.

## Requirements

- **Python 3.11** (Make sure your Python 3.11 installation includes Tkinter.)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyTorch](https://pytorch.org/) (with CUDA support if you plan to use GPU)
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap)
- Tkinter (included with most official Python installations)
- For GPU support, a CUDA-capable NVIDIA GPU, updated drivers, and the CUDA Toolkit must be installed.

---

## Installation Instructions

For GPU (CUDA) Support (Ensure your GPU drivers and CUDA Toolkit are installed.):

pip install openai-whisper torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 ttkbootstrap

For CPU-only:

pip install openai-whisper torch ttkbootstrap

Run the Program:

python whisper-gui.py
