import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import whisper
import torch

# ---------------- Tooltip class ----------------
class CreateToolTip:
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)
    def enter(self, event=None):
        self.schedule()
    def leave(self, event=None):
        self.unschedule()
        self.hidetip()
    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)
    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)
    def showtip(self, event=None):
        # Posiziona il tooltip subito a destra del widget
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 5
        y = self.widget.winfo_rooty()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#e0ffe0", relief='solid', borderwidth=1,
                         font=("Helvetica", 8))
        label.pack(ipadx=1)
    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# ---------------- Custom stdout writer for log widget ----------------
class LogWidgetWriter:
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
    def flush(self):
        pass

# ---------------- Utility function ----------------
def seconds_to_lrc_timestamp(seconds):
    """
    Converts a time in seconds (float) into an LRC timestamp format [MM:SS.xx].
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"[{minutes:02}:{secs:02}.{centiseconds:02}]"

# ---------------- Transcription function ----------------
def transcribe_audio_to_lrc(audio_file, output_lrc, model_size, language, device,
                            beam_size, best_of, temperature, length_penalty,
                            suppress_tokens, initial_prompt, condition_on_previous_text,
                            log_widget):
    try:
        log_widget.insert(tk.END, "Loading Whisper model...\n")
        log_widget.see(tk.END)
        log_widget.update_idletasks()

        # Redirect stdout to show progress (e.g. model download)
        old_stdout = sys.stdout
        sys.stdout = LogWidgetWriter(log_widget)

        model = whisper.load_model(model_size, device=device)

        # Restore stdout
        sys.stdout = old_stdout

        log_widget.insert(tk.END, "Model loaded. Starting transcription...\n")
        log_widget.see(tk.END)
        log_widget.update_idletasks()

        if suppress_tokens:
            try:
                suppress_tokens_list = [int(tok.strip()) for tok in suppress_tokens.split(',') if tok.strip()]
            except ValueError:
                suppress_tokens_list = []
                log_widget.insert(tk.END, "Warning: Suppress Tokens must be a comma-separated list of integers.\n")
        else:
            suppress_tokens_list = []

        result = model.transcribe(
            audio_file,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            length_penalty=length_penalty,
            suppress_tokens=suppress_tokens_list,
            initial_prompt=initial_prompt if initial_prompt else None,
            condition_on_previous_text=condition_on_previous_text
        )
        segments = result.get("segments", [])
        if not segments:
            messagebox.showerror("Error", "No transcription segments found!")
            return

        lrc_lines = []
        for segment in segments:
            start_time = segment["start"]
            text = segment["text"].strip().replace('\n', ' ')
            timestamp = seconds_to_lrc_timestamp(start_time)
            lrc_lines.append(f"{timestamp} {text}")
        with open(output_lrc, 'w', encoding='utf-8') as f:
            f.write("\n".join(lrc_lines))
        log_widget.insert(tk.END, f"LRC file generated and saved: {output_lrc}\n")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        log_widget.insert(tk.END, f"Error: {str(e)}\n")

# ---------------- Main Application ----------------
class App(ttk.Window):
    def __init__(self):
        super().__init__(themename="minty")  # Using the "minty" theme for a green, Spotify-like look
        self.title("Whisper Transcription Interface")
        self.geometry("900x650")
        
        # Set custom default font
        default_font = ("Helvetica", 11)
        self.option_add("*Font", default_font)
        
        self.input_file = None
        self.output_file = None

        # --- File selection frame ---
        file_frame = ttk.Frame(self)
        file_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        file_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(3, weight=1)

        self.select_input_btn = ttk.Button(file_frame, text="Select Audio File", command=self.select_input_file, bootstyle=PRIMARY, padding=6)
        self.select_input_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_label = ttk.Label(file_frame, text="No file selected", anchor="w")
        self.input_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.select_output_btn = ttk.Button(file_frame, text="Select LRC Output Path", command=self.select_output_file, bootstyle=SUCCESS, padding=6)
        self.select_output_btn.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.output_label = ttk.Label(file_frame, text="No output path selected", anchor="w")
        self.output_label.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # --- Model options frame ---
        model_frame = ttk.Labelframe(self, text="Model Options", padding=(10, 5))
        model_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        # Model label with tooltip
        model_frame.columnconfigure(0, weight=0)
        model_label_frame = ttk.Frame(model_frame)
        model_label_frame.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(model_label_frame, text="Model:").pack(side="left")
        model_tip_icon = ttk.Label(model_label_frame, text="?", foreground="green", cursor="question_arrow")
        model_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(model_tip_icon, "Choose the Whisper model. 'Large' requires significant GPU memory if used with CUDA.")
        self.model_var = ttk.StringVar(value="large")
        model_options = ["tiny", "base", "small", "medium", "large"]
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_options, state="readonly", width=10)
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Language label with tooltip
        lang_label_frame = ttk.Frame(model_frame)
        lang_label_frame.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Label(lang_label_frame, text="Language:").pack(side="left")
        lang_tip_icon = ttk.Label(lang_label_frame, text="?", foreground="green", cursor="question_arrow")
        lang_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(lang_tip_icon, "Set the audio language (e.g., 'en' or 'it').")
        self.lang_var = ttk.StringVar(value="en")
        language_options = ["en", "it", "es", "fr", "de", "ru", "zh", "ja"]
        self.lang_combo = ttk.Combobox(model_frame, textvariable=self.lang_var, values=language_options, state="readonly", width=10)
        self.lang_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Device label with tooltip
        device_label_frame = ttk.Frame(model_frame)
        device_label_frame.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Label(device_label_frame, text="Device:").pack(side="left")
        device_tip_icon = ttk.Label(device_label_frame, text="?", foreground="green", cursor="question_arrow")
        device_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(device_tip_icon, "Choose 'cuda' for GPU (requires sufficient VRAM) or 'cpu' for processor.")
        self.device_var = ttk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.device_combo = ttk.Combobox(model_frame, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=10)
        self.device_combo.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        # --- Advanced transcription options frame ---
        advanced_frame = ttk.Labelframe(self, text="Advanced Transcription Options", padding=(10, 5))
        advanced_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=10)
        # Configure 6 columns for advanced options
        for col in range(6):
            advanced_frame.columnconfigure(col, weight=1)

        # Row 0: Beam Size and Best Of
        beam_frame = ttk.Frame(advanced_frame)
        beam_frame.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(beam_frame, text="Beam Size:").pack(side="left")
        beam_tip_icon = ttk.Label(beam_frame, text="?", foreground="green", cursor="question_arrow")
        beam_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(beam_tip_icon, "Number of beams used during beam search. Higher values can improve quality but slow down processing.")
        self.beam_size_entry = ttk.Entry(beam_frame, width=5)
        self.beam_size_entry.insert(0, "5")
        self.beam_size_entry.pack(side="left", padx=(5,0))

        best_frame = ttk.Frame(advanced_frame)
        best_frame.grid(row=0, column=3, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(best_frame, text="Best Of:").pack(side="left")
        best_tip_icon = ttk.Label(best_frame, text="?", foreground="green", cursor="question_arrow")
        best_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(best_tip_icon, "Number of candidate transcriptions per beam. Higher values may yield better results.")
        self.best_of_entry = ttk.Entry(best_frame, width=5)
        self.best_of_entry.insert(0, "5")
        self.best_of_entry.pack(side="left", padx=(5,0))

        # Row 1: Temperature and Length Penalty
        temp_frame = ttk.Frame(advanced_frame)
        temp_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(temp_frame, text="Temperature:").pack(side="left")
        temp_tip_icon = ttk.Label(temp_frame, text="?", foreground="green", cursor="question_arrow")
        temp_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(temp_tip_icon, "Controls randomness. Lower values yield more deterministic results.")
        self.temperature_entry = ttk.Entry(temp_frame, width=5)
        self.temperature_entry.insert(0, "0.0")
        self.temperature_entry.pack(side="left", padx=(5,0))

        length_frame = ttk.Frame(advanced_frame)
        length_frame.grid(row=1, column=3, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(length_frame, text="Length Penalty:").pack(side="left")
        length_tip_icon = ttk.Label(length_frame, text="?", foreground="green", cursor="question_arrow")
        length_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(length_tip_icon, "Adjusts penalty for longer sequences to control output length.")
        self.length_penalty_entry = ttk.Entry(length_frame, width=5)
        self.length_penalty_entry.insert(0, "1.0")
        self.length_penalty_entry.pack(side="left", padx=(5,0))

        # Row 2: Suppress Tokens
        suppress_frame = ttk.Frame(advanced_frame)
        suppress_frame.grid(row=2, column=0, columnspan=6, sticky="w", padx=5, pady=5)
        ttk.Label(suppress_frame, text="Suppress Tokens (comma-separated):").pack(side="left")
        suppress_tip_icon = ttk.Label(suppress_frame, text="?", foreground="green", cursor="question_arrow")
        suppress_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(suppress_tip_icon, "A comma-separated list of token IDs to suppress during decoding.")
        self.suppress_tokens_entry = ttk.Entry(suppress_frame, width=30)
        self.suppress_tokens_entry.pack(side="left", padx=(5,0))

        # Row 3: Initial Prompt
        prompt_frame = ttk.Frame(advanced_frame)
        prompt_frame.grid(row=3, column=0, columnspan=6, sticky="w", padx=5, pady=5)
        ttk.Label(prompt_frame, text="Initial Prompt:").pack(side="left")
        prompt_tip_icon = ttk.Label(prompt_frame, text="?", foreground="green", cursor="question_arrow")
        prompt_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(prompt_tip_icon, "Optional starting text to condition the transcription process.")
        self.initial_prompt_entry = ttk.Entry(prompt_frame, width=50)
        self.initial_prompt_entry.pack(side="left", padx=(5,0))

        # Row 4: Condition on previous text
        condition_frame = ttk.Frame(advanced_frame)
        condition_frame.grid(row=4, column=0, columnspan=6, sticky="w", padx=5, pady=5)
        self.condition_var = ttk.BooleanVar(value=True)
        self.condition_check = ttk.Checkbutton(condition_frame, text="Condition on previous text", variable=self.condition_var)
        self.condition_check.pack(side="left")
        condition_tip_icon = ttk.Label(condition_frame, text="?", foreground="green", cursor="question_arrow")
        condition_tip_icon.pack(side="left", padx=(2,0))
        CreateToolTip(condition_tip_icon, "If enabled, the model uses previous context for more coherent transcriptions.")

        # --- Start transcription button ---
        self.start_btn = ttk.Button(self, text="Start Transcription", command=self.start_transcription, bootstyle=SUCCESS, padding=8)
        self.start_btn.grid(row=3, column=0, padx=15, pady=15, sticky="ew")

        # --- Log text widget with scrollbar ---
        self.log_text = ttk.ScrolledText(self, height=15)
        self.log_text.grid(row=4, column=0, padx=15, pady=5, sticky="nsew")

        # Configure grid weights for responsiveness
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

    def select_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select the audio file to transcribe",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.input_file = file_path
            self.input_label.config(text=file_path)

    def select_output_file(self):
        file_path = filedialog.asksaveasfilename(
            title="Save LRC file as",
            defaultextension=".lrc",
            filetypes=[("LRC Files", "*.lrc")]
        )
        if file_path:
            self.output_file = file_path
            self.output_label.config(text=file_path)

    def start_transcription(self):
        if not self.input_file:
            messagebox.showerror("Error", "Please select an audio file!")
            return
        if not self.output_file:
            messagebox.showerror("Error", "Please select an output path for the LRC file!")
            return
        
        try:
            beam_size = int(self.beam_size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Beam Size must be an integer!")
            return
        try:
            best_of = int(self.best_of_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Best Of must be an integer!")
            return
        try:
            temperature = float(self.temperature_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Temperature must be a decimal number!")
            return
        try:
            length_penalty = float(self.length_penalty_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Length Penalty must be a decimal number!")
            return
        
        model = self.model_var.get()
        language = self.lang_var.get()
        device = self.device_var.get()
        suppress_tokens = self.suppress_tokens_entry.get()
        initial_prompt = self.initial_prompt_entry.get()
        condition_on_previous_text = self.condition_var.get()
        
        thread = threading.Thread(
            target=transcribe_audio_to_lrc, 
            args=(self.input_file, self.output_file, model, language, device,
                  beam_size, best_of, temperature, length_penalty,
                  suppress_tokens, initial_prompt, condition_on_previous_text, self.log_text)
        )
        thread.start()

if __name__ == '__main__':
    app = App()
    app.mainloop()
