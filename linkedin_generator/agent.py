import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import zipfile
import os
import jinja2
import threading
import re
import requests
import json

class CSVAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Content Generator with Ollama")
        self.root.geometry("1000x800")

        self.df = None
        self.filename = None
        self.stop_generation_flag = False

        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3"
        self.is_ollama_connected = False

        # Initialize all UI widgets before checking connection
        self.status_label = None
        self.model_var = None
        self.model_entry = None
        self.refresh_btn = None
        self.console_btn = None
        self.load_btn = None
        self.save_btn = None
        self.choose_file_btn = None
        self.cols_listbox = None
        self.prompt_text = None
        self.generate_btn = None
        self.stop_btn = None
        self.pause_btn = None
        self.resume_btn = None
        self.progress = None
        self.progress_info_label = None
        self.details_frame = None
        self.details_visible = None
        self.error_log_visible = None
        self.error_logs = []
        self.progress_label = None
        self.toggle_details_btn = None
        self.toggle_error_log_btn = None
        self.tree = None
        self.retry_btn = None
        self.copy_btn = None

        self.setup_ui()
        self.check_ollama_connection()

    def open_console(self):
        import subprocess
        try:
            subprocess.Popen(["powershell.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open console: {e}")

    def setup_ui(self):
        try:
            self.status_visible = tk.BooleanVar(value=True)
            self.root.configure(bg="#f5f6fa")
            default_font = ("Segoe UI", 11)
        self.setup_ui()
        self.check_ollama_connection()
            # All widget setup code from main.py
            # ...existing code...
        except Exception as e:
            import traceback
            err_msg = f"UI setup failed: {e}\n{traceback.format_exc()}"
            self.error_logs.append(err_msg)
            messagebox.showerror("UI Error", err_msg)
            fallback = tk.Label(self.root, text="Critical UI error. See error log.", fg="red", bg="#f5f6fa", font=("Segoe UI", 14, "bold"))
            fallback.pack(padx=20, pady=20)

    # ...existing code...
    # All other methods from main.py
    # ...existing code...
