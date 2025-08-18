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
from bs4 import BeautifulSoup  # Add this import for web scraping


class CSVAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Content Generator with Ollama")
        self.root.geometry("1000x800")

        self.df = None
        self.filename = None
        self.stop_generation_flag = False
        self.is_paused = False

        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3"
        self.is_ollama_connected = False

        # UI widgets
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
        self.download_btn = None

        self.setup_ui()
        self.check_ollama_connection()

    def open_console(self):
        """Open a system console window for installing LLM models."""
        import subprocess
        try:
            # Open PowerShell on Windows
            subprocess.Popen(["powershell.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open console: {e}")

    def setup_ui(self):
        try:
            self.status_visible = tk.BooleanVar(value=True)
            self.root.configure(bg="#f5f6fa")
            default_font = ("Segoe UI", 11)
            self.root.option_add("*Font", default_font)

            # --- Top controls frame ---
            top_frame = tk.Frame(self.root, bg="#f5f6fa")
            top_frame.pack(padx=10, pady=8, fill="x")

            self.console_btn = tk.Button(top_frame, text="🖥️ Console", command=self.open_console, bg="#dff9fb", fg="#30336b", font=("Segoe UI", 10, "bold"))
            self.console_btn.pack(side="left", padx=5)

            self.load_btn = tk.Button(top_frame, text="📂 Load CSV/ZIP", command=self.load_file, bg="#7ed6df", fg="#222f3e", font=("Segoe UI", 10, "bold"))
            self.load_btn.pack(side="left", padx=5)

            self.save_btn = tk.Button(top_frame, text="💾 Save", command=self.save_csv, state="disabled", bg="#f6e58d", fg="#222f3e", font=("Segoe UI", 10, "bold"))
            self.save_btn.pack(side="left", padx=5)

            self.download_btn = tk.Button(top_frame, text="⬇️ Download Results", command=self.download_results, state="disabled", bg="#badc58", fg="#222f3e", font=("Segoe UI", 10, "bold"))
            self.download_btn.pack(side="left", padx=5)

            # Move Generate and Stop buttons to top_frame
            self.generate_btn = tk.Button(top_frame, text="⚙️ Generate", command=self.start_generation,
                                          state="disabled", bg="#7ed6df", fg="#222f3e", font=("Segoe UI", 10, "bold"))
            self.generate_btn.pack(side="left", padx=5)

            self.stop_btn = tk.Button(top_frame, text="⏹️ Stop", command=self.stop_generation, state="disabled", bg="#f6e58d", fg="#222f3e", font=("Segoe UI", 10, "bold"))
            self.stop_btn.pack(side="left", padx=5)

            # Add Chat Test Button
            self.chat_test_btn = tk.Button(top_frame, text="💬 Chat Test", command=self.open_chat_test, bg="#badc58", fg="#222f3e", font=("Segoe UI", 10, "bold"))
            self.chat_test_btn.pack(side="left", padx=5)

            # --- Model & status frame ---
            model_frame = tk.Frame(self.root, bg="#f5f6fa")
            model_frame.pack(padx=10, pady=4, fill="x")

            tk.Label(model_frame, text="Ollama Status:", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 10, "bold")).pack(side="left")
            self.status_label = tk.Label(model_frame, text="⚪ Checking...", fg="orange", bg="#f5f6fa", font=("Segoe UI", 10, "bold"))
            self.status_label.pack(side="left", padx=5)

            self.model_var = tk.StringVar(value="llama3")
            tk.Label(model_frame, text="Model:", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(20, 5))
            self.model_entry = tk.Entry(model_frame, textvariable=self.model_var, width=15, font=("Segoe UI", 10))
            self.model_entry.pack(side="left")

            self.refresh_btn = tk.Button(model_frame, text="🔄", command=self.check_ollama_connection, width=3, bg="#dff9fb", fg="#30336b", font=("Segoe UI", 10, "bold"))
            self.refresh_btn.pack(side="left", padx=5)

            # --- Columns & prompt frame ---
            cols_prompt_frame = tk.Frame(self.root, bg="#f5f6fa")
            cols_prompt_frame.pack(padx=10, pady=4, fill="x")

            cols_frame = tk.LabelFrame(cols_prompt_frame, text="🔧 Columns as Variables", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 11, "bold"))
            cols_frame.pack(side="left", fill="y", padx=2, pady=2)

            cols_inner = tk.Frame(cols_frame, bg="#f5f6fa")
            cols_inner.pack(fill="both", expand=True, padx=5, pady=5)

            self.cols_listbox = tk.Listbox(cols_inner, selectmode="multiple", exportselection=0, height=6, font=("Segoe UI", 10), bg="#dff9fb", fg="#30336b")
            self.cols_listbox.pack(side="left", fill="both", expand=True)

            cols_scrollbar = ttk.Scrollbar(cols_inner, orient="vertical", command=self.cols_listbox.yview)
            cols_scrollbar.pack(side="right", fill="y")
            self.cols_listbox.configure(yscrollcommand=cols_scrollbar.set)

            self.cols_listbox.bind("<Double-Button-1>", self.insert_placeholder_from_listbox)

            prompt_frame = tk.LabelFrame(cols_prompt_frame, text="📝 Prompt Template", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 11, "bold"))
            prompt_frame.pack(side="left", fill="both", expand=True, padx=2, pady=2)

            self.prompt_text = tk.Text(prompt_frame, height=4, font=("Segoe UI", 10), bg="#dff9fb", fg="#30336b")
            self.prompt_text.pack(fill="x", padx=5, pady=5)

            sample_prompt = "Create a social media post about {Topic} for {Industry Domain}. Include relevant hashtags."
            self.prompt_text.insert("1.0", sample_prompt)

            self.prompt_text.bind("<KeyRelease>", lambda e: self.validate_prompt_columns())
            self.cols_listbox.bind("<<ListboxSelect>>", lambda e: self.validate_prompt_columns())

            # --- Generate & progress frame ---
            gen_frame = tk.Frame(self.root, bg="#f5f6fa")
            gen_frame.pack(padx=10, pady=8, fill="x")

            # Remove Generate and Stop buttons from here

            # Progress row with details/message buttons (blue arrow row)
            progress_row = tk.Frame(gen_frame, bg="#f5f6fa")
            progress_row.pack(fill="x", pady=(0,2))

            self.progress = ttk.Progressbar(progress_row, orient="horizontal", length=200, mode="determinate")
            self.progress.pack(side="left", padx=(0,8))

            self.progress_info_label = tk.Label(progress_row, text="Ready", fg="#30336b", bg="#f5f6fa", font=("Segoe UI", 10, "bold"))
            self.progress_info_label.pack(side="left")

            # Place Hide Details and Hide Message Section buttons in the same row
            self.toggle_details_btn = tk.Button(progress_row, text="▼ Show Details", command=self.toggle_details, bg="#dff9fb", fg="#30336b", font=("Segoe UI", 9))
            self.toggle_details_btn.pack(side="left", padx=5)
            self.toggle_error_log_btn = tk.Button(progress_row, text="💬 Show Message Section", command=self.toggle_error_log, bg="#f6e58d", fg="#222f3e", font=("Segoe UI", 9))
            self.toggle_error_log_btn.pack(side="left", padx=5)

            # Foldable message section (was error section)
            self.details_frame = tk.Frame(gen_frame, bg="#f5f6fa")
            self.details_frame.pack(fill="x")
            self.details_visible = tk.BooleanVar(value=False)
            self.error_log_visible = tk.BooleanVar(value=False)
            self.error_logs = []
            self.progress_label = tk.Label(self.details_frame, text="", fg="#576574", bg="#f5f6fa", font=("Segoe UI", 9), justify="left", wraplength=700)
            self.progress_label.pack_forget()
            # Remove toggle buttons from here

            # --- Table preview ---
            preview_frame = tk.LabelFrame(self.root, text="📊 Table Preview", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 11, "bold"))
            preview_frame.pack(padx=10, pady=8, fill="both", expand=True)

            tree_scrollbar_y = ttk.Scrollbar(preview_frame, orient="vertical")
            tree_scrollbar_y.pack(side="right", fill="y")
            tree_scrollbar_x = ttk.Scrollbar(preview_frame, orient="horizontal")
            tree_scrollbar_x.pack(side="bottom", fill="x")

            self.tree = ttk.Treeview(preview_frame, yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)
            self.tree.pack(fill="both", expand=True)
            tree_scrollbar_y.config(command=self.tree.yview)
            tree_scrollbar_x.config(command=self.tree.xview)

            # Add retry button for selected row
            retry_frame = tk.Frame(preview_frame, bg="#f5f6fa")
            retry_frame.pack(fill="x", pady=4)
            self.retry_btn = tk.Button(retry_frame, text="🔄 Retry Selected Row", command=self.retry_selected_row, bg="#7ed6df", fg="#222f3e", font=("Segoe UI", 12, "bold"))
            self.retry_btn.pack(side="left", padx=5)

        except Exception as e:
            import traceback
            err_msg = f"UI setup failed: {e}\n{traceback.format_exc()}"
            self.error_logs.append(err_msg)
            messagebox.showerror("UI Error", err_msg)
            fallback = tk.Label(self.root, text="Critical UI error. See error log.", fg="red", bg="#f5f6fa", font=("Segoe UI", 14, "bold"))
            fallback.pack(padx=20, pady=20)

    def copy_selected_cell(self):
        """Copy the value of the selected cell to clipboard."""
        selected = self.tree.selection()
        if selected:
            item = selected[0]
            # Try to get the column index from mouse position
            col_id = self.tree.identify_column(self.tree.winfo_pointerx() - self.tree.winfo_rootx())
            if col_id and col_id.startswith('#'):
                col_idx = int(col_id[1:]) - 1
            else:
                col_idx = 0
            values = self.tree.item(item, 'values')
            if values and col_idx < len(values):
                value = values[col_idx]
                self.root.clipboard_clear()
                self.root.clipboard_append(value)
                messagebox.showinfo("Copied", f"Copied to clipboard:\n{value}")
            else:
                messagebox.showwarning("Copy", "No cell value found.")
        else:
            messagebox.showwarning("Copy", "No row selected.")

    def choose_file_again(self):
        """Allow user to choose a new file and reload."""
        self.load_file()

    def pause_generation(self):
        """Pause the generation process temporarily."""
        self.is_paused = True
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="normal")
        self.progress_label.config(text="Paused. Click Resume to continue.")

    def resume_generation(self):
        """Resume the generation process after pause."""
        self.is_paused = False
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")
        self.progress_label.config(text="Resumed.")

    def toggle_details(self):
        """Show/hide the detailed status info below progress bar."""
        if self.details_visible.get():
            self.progress_label.pack_forget()
            self.toggle_details_btn.config(text="▼ Show Details")
            self.details_visible.set(False)
        else:
            self.progress_label.pack(fill="x", padx=5)
            self.toggle_details_btn.config(text="▲ Hide Details")
            self.details_visible.set(True)
        self.update_details_area()

    def toggle_error_log(self):
        """Show/hide the message section in the details area."""
        self.error_log_visible.set(not self.error_log_visible.get())
        self.toggle_error_log_btn.config(
            text="🙈 Hide Message Section" if self.error_log_visible.get() else "💬 Show Message Section"
        )
        self.update_details_area()

    def update_details_area(self):
        """Update the details area to show all messages if toggled."""
        if self.error_log_visible.get():
            log_text = "\n".join(self.error_logs[-50:]) if self.error_logs else "No messages yet."
            self.progress_label.config(text=log_text)
        else:
            self.progress_label.config(text="Ready")

    def adapt_to_csv(self):
        """Update UI elements to match the uploaded CSV file."""
        if self.df is not None:
            # Update columns listbox
            self.cols_listbox.delete(0, "end")
            for col in self.df.columns:
                self.cols_listbox.insert("end", col)

            # Suggest a prompt template using the first two columns (use exact names)
            cols = list(self.df.columns)
            template = (
                f"Create a post about {{{cols[0]}}} for {{{cols[1]}}}." if len(cols) >= 2 else
                f"Create a post about {{{cols[0]}}}." if len(cols) == 1 else
                "Enter your prompt template here."
            )
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", template)
            self.validate_prompt_columns()

    def insert_placeholder_from_listbox(self, event):
        """Insert {ColumnName} into the prompt at cursor position on double-click."""
        selection = self.cols_listbox.curselection()
        if selection:
            col_name = self.cols_listbox.get(selection[0])
            self.prompt_text.insert(tk.INSERT, f"{{{col_name}}}")

    def validate_prompt_columns(self):
        """Validate that prompt placeholders match selected columns. Disable generate button if not."""
        prompt_template_raw = self.prompt_text.get("1.0", "end").strip()
        placeholders = set(re.findall(r"\{(.*?)\}", prompt_template_raw))
        selected_indices = self.cols_listbox.curselection()
        selected_cols = [self.cols_listbox.get(i) for i in selected_indices]
        # Only allow placeholders that exactly match selected columns
        invalid_placeholders = [p for p in placeholders if p not in selected_cols]
        if invalid_placeholders or not placeholders:
            self.generate_btn.config(state="disabled")
            self.progress_label.config(
                text="No placeholders found in prompt." if not placeholders else
                f"Invalid/missing columns for placeholders: {', '.join(invalid_placeholders)}. Use double-click on column to insert exact placeholder."
            )
        else:
            self.generate_btn.config(state="normal")
            self.progress_label.config(text="Ready")

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV or ZIP file",
            filetypes=[("CSV files", "*.csv"), ("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            if filepath.lower().endswith(".zip"):
                self.load_from_zip(filepath)
            else:
                self.load_from_csv(filepath)

            self.filename = filepath
            self.show_table_preview()
            self.adapt_to_csv()

            self.generate_btn.config(state="normal")
            self.save_btn.config(state="disabled")
            self.download_btn.config(state="disabled")
            self.log_message(f"✅ Loaded file: {os.path.basename(filepath)} | Rows: {len(self.df)}", "info")

        except Exception as e:
            self.log_message(f"❌ Failed to load file: {str(e)}", "error")

    def load_from_zip(self, filepath):
        with zipfile.ZipFile(filepath, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError("ZIP archive contains no CSV files.")

            # Use the first CSV file found
            with z.open(csv_files[0]) as f:
                # Try different separators and encodings
                content = f.read()
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        decoded_content = content.decode(encoding)
                        for sep in ['\t', ',', ';']:
                            try:
                                from io import StringIO
                                self.df = pd.read_csv(StringIO(decoded_content), sep=sep)
                                if len(self.df.columns) > 1:  # Successful parse
                                    return
                            except:
                                continue
                    except:
                        continue
                raise ValueError("Could not parse CSV from ZIP file")

    def load_from_csv(self, filepath):
        # Try different separators and encodings
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            for sep in ['\t', ',', ';']:
                try:
                    self.df = pd.read_csv(filepath, sep=sep, encoding=encoding)
                    if len(self.df.columns) > 1:  # Successful parse
                        return
                except:
                    continue
        raise ValueError("Could not parse CSV file with common separators")

    def show_table_preview(self):
        # Clear existing tree
        if self.tree is None:
            messagebox.showerror("Error", "Table widget not initialized. Please restart the app or check UI setup.")
            return
        self.tree.delete(*self.tree.get_children())

        # Configure columns
        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"

        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="w")

        # Insert all rows for preview (virtual scrolling for large files)
        max_rows = 1000  # Show up to 1000 rows for performance
        for idx, row in self.df.head(max_rows).iterrows():
            values = [str(row[col])[:47] + "..." if pd.notna(row[col]) and len(str(row[col])) > 50 else str(row[col]) if pd.notna(row[col]) else "" for col in self.df.columns]
            self.tree.insert("", "end", iid=idx, values=values)

        # If more rows, show a message
        if len(self.df) > max_rows:
            self.tree.insert("", "end", values=["... (showing first 1000 rows)"] + [""] * (len(self.df.columns) - 1))

    def retry_selected_row(self):
        selected = self.tree.selection()
        if selected:
            idx = int(selected[0])
            self.retry_row_generation(idx)

    def check_ollama_connection(self):
        """Check if Ollama server is running and update status"""

        def check():
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    self.is_ollama_connected = True
                    self.root.after(0, lambda: self.status_label and self.status_label.config(text="🟢 Connected", fg="green"))
                else:
                    self.is_ollama_connected = False
                    self.root.after(0, lambda: self.status_label and self.status_label.config(text="🔴 Server Error", fg="red"))
            except requests.exceptions.RequestException:
                self.is_ollama_connected = False
                self.root.after(0, lambda: self.status_label and self.status_label.config(text="🔴 Not Connected", fg="red"))

        threading.Thread(target=check, daemon=True).start()

    def generate_with_ollama(self, prompt, max_retries=2):
        """Generate content using Ollama local server with retry logic"""
        self.model_name = self.model_var.get().strip() or "llama3"

        # Optimize prompt for better performance
        optimized_prompt = f"{prompt}\n\nResponse:"

        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": optimized_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 200,  # Limit response length
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 2048,  # Reduce context window
                        "num_thread": 4  # Optimize threading
                    }
                }

                # Shorter timeout with exponential backoff
                timeout = 30 + (attempt * 15)  # 30s, 45s, 60s

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "").strip()
                    if generated_text:
                        return generated_text
                    else:
                        return "No response generated by the model."
                else:
                    if attempt == max_retries:
                        return f"Error: HTTP {response.status_code} after {max_retries + 1} attempts"
                    continue

            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    return f"Error: Request timed out after {max_retries + 1} attempts. Try using a smaller/faster model."
                continue
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    return f"Error: Connection failed - {str(e)}"
                continue
            except Exception as e:
                return f"Error: {str(e)}"

        return "Error: All retry attempts failed."

    def start_generation(self):
        """Start generation in a separate thread to prevent UI freezing"""
        if not self.is_ollama_connected:
            messagebox.showerror("Error",
                                 "❌ Ollama server is not connected!\n\nPlease ensure:\n1. Ollama is installed\n2. Ollama server is running (ollama serve)\n3. The model is available")
            return

        self.stop_generation_flag = False
        self.generate_btn.config(state="disabled", text="⏳ Generating...")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self.generate_posts, daemon=True).start()

    def stop_generation(self):
        """Stop the generation process"""
        self.stop_generation_flag = True
        self.stop_btn.config(state="disabled")
        self.progress_label.config(text="Stopping...")
        messagebox.showwarning("Stopping", "Generation will stop after current item completes.")

    def scrape_website(self, url):
        """Scrape the content of a website and return the text."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            # Extract visible text
            texts = soup.stripped_strings
            visible_text = ' '.join(texts)

            # Limit the text length to avoid excessive data
            max_length = 5000
            if len(visible_text) > max_length:
                visible_text = visible_text[:max_length] + " [Content truncated]"

            return visible_text

        except requests.exceptions.RequestException as e:
            return f"Error fetching website: {e}"
        except Exception as e:
            return f"Error processing website content: {e}"

    def generate_posts(self):
        import time
        try:
            selected_indices = self.cols_listbox.curselection()
            if not selected_indices:
                msg = "Please select at least one column to use as variables."
                self.error_logs.append(msg)
                self.root.after(0, lambda: messagebox.showerror("Error", msg))
                self.root.after(0, self.update_details_area)
                return

            selected_cols_raw = [self.cols_listbox.get(i) for i in selected_indices]
            selected_cols = [col.strip() for col in selected_cols_raw]
            col_map = {col.strip(): col for col in selected_cols_raw}

            prompt_template_raw = self.prompt_text.get("1.0", "end").strip()
            if not prompt_template_raw:
                msg = "Please enter a prompt template."
                self.error_logs.append(msg)
                self.root.after(0, lambda: messagebox.showerror("Error", msg))
                self.root.after(0, self.update_details_area)
                return

            placeholders_raw = re.findall(r"\{(.*?)\}", prompt_template_raw)
            placeholders = [p.strip() for p in placeholders_raw]
            invalid_placeholders = [p for p in placeholders if p not in selected_cols]
            if invalid_placeholders or not placeholders:
                msg = f"Prompt placeholders not in selected columns: {invalid_placeholders}"
                self.error_logs.append(msg)
                self.root.after(0, lambda: messagebox.showerror("Error", msg))
                self.root.after(0, self.update_details_area)
                return

            template = jinja2.Template(prompt_template_raw)

            total_rows = len(self.df)
            self.progress["maximum"] = total_rows
            self.progress["value"] = 0

            results = []
            self.df["AI_Generated_Column"] = ""

            start_time = time.time()
            avg_time_per_row = None

            for idx, row in self.df.iterrows():
                if self.stop_generation_flag:
                    self.root.after(0, lambda: messagebox.showinfo("Stopped", "Generation stopped by user."))
                    break

                if self.is_paused:
                    while self.is_paused and not self.stop_generation_flag:
                        time.sleep(0.2)

                row_start = time.time()
                self.progress["value"] = idx + 1

                context = {col_map.get(p, p): str(row[col_map.get(p, p)]) if col_map.get(p, p) in row and pd.notna(row[col_map.get(p, p)]) else "" for p in placeholders}

                # Scrape website content if 'Website' or similar column is present
                if 'Website' in context and context['Website']:
                    scraped_content = self.scrape_website(context['Website'])
                    context['WebsiteContent'] = scraped_content  # Add scraped content to context

                missing_in_context = [p for p in placeholders if not context.get(col_map.get(p, p))]
                if missing_in_context:
                    generated_post = f"Error: Missing variables for placeholders: {missing_in_context}"
                    rendered_prompt = prompt_template_raw
                    self.error_logs.append(generated_post)
                    self.root.after(0, self.update_details_area)
                else:
                    try:
                        rendered_prompt = template.render(**context)
                        if len(rendered_prompt.strip()) < 10:
                            generated_post = "Error: Prompt too short or empty"
                            self.error_logs.append(generated_post)
                            self.root.after(0, self.update_details_area)
                        else:
                            json_data = json.dumps(context, ensure_ascii=False, indent=2)
                            llama_prompt = (
                                f"Be concise and direct. Use the following data to create a post:\n"
                                f"{json_data}\n"
                                f"Prompt: {rendered_prompt}\nResponse:"
                            )
                            elapsed = time.time() - start_time
                            rows_left = total_rows - (idx + 1)
                            if idx > 0:
                                avg_time_per_row = elapsed / (idx + 1)
                                est_time_left = avg_time_per_row * rows_left
                                est_str = time.strftime('%M:%S', time.gmtime(est_time_left))
                            else:
                                est_str = "--:--"
                            info_text = f"Rows left: {rows_left}   |   Estimated time left: {est_str}"
                            self.root.after(0, lambda t=info_text: self.progress_info_label.config(text=t))

                            generated_post = self.generate_with_ollama(llama_prompt)
                    except Exception as e:
                        generated_post = f"Error rendering template: {e}"
                        self.error_logs.append(str(e))
                        self.root.after(0, self.update_details_area)

                self.df.at[idx, "AI_Generated_Column"] = generated_post
                results.append(generated_post)

                if idx % 10 == 0 or idx == total_rows - 1:
                    self.root.after(0, self.show_table_preview)
                self.root.update_idletasks()

            self.df["AI_Generated_Column"] = results
            self.root.after(0, self.update_ui_after_generation, "")

        except Exception as e:
            self.error_logs.append(f"Generation failed: {str(e)}")
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Generation failed:\n{str(e)}"))
            self.root.after(0, self.update_details_area)
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state="normal", text="⚙️ Generate Posts"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
            self.root.after(0, lambda: self.progress_label.config(text=""))
            self.root.after(0, lambda: self.progress_info_label.config(text="Ready"))

    def update_ui_after_generation(self, preview_text):
        """Update UI elements after generation is complete"""
        self.show_table_preview()
        # Enable save and download if at least one row is processed
        if "AI_Generated_Column" in self.df.columns and any(self.df["AI_Generated_Column"].astype(str).str.strip() != ""):
            self.save_btn.config(state="normal")
            self.download_btn.config(state="normal")

    def retry_row_generation(self, idx):
        """Retry Ollama generation for a specific row."""
        import time
        try:
            selected_indices = self.cols_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select at least one column to use as variables.")
                return

            selected_cols_raw = [self.cols_listbox.get(i) for i in selected_indices]
            selected_cols = [col.strip() for col in selected_cols_raw]
            col_map = {col.strip(): col for col in selected_cols_raw}

            prompt_template_raw = self.prompt_text.get("1.0", "end").strip()
            if not prompt_template_raw:
                messagebox.showerror("Error", "Please enter a prompt template.")
                return

            placeholders_raw = re.findall(r"\{(.*?)\}", prompt_template_raw)
            placeholders = [p.strip() for p in placeholders_raw]
            invalid_placeholders = [p for p in placeholders if p not in selected_cols]
            if invalid_placeholders or not placeholders:
                messagebox.showerror("Error", f"Prompt placeholders not in selected columns: {invalid_placeholders}")
                return

            template = jinja2.Template(prompt_template_raw)

            row = self.df.iloc[idx]
            context = {col_map.get(p, p): str(row[col_map.get(p, p)]) if col_map.get(p, p) in row and pd.notna(row[col_map.get(p, p)]) else "" for p in placeholders}

            missing_in_context = [p for p in placeholders if not context.get(col_map.get(p, p))]
            if missing_in_context:
                generated_post = f"Error: Missing variables for placeholders: {missing_in_context}"
            else:
                try:
                    rendered_prompt = template.render(**context)
                    if len(rendered_prompt.strip()) < 10:
                        generated_post = "Error: Prompt too short or empty"
                    else:
                        json_data = json.dumps(context, ensure_ascii=False, indent=2)
                        llama_prompt = (
                            f"Be concise and direct. Use the following data to create a post:\n"
                            f"{json_data}\n"
                            f"Prompt: {rendered_prompt}\nResponse:"
                        )
                        generated_post = self.generate_with_ollama(llama_prompt)
                except Exception as e:
                    generated_post = f"Error rendering template: {e}"

            self.df.at[idx, "AI_Generated_Column"] = generated_post
            self.show_table_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Retry failed: {e}")
        finally:
            self.save_btn.config(state="normal")
            self.progress["value"] = 0
            self.progress_label.config(text="Ready")
            self.generate_btn.config(state="normal", text="⚙️ Generate Posts")
            self.stop_btn.config(state="disabled")

            if not self.stop_generation_flag:
                messagebox.showinfo("Success",
                                    "✅ Post generation completed! Check the preview below and save your results.")
            else:
                messagebox.showinfo("Partially Complete", "⚠️ Generation was stopped. Partial results are available.")

    def save_csv(self):
        if self.df is None or "AI_Generated_Column" not in self.df.columns or not any(self.df["AI_Generated_Column"].astype(str).str.strip() != ""):
            self.log_message("No generated posts to save. Please generate first.", "error")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not save_path:
            return

        try:
            self.df.to_csv(save_path, index=False, encoding='utf-8')
            self.log_message(f"✅ File saved successfully to: {save_path}", "info")
        except Exception as e:
            self.log_message(f"❌ Failed to save file: {str(e)}", "error")

    def download_results(self):
        if self.df is None or "AI_Generated_Column" not in self.df.columns or not any(self.df["AI_Generated_Column"].astype(str).str.strip() != ""):
            self.log_message("No generated posts to download. Please generate first.", "error")
            return

        download_path = filedialog.asksaveasfilename(
            title="Download Results CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not download_path:
            return

        try:
            self.df.to_csv(download_path, index=False, encoding='utf-8')
            self.log_message(f"✅ Results downloaded successfully to: {download_path}", "info")
        except Exception as e:
            self.log_message(f"❌ Failed to download results: {str(e)}", "error")

    def open_chat_test(self):
        """Open a chat test window for testing requests."""
        chat_window = tk.Toplevel(self.root)
        chat_window.title("Chat Test")
        chat_window.geometry("600x400")
        chat_window.configure(bg="#f5f6fa")

        tk.Label(chat_window, text="Enter your prompt:", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 11, "bold")).pack(pady=10)

        prompt_entry = tk.Text(chat_window, height=5, font=("Segoe UI", 10), bg="#dff9fb", fg="#30336b")
        prompt_entry.pack(fill="x", padx=10, pady=5)

        response_label = tk.Label(chat_window, text="Response:", bg="#f5f6fa", fg="#30336b", font=("Segoe UI", 11, "bold"))
        response_label.pack(pady=10)

        response_text = tk.Text(chat_window, height=10, font=("Segoe UI", 10), bg="#dff9fb", fg="#30336b", state="disabled")
        response_text.pack(fill="both", padx=10, pady=5, expand=True)

        def send_prompt():
            prompt = prompt_entry.get("1.0", "end").strip()
            if not prompt:
                messagebox.showwarning("Warning", "Please enter a prompt.")
                return

            response_text.config(state="normal")
            response_text.delete("1.0", "end")
            response_text.insert("1.0", "⏳ Generating response...")
            response_text.config(state="disabled")

            def generate_response():
                response = self.generate_with_ollama(prompt)
                response_text.config(state="normal")
                response_text.delete("1.0", "end")
                response_text.insert("1.0", response)
                response_text.config(state="disabled")

            threading.Thread(target=generate_response, daemon=True).start()

        send_btn = tk.Button(chat_window, text="Send", command=send_prompt, bg="#7ed6df", fg="#222f3e", font=("Segoe UI", 10, "bold"))
        send_btn.pack(pady=10)


def main():
    root = tk.Tk()
    app = CSVAgentApp(root)

    # Center the window
    root.eval('tk::PlaceWindow . center')

    root.mainloop()


if __name__ == "__main__":
    main()