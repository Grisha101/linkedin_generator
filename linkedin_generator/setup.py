from cx_Freeze import setup, Executable

setup(
    name="CSVAgentApp",
    version="1.0",
    description="AI Content Generator with Ollama",
    executables=[Executable("main.py", base="Win32GUI")]
)
