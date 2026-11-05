"""
VisionAssist 2.0 - Main Entry Point

Launches the VisionAssist 2.0 desktop application.
"""

import tkinter as tk

from app import VisionAssistApp


def main() -> None:
    """Start the VisionAssist 2.0 application."""

    root = tk.Tk()
    app = VisionAssistApp(root)

    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.mainloop()


if __name__ == "__main__":
    main()