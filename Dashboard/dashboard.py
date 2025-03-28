import db_manager
import io
import queue
import base64
import tkinter as tk
from tkinter import Canvas, Scrollbar
from PIL import Image, ImageTk
from datetime import datetime

# Fixed window size
WINDOW_WIDTH = 1350
WINDOW_HEIGHT = 600

# Image Grid Settings
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 240
GRID_COLUMNS = 3  # Number of images per row

image_references = [] # Hold the image data


def initialise_dashboard(message_queue):
    """Initialise Tkinter interface"""
    message_queue = message_queue

    root = tk.Tk()
    root.title("Dashboard")
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

    # Main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # Left frame (Images)
    image_frame = tk.Frame(main_frame, width=WINDOW_WIDTH//2, height=WINDOW_HEIGHT)
    image_frame.pack(side="left", fill="both", expand=True)

    canvas = Canvas(image_frame)
    scroll_y = Scrollbar(image_frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set)

    canvas.pack(side="left", fill="both", expand=True)
    scroll_y.pack(side="right", fill="y")

    # Right frame (Text)
    text_frame = tk.Frame(main_frame, width=WINDOW_WIDTH//2, height=WINDOW_HEIGHT)
    text_frame.pack(side="right", fill="both", expand=True)

    text_listbox = tk.Listbox(text_frame, font=("Arial", 12))
    text_listbox.pack(side="left", fill="both", expand=True)

    text_scroll = Scrollbar(text_frame, orient="vertical", command=text_listbox.yview)
    text_scroll.pack(side="right", fill="y")
    text_listbox.config(yscrollcommand=text_scroll.set)

    load_data(text_listbox, scroll_frame)

    root.after(1000, update_ui, root, text_listbox, scroll_frame, message_queue)

    root.mainloop()


def update_ui(root, text_listbox, scroll_frame, message_queue):
    """Update the UI if there is a new queue item every 1 second"""
    if not message_queue.empty():
        # If message queue not empty, get and display contents
        msg_type, data = message_queue.get()

        if msg_type == "text":
            display_text(text_listbox, data)

        elif msg_type == "image":
            try:
                display_image(scroll_frame, data)
            except Exception as e:
                print(e)

    root.after(1000, update_ui, root, text_listbox, scroll_frame, message_queue)


def load_data(text_listbox, scroll_frame):
    """Get database content and load data into interface"""
    data = db_manager.getAllContent()
    data.sort(key=lambda x: datetime.strptime(x["payload"]["timestamp"], "%Y-%m-%d %H:%M:%S.%f"))

    for x in data:
        payload = x["payload"]

        if payload["type"] == 'text':
            display_text(text_listbox, payload["timestamp"] + " " + payload["text"])
        elif payload["type"] == 'image':
            try:
                decoded_content = base64.b64decode(payload["text"].encode("utf-8"))
                display_image(scroll_frame, bytearray(decoded_content))
            except Exception as e:
                print(e)


def display_text(text_listbox, data):
    """Append text to list"""
    text_listbox.insert(tk.END, data)  # Add text item


def display_image(scroll_frame, data):
    """Append image to frame"""
    image = Image.open(io.BytesIO(data))
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    photo = ImageTk.PhotoImage(image)
    
    image_label = tk.Label(scroll_frame, image=photo)
    image_label.image = photo  # Keep reference
    image_references.append(photo)  # Prevent garbage collection

    row = len(image_references) // GRID_COLUMNS
    col = len(image_references) % GRID_COLUMNS

    image_label.grid(row=row, column=col, padx=5, pady=5)
