import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
from neural_network import Network

tr_data, val_data, te_data = ld.load_and_wrap()
net = Network([784, 30, 30, 10])


class NumberDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Number")

        self.image = Image.new("L", (280, 280), "black")
        self.draw_tool = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(root, width=280, height=280, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, pady=10, padx=10)

        self.predict_button = tk.Button(root, text="Process Image", command=self.process_image)
        self.predict_button.grid(row=1, column=0, pady=10, padx=10)

        self.train_button = tk.Button(root, text="Train", command=self.train_network)
        self.train_button.grid(row=1, column=1, pady=10, padx=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=2, pady=10, padx=10)

        # --- NEW: Bind all three mouse events ---
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        self.last_x, self.last_y = None, None

    def start_stroke(self, event):
        """Records the starting position of a new stroke."""
        self.last_x, self.last_y = event.x, event.y

    def draw_stroke(self, event):
        """Draws a line segment for the current stroke."""
        if self.last_x and self.last_y:
            self.draw_tool.line(
                [(self.last_x, self.last_y), (event.x, event.y)],
                fill="white",
                width=25,
                joint="round",
            )
            self.update_canvas()
        
        # Update position for the next segment in the same stroke
        self.last_x, self.last_y = event.x, event.y
    
    def end_stroke(self, event):
        """Forgets the last position when the mouse button is lifted."""
        self.last_x, self.last_y = None, None

    def update_canvas(self):
        """Displays the in-memory image on the canvas."""
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def clear_canvas(self):
        """Clears the canvas and also forgets the last mouse position."""
        self.draw_tool.rectangle([0, 0, 280, 280], fill="black")
        self.canvas.delete("all")
        # --- FIX: Reset the coordinates on clear ---
        self.end_stroke(None) # Call end_stroke to reset coordinates
        print("Canvas cleared.")

    def train_network(self):
        net.SGD(tr_data, 20, 10, 3.0, test_data=te_data)
        print("Network trained successfully!")

    def process_image(self):
        """Processes the drawing directly from the in-memory image."""
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = img_array / 255.0

        print("Image processed successfully!")
        print("Shape of the array:", img_array.shape)

        plt.imshow(img_array, cmap='gray')
        plt.title("Processed 28x28 Image")
        plt.show()

        img_processed = np.reshape(img_array, (784, 1))
        print(net.recognise(img_processed))



# --- Main application setup (remains the same) ---
if __name__ == "__main__":
    root = tk.Tk()
    app = NumberDrawer(root)
    root.mainloop()