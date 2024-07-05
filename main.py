import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import threading


# Initialize main window
root = tk.Tk()
root.title("Document-Based Question Answering System - The Pycodes")
root.geometry("800x600")


# Initialize T5 tokenizer and model
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Setting up the document content variable
document_content = ""


def upload_document():
   global document_content
   # We Open file dialog to select a text file
   file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
   if file_path:
       with open(file_path, 'r') as file:
           document_content = file.read()
       document_text.delete(1.0, tk.END)
       document_text.insert(tk.END, document_content)


def get_answer_thread():
   # Start a new thread to keep the GUI responsive
   threading.Thread(target=get_answer).start()


def get_answer():
   try:
       global document_content
       # Get question from GUI input
       question = question_entry.get()
       context = document_content or document_text.get(1.0, tk.END).strip()


       print(f"Question: {question}")
       print(f"Context: {context[:200]}...")  # Display only the first 200 characters for debugging


       if not question or not context:
           messagebox.showerror("Error", "Please enter a question and provide The document content.")
           return


       # Prepare the input text for T5
       input_text = f"question: {question} context: {context}"
       inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)


       # Generate the answer
       with torch.no_grad():
           outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)


       answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
       print(f"Answer: {answer}")


       # Update answer label in GUI
       answer_label.config(text=f"Answer: {answer}")
   except Exception as e:
       print(f"Error: {e}")
       messagebox.showerror("Error", f"An error occurred: {e}")


# Create GUI elements
# Title
tk.Label(root, text="The Pycodes: Document-Based Question Answering System", font=("Helvetica", 16, "bold")).pack(pady=10)


# Upload button
upload_button = tk.Button(root, text="Upload Document", command=upload_document, bg="lightblue", fg="black")
upload_button.pack(pady=10)


# Document display
tk.Label(root, text="Document Content:").pack()
document_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
document_text.pack(pady=10)


# Question input
tk.Label(root, text="Enter your question:").pack()
question_entry = tk.Entry(root, width=80)
question_entry.pack(pady=10)


# Answer display
answer_label = tk.Label(root, text="Answer will be displayed here.", wraplength=700, justify=tk.LEFT, bg="lightgrey", anchor="w")
answer_label.pack(pady=10, fill=tk.BOTH, padx=10)


# Answer button
answer_button = tk.Button(root, text="Get Answer", command=get_answer_thread, bg="lightblue", fg="black")
answer_button.pack(pady=10)


# Run the GUI application
root.mainloop()
