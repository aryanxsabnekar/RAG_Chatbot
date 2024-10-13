import os
import threading
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import pytz

# Set OpenAI API key for accessing the OpenAI API
os.environ["OPENAI_API_KEY"] = ""  # Enter your own OPENAI API KEY

'''
Create an instance of TextLoader to load the content from "handbook_data.txt".
The `load()` method reads the text file and returns a list of documents,
which will be processed further for embedding and retrieval.
'''
loader = TextLoader("handbook_data.txt")
docs = loader.load()

'''
Initialize a RecursiveCharacterTextSplitter with a chunk size of 1000 characters 
and an overlap of 200 characters, then splits the documents ('docs') into smaller 
manageable chunks, storing the results in 'txts'.
'''
txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
txts = txt_splitter.split_documents(docs)

# Create an instance of OpenAIEmbeddings to generate embeddings for text chunks.
embeddings = OpenAIEmbeddings()

# Create a Chroma vector store from the previously split documents (txts) and their embeddings.
vectorstore = Chroma.from_documents(txts, embeddings)

# Create a RetrievalQA instance for question-answering using a retrieval-based approach.
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())


def get_eastern_time():
    """
    Retrieve the current time in the Eastern Time Zone (America/New_York).
    Returns the time formatted as a string in "HH:MM:SS" format.
    """
    eastern = pytz.timezone('America/New_York')
    return datetime.now(eastern).strftime("%H:%M:%S")


def create_rounded_button(parent, text, command):
    """
    Create a themed rounded button for a given parent widget.

    Parameters:
    - parent: The parent widget to which the button will be attached.
    - text: The label text for the button.
    - command: The function to be executed when the button is clicked.

    Returns:
    A ttk.Button instance styled as a rounded button.
    """
    style = ttk.Style()
    style.configure("Round.TButton", padding=6, relief="flat", background="#4CAF50")
    return ttk.Button(parent, text=text, command=command, style="Round.TButton")


'''
Initialize the main application window for the RAGatha Chatbot.

This sets up the window with a themed appearance, title, and specified dimensions.
It also configures the grid layout to allow for responsive resizing of the window's
rows and columns based on the weight settings.
'''
root = ThemedTk(theme="arc")
root.title("RAGatha Chatbot")
root.geometry("800x600")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

'''
Create and configure the chat frame within the main application window.

This frame serves as the container for the chat interface. It is padded with 10 pixels,
positioned in the grid at row 0, column 0, and set to expand and fill the available space 
in both vertical and horizontal directions. The row and column configurations allow 
for responsive resizing.
'''
chat_frame = ttk.Frame(root, padding="10")
chat_frame.grid(row=0, column=0, sticky="nsew")
chat_frame.grid_rowconfigure(0, weight=1)
chat_frame.grid_columnconfigure(0, weight=1)

'''
Create and configure the chat history text widget.

This widget displays the conversation history within the chat frame. It is initialized 
as disabled (non-editable) to prevent user input, wraps text at word boundaries, 
and uses a font size of 12 in Arial.
'''
chat_history = tk.Text(chat_frame, state=tk.DISABLED, wrap=tk.WORD, font=("Arial", 12))
chat_history.grid(row=0, column=0, sticky="nsew")

'''
Create a vertical scrollbar for the chat history.

The scrollbar is linked to the chat history widget for scrolling through the text. 
It is placed in the grid at row 0, column 1, and configured to control the widget's 
vertical view while updating its position as needed.
'''
scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=chat_history.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
chat_history.configure(yscrollcommand=scrollbar.set)


def display_initial_message():
    """
    Display the initial greeting message from the chatbot in the chat history.

    This function temporarily enables the chat history for editing, inserts a welcome 
    message, configures the message's appearance, and then disables the chat history 
    to prevent user edits. It also ensures the latest message is visible.
    """
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "RAGatha: Hello, I am RAGatha the Chatbot. What can I answer?\n\n", "ai")
    chat_history.tag_configure("ai", foreground="#28a745", font=("Arial", 12, "bold"))
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)


display_initial_message()  # Call the function to show the welcome message

'''
Create and configure the input frame and entry widget for user input.

The input frame is padded with 10 pixels and positioned in the grid at row 1, 
column 0, with horizontal expansion. The entry widget is initialized with a font 
size of 12 in Arial and is placed within the input frame, allowing users to type 
their messages. The entry field is set to expand horizontally with padding on the right.
'''
input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=1, column=0, sticky="ew")
input_frame.grid_columnconfigure(0, weight=1)

entry = ttk.Entry(input_frame, font=("Arial", 12))
entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))


def generate_response(user_input):
    """
    Generate a response to the user's input and update the chat history.

    This function processes the user's input using the qa_chain to obtain a response. 
    It retrieves the current Eastern Time for timestamping, updates the chat history 
    to display the response, formats the message with a specific appearance, and ensures 
    the latest message is visible while disabling further editing in the chat history.
    """
    result = qa_chain.run(user_input)
    timestamp = get_eastern_time()

    chat_history.config(state=tk.NORMAL)
    chat_history.delete("end-2l", "end-1c")
    chat_history.insert(tk.END, f"[{timestamp}] RAGatha: {result}\n\n", "ai")
    chat_history.tag_configure("ai", foreground="#28a745", font=("Arial", 12, "bold"))
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)


def handle_message():
    """
    Handle the user's message input and update the chat history.

    This function retrieves the user's input from the entry widget and checks if it's 
    not empty. If valid, it timestamps the message, updates the chat history to display 
    both the user's input and a temporary message indicating that a response is being 
    generated. The chat history is formatted for each message type, and then a separate 
    thread is started to generate the chatbot's response, allowing the interface to remain responsive.
    """
    user_input = entry.get()
    if user_input.strip():
        timestamp = get_eastern_time()
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"[{timestamp}] You: {user_input}\n\n", "user")
        chat_history.tag_configure("user", foreground="#007bff", font=("Arial", 12, "bold"))

        chat_history.insert(tk.END, f"[{timestamp}] RAGatha: Generating response...\n", "generating")
        chat_history.tag_configure("generating", foreground="#FFA500", font=("Arial", 12, "italic"))

        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END)

        entry.delete(0, tk.END)

        threading.Thread(target=generate_response, args=(user_input,), daemon=True).start()


'''
Create a rounded button in the input frame labeled "Send", 
which triggers the handle_message function when clicked.
'''
send_button = create_rounded_button(input_frame, "Send", handle_message)
send_button.grid(row=0, column=1)

'''
Bind the Enter key to the handle_message function, 
allowing users to submit their input by pressing Enter.
'''
root.bind('<Return>', lambda event: handle_message())

# Start the main event loop to run the application.
root.mainloop()
