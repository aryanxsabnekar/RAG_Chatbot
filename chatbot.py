import os
import threading
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import pytz


os.environ["OPENAI_API_KEY"] = "sk-zwyjV0GR8upv6-6jozlUwh9zcMUlSe_J3iWm5tHrcIT3BlbkFJYr_Ob38FnM854kk3iqbLjDGPQdCozGbaYcklu2i6AA"

loader = TextLoader("handbook_data.txt")
docs = loader.load()

txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)
txts = txt_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(txts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=vectorstore.as_retriever())

def get_eastern_time():
    eastern = pytz.timezone('America/New_York')
    return datetime.now(eastern).strftime("%H:%M:%S")


def create_rounded_button(parent,text,command):
    style = ttk.Style()
    style.configure("Round.TButton",padding=6,relief="flat",background="#4CAF50")
    return ttk.Button(parent,text=text,command=command,style="Round.TButton")

root = ThemedTk(theme="arc")
root.title("RAGatha Chatbot")
root.geometry("800x600")

root.grid_rowconfigure(0,weight=1)
root.grid_columnconfigure(0,weight=1)

chat_frame = ttk.Frame(root,padding="10")
chat_frame.grid(row=0,column=0,sticky="nsew")
chat_frame.grid_rowconfigure(0,weight=1)
chat_frame.grid_columnconfigure(0,weight=1)

chat_history = tk.Text(chat_frame,state=tk.DISABLED,wrap=tk.WORD,font=("Arial",12))
chat_history.grid(row=0,column=0,sticky="nsew")

scrollbar = ttk.Scrollbar(chat_frame,orient="vertical",command=chat_history.yview)
scrollbar.grid(row=0,column=1,sticky="ns")
chat_history.configure(yscrollcommand=scrollbar.set)

def display_initial_message():
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "RAGatha: Hello, I am RAGatha the Chatbot. What can I answer?\n\n","ai")
    chat_history.tag_configure("ai",foreground="#28a745",font=("Arial",12,"bold"))
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)
    
display_initial_message()

input_frame = ttk.Frame(root,padding="10")
input_frame.grid(row=1,column=0,sticky="ew")
input_frame.grid_columnconfigure(0,weight=1)

entry = ttk.Entry(input_frame,font=("Arial",12))
entry.grid(row=0,column=0,sticky="ew",padx=(0,10))

def generate_response(user_input):
    result = qa_chain.run(user_input)
    timestamp = get_eastern_time()
    
    chat_history.config(state=tk.NORMAL)
    chat_history.delete("end-2l","end-1c")
    chat_history.insert(tk.END, f"[{timestamp}] RAGatha: {result}\n\n", "ai")
    chat_history.tag_configure("ai",foreground="#28a745",font=("Arial", 12,"bold"))
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)
    


def handle_message():
    user_input = entry.get()
    if user_input.strip():
        timestamp = get_eastern_time()
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"[{timestamp}] You: {user_input}\n\n", "user")
        chat_history.tag_configure("user",foreground="#007bff",font=("Arial",12,"bold"))
        
        chat_history.insert(tk.END, f"[{timestamp}] RAGatha: Generating response...\n", "generating")
        chat_history.tag_configure("generating",foreground="#FFA500",font=("Arial", 12,"italic"))
        
        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END)

        entry.delete(0, tk.END)

        threading.Thread(target=generate_response,args=(user_input,),daemon=True).start()

send_button = create_rounded_button(input_frame,"Send",handle_message)
send_button.grid(row=0,column=1)

root.bind('<Return>', lambda event: handle_message())

root.mainloop()
