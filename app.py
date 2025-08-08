import os
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
from PIL import Image

# Load the .env file
load_dotenv()

# Set the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro Vision model (for multimodal)
model = genai.GenerativeModel('gemini-pro-vision')

# Define a function that takes text and image input
def gemini_multimodal_chat(text_input, image_input):
    if image_input is not None:
        image = Image.fromarray(image_input)
        response = model.generate_content([text_input, image])
    else:
        response = model.generate_content(text_input)
    return response.text

# Build the Gradio interface
gr.Interface(
    fn=gemini_multimodal_chat,
    inputs=[
        gr.Textbox(label="Your Question or Prompt"),
        gr.Image(label="Upload Image (Optional)", type="numpy")
    ],
    outputs="text",
    title="Gemini Multi-Modal Chatbot",
    description="Ask questions with text and/or images using Gemini!"
).launch()
