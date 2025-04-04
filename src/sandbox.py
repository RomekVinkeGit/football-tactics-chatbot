
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from preprocessing.langchain_helper import process_youtube_videos

load_dotenv()

# Retrieve variables

videos = ["https://www.youtube.com/watch?v=gRr-elA4CAI", "https://www.youtube.com/watch?v=3OYQpXe-ys4"]

result = process_youtube_videos(videos, r"C:\Users\Romek\Documents\projects\football-tactics-chatbot\data")


#var1 = os.getenv("OPENAI_API_KEY")

 