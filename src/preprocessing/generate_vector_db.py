"""
YouTube Video Processing and Vector Database Generation.

This module provides functionality for processing YouTube videos about football tactics,
extracting their transcripts, and creating a FAISS vector database for efficient
semantic search. It supports Dutch language videos and handles multiple video URLs
in batch.
"""

import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class YouTubeProcessor:
    """A processor for YouTube videos to create a vector database.
    
    This class handles the extraction of transcripts from YouTube videos,
    processes them into chunks, and creates a FAISS vector database for
    efficient semantic search.
    
    Attributes:
        output_dir (str): Directory to store the FAISS database
        embeddings (OpenAIEmbeddings): Embedding model for vector creation
    """

    def __init__(self, output_dir: str = "faiss_db") -> None:
        """
        Initialize the YouTube processor.
        
        Args:
            output_dir (str): Directory to store the FAISS database
        """
        load_dotenv()
        self.output_dir = output_dir
        self.embeddings = OpenAIEmbeddings()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _load_video_transcript(self, url: str) -> Optional[List[Document]]:
        """
        Load and process a single YouTube video transcript.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Optional[List[Document]]: List of document chunks if successful, None if failed
            
        Raises:
            Exception: If there's an error processing the video
        """
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=["nl"],
                translation="nl"
            )
            return loader.load()
        except Exception as e:
            print(f"Error processing video {url}: {str(e)}")
            return None

    def _split_transcripts(self, documents: List[Document]) -> List[Document]:
        """
        Split transcripts into smaller chunks for better processing.
        
        Args:
            documents (List[Document]): List of document objects to split
            
        Returns:
            List[Document]: List of split document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def process_videos(self, video_urls: List[str]) -> None:
        """
        Process a list of YouTube videos, extract transcripts, and create a FAISS vector database.
        
        This method:
        1. Loads transcripts from all videos
        2. Splits them into manageable chunks
        3. Creates a FAISS vector database
        4. Saves the database to disk
        
        Args:
            video_urls (List[str]): List of YouTube video URLs
            
        Raises:
            ValueError: If no transcripts were successfully loaded
        """
        # Load and merge transcripts from all videos
        all_transcripts = []
        for url in video_urls:
            documents = self._load_video_transcript(url)
            if documents:
                all_transcripts.extend(documents)

        if not all_transcripts:
            raise ValueError("No transcripts were successfully loaded from the provided videos")

        # Split the merged transcripts into chunks
        chunks = self._split_transcripts(all_transcripts)

        # Create and save FAISS vector database
        db = FAISS.from_documents(chunks, self.embeddings)
        db.save_local(self.output_dir)
        print(f"FAISS database saved to {self.output_dir}")

def process_youtube_videos(video_urls: List[str], output_dir: str = "faiss_db") -> None:
    """
    Process YouTube videos and create a FAISS vector database.
    
    This is a convenience function that creates a YouTubeProcessor instance
    and uses it to process the provided videos.
    
    Args:
        video_urls (List[str]): List of YouTube video URLs
        output_dir (str): Directory to store the FAISS database
    """
    processor = YouTubeProcessor(output_dir)
    processor.process_videos(video_urls)

if __name__ == "__main__":
    # Example video URLs about Ajax football tactics
    VIDEOS = [
        "https://www.youtube.com/watch?v=gRr-elA4CAI",
        "https://www.youtube.com/watch?v=musCHUwvaCQ",
        "https://www.youtube.com/watch?v=fjgZL41lE0s&pp=ygUadm9ldGJhbCBpbnRlcm5hdGlvbmFsIGFqYXg%3D",
        "https://www.youtube.com/watch?v=6B4wMci5W-o&t=1040s&pp=ygUadm9ldGJhbCBpbnRlcm5hdGlvbmFsIGFqYXg%3D",
        "https://www.youtube.com/watch?v=EVD4DbWMevs&pp=ygUidm9ldGJhbCBpbnRlcm5hdGlvbmFsIGFqYXggdGFjdGllaw%3D%3D",
        "https://www.youtube.com/watch?v=aBOYYDvYR4g",
        "https://www.youtube.com/watch?v=-a3nXl12glU",
        "https://www.youtube.com/watch?v=-W9nj4RCMW0",
        "https://www.youtube.com/watch?v=AA5A2o4t4kc&t=474s",
        "https://www.youtube.com/watch?v=V5cmGvLAOXg&t=825s"
    ]
    
    # Set output directory relative to project root
    OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    
    process_youtube_videos(VIDEOS, str(OUTPUT_DIR))