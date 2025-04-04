"""
Unit tests for the YouTubeProcessor class.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import sys
import os

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_dir))

from preprocessing.generate_vector_db import YouTubeProcessor
from langchain.schema import Document

class TestYouTubeProcessor(unittest.TestCase):
    """Test cases for the YouTubeProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = "test_output"
        self.processor = YouTubeProcessor(self.test_dir)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove test directory if it exists
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_init_creates_output_dir(self):
        """Test that initialization creates the output directory."""
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.isdir(self.test_dir))

    @patch('langchain_community.document_loaders.YoutubeLoader.from_youtube_url')
    def test_load_video_transcript(self, mock_loader):
        """Test loading a video transcript."""
        # Mock the loader's load method
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Test transcript content")
        ]
        mock_loader.return_value = mock_loader_instance

        # Test loading a transcript
        test_url = "https://www.youtube.com/watch?v=test"
        result = self.processor._load_video_transcript(test_url)

        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "Test transcript content")

        # Verify the loader was called with correct parameters
        mock_loader.assert_called_once_with(
            test_url,
            add_video_info=False,
            language=["nl"],
            translation="nl"
        )

    def test_split_transcripts(self):
        """Test splitting transcripts into chunks."""
        # Create test documents
        test_docs = [
            Document(page_content="First test content " * 50),  # Long content
            Document(page_content="Second test content " * 50)  # Long content
        ]

        # Split the documents
        chunks = self.processor._split_transcripts(test_docs)

        # Verify the chunks
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > len(test_docs))  # Should create more chunks
        self.assertTrue(all(len(chunk.page_content) <= 1000 for chunk in chunks))

    @patch('langchain_community.vectorstores.FAISS.from_documents')
    def test_process_videos(self, mock_faiss):
        """Test processing multiple videos."""
        # Mock FAISS database creation
        mock_faiss.return_value = Mock()

        # Create test video URLs
        test_urls = [
            "https://www.youtube.com/watch?v=test1",
            "https://www.youtube.com/watch?v=test2"
        ]

        # Mock the transcript loading
        with patch.object(
            self.processor,
            '_load_video_transcript',
            return_value=[Document(page_content="Test content")]
        ):
            # Process the videos
            self.processor.process_videos(test_urls)

            # Verify FAISS database was created
            mock_faiss.assert_called_once()

    def test_process_videos_no_transcripts(self):
        """Test handling when no transcripts are loaded."""
        # Mock transcript loading to return None (failed to load)
        with patch.object(self.processor, '_load_video_transcript', return_value=None):
            # Test with empty video list
            with self.assertRaises(ValueError):
                self.processor.process_videos(["https://www.youtube.com/watch?v=test"])

if __name__ == '__main__':
    unittest.main() 