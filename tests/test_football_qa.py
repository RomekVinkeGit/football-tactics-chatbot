"""
Unit tests for the FootballQA class and related functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_dir))

from llm.get_question_answer import FootballQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class TestFootballQA(unittest.TestCase):
    """Test cases for the FootballQA class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock FAISS database
        self.mock_faiss = Mock(spec=FAISS)
        self.qa = FootballQA(self.mock_faiss)

    def test_format_context(self):
        """Test the _format_context method."""
        # Create test documents
        test_docs = [
            Document(page_content="Test content 1"),
            Document(page_content="Test content 2")
        ]

        # Format the context
        formatted = self.qa._format_context(test_docs)

        # Check the formatting
        expected = "Source 1:\nTest content 1\n\nSource 2:\nTest content 2"
        self.assertEqual(formatted, expected)

    def test_clean_response(self):
        """Test the _clean_response method."""
        # Test input with multiple lines and spaces
        test_input = """
        This is a test response.
        It has multiple lines.
        And some extra spaces.   
        """

        cleaned = self.qa._clean_response(test_input)

        # Check that extra whitespace is removed and sentences are properly spaced
        expected = "This is a test response.\n\nIt has multiple lines.\n\nAnd some extra spaces."
        self.assertEqual(cleaned, expected)

    @patch('langchain_openai.ChatOpenAI')
    def test_get_answer(self, mock_chat):
        """Test the get_answer method."""
        # Mock the similarity search response
        self.mock_faiss.similarity_search.return_value = [
            Document(page_content="Ajax plays attacking football")
        ]

        # Mock the ChatOpenAI response
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = Mock(content="Test response about Ajax tactics")
        mock_chat.return_value = mock_chat_instance

        # Test getting an answer
        question = "How does Ajax play?"
        answer = self.qa.get_answer(question)

        # Verify the response is cleaned
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

        # Verify the FAISS database was queried
        self.mock_faiss.similarity_search.assert_called_once_with(question, k=4)

    def test_non_football_question(self):
        """Test handling of non-football questions."""
        # Mock the similarity search response
        self.mock_faiss.similarity_search.return_value = [
            Document(page_content="Some content")
        ]

        # Test with a non-football question
        question = "What is the weather like today?"
        answer = self.qa.get_answer(question)

        # Verify the response contains the Dutch message about football-only questions
        self.assertIn("Ik kan alleen vragen over voetbal beantwoorden", answer)

if __name__ == '__main__':
    unittest.main() 