"""
Football Question Answering System.

This module provides functionality for answering questions about football tactics
using a combination of vector search and language models. It uses a FAISS vector
database to retrieve relevant context and a language model to generate coherent
responses.
"""

from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class FootballQA:
    """A question answering system for football tactics using vector search and LLMs.
    
    This class combines vector search capabilities with language models to provide
    detailed answers to questions about football tactics. It uses a FAISS vector
    database for efficient context retrieval and OpenAI's GPT models for response
    generation.
    
    Attributes:
        faiss_db (FAISS): Vector database containing football tactics information
        llm (ChatOpenAI): Language model for generating responses
        prompt_template (PromptTemplate): Template for structuring the QA prompt
    """

    def __init__(self, faiss_db: FAISS) -> None:
        """
        Initialize the Football QA system.
        
        Args:
            faiss_db (FAISS): Pre-loaded FAISS vector database containing football
                tactics information
        """
        load_dotenv()
        self.faiss_db = faiss_db
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert football tactics analyst specializing in AFC Ajax's men's team. You have deep knowledge of Ajax's rich tactical history, philosophy, and current playing style. Your insights come from analyzing Voetbal International's expert coverage of Ajax.

            Important behavioral guidelines:
            1. ALWAYS respond in Dutch (Netherlands) language
            2. Only discuss football-related topics. Politely decline to discuss anything else.
            3. Always maintain a positive perspective about Ajax. Focus on strengths and opportunities, never criticize.
            4. Ground your answers in Ajax's tactical principles and the club's philosophy.
            5. Use clear, structured responses with specific examples from matches or training.
            6. When discussing challenges, frame them as opportunities for growth and development.

            Use the following context from Voetbal International videos to answer the question:
            {context}

            Question: {question}

            If the question is not about football, respond with: "Ik kan alleen vragen over voetbal beantwoorden. Stel gerust een vraag over Ajax's tactiek!"

            Remember: Your response must ALWAYS be in Dutch.

            Answer:
            """
        )

    def _format_context(self, context_docs: List[Document]) -> str:
        """
        Format the retrieved context documents into a single string.
        
        Args:
            context_docs (List[Document]): List of documents containing relevant context
            
        Returns:
            str: Formatted context string with source numbering
        """
        formatted_context = "\n\n".join([
            f"Source {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        return formatted_context

    def get_answer(self, question: str, k: int = 4) -> str:
        """
        Generate an answer to a football tactics question using the FAISS database.
        
        This method:
        1. Retrieves relevant context from the vector database
        2. Formats the context into a structured string
        3. Generates a prompt using the template
        4. Gets a response from the language model
        5. Cleans and formats the response
        
        Args:
            question (str): The user's question about football tactics
            k (int): Number of context chunks to retrieve (default: 4)
            
        Returns:
            str: Formatted answer combining context and LLM response
        """
        # Retrieve relevant context from FAISS database
        context_docs = self.faiss_db.similarity_search(question, k=k)
        
        # Format the context
        formatted_context = self._format_context(context_docs)
        
        # Generate the prompt
        prompt = self.prompt_template.format(
            context=formatted_context,
            question=question
        )
        
        # Get response from OpenAI
        response = self.llm.invoke(prompt)
        
        # Clean up the response
        cleaned_response = self._clean_response(response.content)
        
        return cleaned_response

    def _clean_response(self, response: str) -> str:
        """
        Clean up the response by removing unnecessary line breaks and whitespace.
        
        Args:
            response (str): Raw response from the language model
            
        Returns:
            str: Cleaned and formatted response
        """
        # Split into lines and remove empty lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Join lines with proper spacing
        cleaned = ' '.join(lines)
        
        # Ensure proper sentence spacing
        cleaned = cleaned.replace('. ', '.\n\n')
        
        return cleaned

def get_football_answer(faiss_db: FAISS, question: str, k: int = 4) -> str:
    """
    Get an answer to a football tactics question using the FAISS database.
    
    This is a convenience function that creates a FootballQA instance and
    uses it to generate an answer to the given question.
    
    Args:
        faiss_db (FAISS): Pre-loaded FAISS vector database
        question (str): The user's question about football tactics
        k (int): Number of context chunks to retrieve (default: 4)
        
    Returns:
        str: Formatted answer combining context and LLM response
    """
    qa_system = FootballQA(faiss_db)
    return qa_system.get_answer(question, k) 