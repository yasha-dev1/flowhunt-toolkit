"""DOCX processing utilities for extracting and chunking text."""

from pathlib import Path
from typing import List, Tuple
import tiktoken
from docling.document_converter import DocumentConverter
from .markdown_processor import MarkdownProcessor


class DOCXProcessor:
    """Handles DOCX text extraction and token-based chunking using docling and markdown splitting."""

    def __init__(self, max_tokens: int = 4000, overlap_tokens: int = 100, encoding_name: str = "cl100k_base"):
        """Initialize DOCX processor.

        Args:
            max_tokens: Maximum tokens per chunk (default: 4000)
            overlap_tokens: Number of tokens to overlap between chunks (default: 100)
            encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4/ChatGPT)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.markdown_processor = MarkdownProcessor(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            encoding_name=encoding_name
        )
        self.converter = DocumentConverter()

    def extract_text(self, docx_path: Path) -> str:
        """Extract text from a DOCX file and convert to markdown using docling.

        Args:
            docx_path: Path to the DOCX file

        Returns:
            Extracted text in markdown format

        Raises:
            FileNotFoundError: If DOCX file doesn't exist
            Exception: If DOCX cannot be read or converted
        """
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        try:
            # Convert DOCX to markdown using docling
            result = self.converter.convert(str(docx_path))
            markdown_content = result.document.export_to_markdown()
            return markdown_content

        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = None, docx_filename: str = None) -> List[Tuple[str, int]]:
        """Split markdown text into chunks based on headers and token count.

        Args:
            text: Markdown text to chunk
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)
            docx_filename: Optional DOCX filename to use as source identifier

        Returns:
            List of (chunk_text, token_count) tuples
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Use MarkdownProcessor to chunk the text
        return self.markdown_processor.chunk_markdown(
            markdown_text=text,
            max_tokens=max_tokens,
            source_header=docx_filename
        )

    def process_docx(self, docx_path: Path, max_tokens: int = None) -> List[Tuple[str, int]]:
        """Extract text from DOCX and chunk it based on token count.

        Args:
            docx_path: Path to the DOCX file
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)

        Returns:
            List of (chunk_text, token_count) tuples
        """
        text = self.extract_text(docx_path)
        return self.chunk_text(text, max_tokens, docx_filename=docx_path.name)
