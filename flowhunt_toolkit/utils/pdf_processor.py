"""PDF processing utilities for extracting and chunking text."""

from pathlib import Path
from typing import List, Tuple
import tiktoken
from pypdf import PdfReader


class PDFProcessor:
    """Handles PDF text extraction and token-based chunking."""

    def __init__(self, max_tokens: int = 8000, encoding_name: str = "cl100k_base"):
        """Initialize PDF processor.

        Args:
            max_tokens: Maximum tokens per chunk (default: 8000)
            encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4/ChatGPT)
        """
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text from all pages

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF cannot be read
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = None, pdf_filename: str = None) -> List[Tuple[str, int]]:
        """Split text into chunks based on token count.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)
            pdf_filename: Optional PDF filename to prepend as H1 header to each chunk

        Returns:
            List of (chunk_text, token_count) tuples
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Calculate header token cost if filename is provided
        header = ""
        header_tokens = 0
        if pdf_filename:
            header = f"# {pdf_filename}\n\n"
            header_tokens = self.count_tokens(header)
            # Adjust max_tokens to account for header
            max_tokens = max_tokens - header_tokens

        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # If a single paragraph exceeds max_tokens, split it by sentences
            if para_tokens > max_tokens:
                # If we have accumulated content, save it first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if header:
                        chunk_text = header + chunk_text
                    chunks.append((chunk_text, current_tokens + header_tokens))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph into sentences
                sentences = para.replace('!', '.').replace('?', '.').split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sent_tokens = self.count_tokens(sentence + '.')

                    if current_tokens + sent_tokens > max_tokens and current_chunk:
                        # Save current chunk
                        chunk_text = '\n\n'.join(current_chunk)
                        if header:
                            chunk_text = header + chunk_text
                        chunks.append((chunk_text, current_tokens + header_tokens))
                        current_chunk = [sentence + '.']
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sentence + '.')
                        current_tokens += sent_tokens

            # Normal case: add paragraph to current chunk
            elif current_tokens + para_tokens <= max_tokens:
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if header:
                        chunk_text = header + chunk_text
                    chunks.append((chunk_text, current_tokens + header_tokens))

                current_chunk = [para]
                current_tokens = para_tokens

        # Add final chunk if any
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if header:
                chunk_text = header + chunk_text
            chunks.append((chunk_text, current_tokens + header_tokens))

        return chunks

    def process_pdf(self, pdf_path: Path, max_tokens: int = None) -> List[Tuple[str, int]]:
        """Extract text from PDF and chunk it based on token count.

        Args:
            pdf_path: Path to the PDF file
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)

        Returns:
            List of (chunk_text, token_count) tuples
        """
        text = self.extract_text(pdf_path)
        return self.chunk_text(text, max_tokens, pdf_filename=pdf_path.name)
