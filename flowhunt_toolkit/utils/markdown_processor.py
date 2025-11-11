"""Markdown processing utilities for chunking text based on headers."""

from typing import List, Tuple, Optional
import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class MarkdownProcessor:
    """Handles markdown text chunking based on headers with token-based splitting."""

    def __init__(self, max_tokens: int = 4000, overlap_tokens: int = 100, encoding_name: str = "cl100k_base"):
        """Initialize Markdown processor.

        Args:
            max_tokens: Maximum tokens per chunk (default: 4000)
            overlap_tokens: Number of tokens to overlap between chunks (default: 100)
            encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4/ChatGPT)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

        # Define headers to split on (H1 and H2)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        # Initialize markdown header splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def _token_length_function(self, text: str) -> int:
        """Length function for RecursiveCharacterTextSplitter using tiktoken.

        Args:
            text: Text to measure

        Returns:
            Number of tokens
        """
        return self.count_tokens(text)

    def _get_overlap_text(self, text: str, from_end: bool = True) -> str:
        """Extract overlap text of approximately overlap_tokens length.

        Args:
            text: Text to extract overlap from
            from_end: If True, extract from end; if False, extract from start

        Returns:
            Overlap text
        """
        # Tokenize the text
        tokens = self.encoding.encode(text)

        if from_end:
            # Get last overlap_tokens tokens
            overlap_tokens = tokens[-self.overlap_tokens:] if len(tokens) > self.overlap_tokens else tokens
        else:
            # Get first overlap_tokens tokens
            overlap_tokens = tokens[:self.overlap_tokens] if len(tokens) > self.overlap_tokens else tokens

        # Decode back to text
        return self.encoding.decode(overlap_tokens)

    def _extract_h2_headers(self, text: str) -> List[str]:
        """Extract H2 headers from text.

        Args:
            text: Text to extract headers from

        Returns:
            List of H2 header strings (without the ## prefix)
        """
        headers = []
        for line in text.split('\n'):
            stripped = line.strip()
            # Match ## but not ###
            if stripped.startswith('## ') and not stripped.startswith('### '):
                # Remove the ## prefix and clean up
                header_text = stripped[3:].strip()
                if header_text:
                    headers.append(header_text)
        return headers

    def _create_h2_list_section(self, sections: List[str]) -> str:
        """Create a 'List of H2 Headings' section from the given sections.

        Args:
            sections: List of section texts to extract H2 headers from

        Returns:
            Formatted list section as markdown string
        """
        all_headers = []
        for section in sections:
            headers = self._extract_h2_headers(section)
            all_headers.extend(headers)

        if not all_headers:
            return ""

        # Build the list section
        list_lines = ["## List of H2 Headings", ""]
        for header in all_headers:
            list_lines.append(f"- {header}")
        list_lines.append("")  # Empty line after list

        return '\n'.join(list_lines)

    def chunk_markdown(
        self,
        markdown_text: str,
        max_tokens: Optional[int] = None,
        source_header: Optional[str] = None
    ) -> List[Tuple[str, int]]:
        """Split markdown text into chunks based on headers and token count.

        The chunking strategy:
        1. Split by H2 headers first
        2. Accumulate multiple H2 sections into a chunk until max_tokens is reached
        3. Each chunk gets the filename header
        4. If a single H2 section exceeds max_tokens, split it but preserve the H2 header
        5. Add overlap_tokens from the previous chunk to maintain context

        Args:
            markdown_text: Markdown text to chunk
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)
            source_header: Optional header (like filename/URL) to prepend to each chunk

        Returns:
            List of (chunk_text, token_count) tuples
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Prepare source header
        source_header_text = ""
        source_header_tokens = 0
        if source_header:
            source_header_text = f"# {source_header}\n\n"
            source_header_tokens = self.count_tokens(source_header_text)
            # Add to markdown if not already present
            if not markdown_text.strip().startswith(f"# {source_header}"):
                markdown_text = source_header_text + markdown_text

        # First pass: Split by markdown headers (H2 sections)
        try:
            md_header_splits = self.markdown_splitter.split_text(markdown_text)
        except Exception as e:
            # If header splitting fails, treat entire text as single section
            md_header_splits = [markdown_text]

        # Clean up sections and remove source header duplicates
        cleaned_sections = []
        for split in md_header_splits:
            # Handle both string and Document objects from LangChain
            if isinstance(split, str):
                section_text = split
            elif hasattr(split, 'page_content'):
                section_text = split.page_content
            else:
                section_text = str(split)

            # Remove source header from section if present
            if source_header_text:
                header_line = f"# {source_header}"
                section_lines = section_text.split('\n')
                cleaned_lines = []
                skip_empty_after_header = False
                for line in section_lines:
                    if line.strip() == header_line:
                        skip_empty_after_header = True
                        continue
                    if skip_empty_after_header and not line.strip():
                        skip_empty_after_header = False
                        continue
                    cleaned_lines.append(line)
                section_text = '\n'.join(cleaned_lines).strip()

            if section_text:  # Only add non-empty sections
                cleaned_sections.append(section_text)

        # Second pass: Accumulate sections into chunks up to max_tokens
        chunks = []
        previous_chunk_text = ""
        current_chunk_sections = []
        current_chunk_tokens = source_header_tokens

        for section_idx, section_text in enumerate(cleaned_sections):
            section_tokens = self.count_tokens(section_text)

            # Check if adding this section would exceed max_tokens
            # (accounting for source header and newlines between sections)
            section_with_separator_tokens = section_tokens + (2 if current_chunk_sections else 0)  # Add 2 tokens for "\n\n" separator

            if current_chunk_sections and (current_chunk_tokens + section_with_separator_tokens > max_tokens):
                # Finalize current chunk
                chunk_content = '\n\n'.join(current_chunk_sections)
                h2_list = self._create_h2_list_section(current_chunk_sections)

                # Build chunk with source header, H2 list, and content
                if source_header:
                    chunk_text = source_header_text + h2_list + "\n\n" + chunk_content
                else:
                    chunk_text = h2_list + "\n\n" + chunk_content if h2_list else chunk_content

                # Add overlap from previous chunk
                if previous_chunk_text and self.overlap_tokens > 0:
                    overlap_text = self._get_overlap_text(previous_chunk_text, from_end=True)
                    if source_header:
                        chunk_text = source_header_text + h2_list + "\n\n" + overlap_text + "\n\n" + chunk_content
                    else:
                        content_with_overlap = overlap_text + "\n\n" + chunk_content
                        chunk_text = h2_list + "\n\n" + content_with_overlap if h2_list else content_with_overlap

                chunk_tokens = self.count_tokens(chunk_text)
                chunks.append((chunk_text, chunk_tokens))
                previous_chunk_text = chunk_text

                # Start new chunk with current section
                current_chunk_sections = [section_text]
                current_chunk_tokens = source_header_tokens + section_tokens

            elif section_tokens > max_tokens - source_header_tokens:
                # This single section is too large, need to split it
                # First, finalize any accumulated chunk
                if current_chunk_sections:
                    chunk_content = '\n\n'.join(current_chunk_sections)
                    h2_list = self._create_h2_list_section(current_chunk_sections)

                    # Build chunk with source header, H2 list, and content
                    if source_header:
                        chunk_text = source_header_text + h2_list + "\n\n" + chunk_content
                    else:
                        chunk_text = h2_list + "\n\n" + chunk_content if h2_list else chunk_content

                    if previous_chunk_text and self.overlap_tokens > 0:
                        overlap_text = self._get_overlap_text(previous_chunk_text, from_end=True)
                        if source_header:
                            chunk_text = source_header_text + h2_list + "\n\n" + overlap_text + "\n\n" + chunk_content
                        else:
                            content_with_overlap = overlap_text + "\n\n" + chunk_content
                            chunk_text = h2_list + "\n\n" + content_with_overlap if h2_list else content_with_overlap

                    chunk_tokens = self.count_tokens(chunk_text)
                    chunks.append((chunk_text, chunk_tokens))
                    previous_chunk_text = chunk_text
                    current_chunk_sections = []
                    current_chunk_tokens = source_header_tokens

                # Now split this large section
                # Extract H2 header if present
                h2_header = ""
                h2_header_tokens = 0
                lines = section_text.split('\n')
                for line in lines:
                    if line.strip().startswith('## ') and not line.strip().startswith('### '):
                        h2_header = line.strip() + '\n\n'
                        h2_header_tokens = self.count_tokens(h2_header)
                        break

                # Remove H2 header from content
                content_without_h2 = section_text
                if h2_header:
                    content_without_h2 = '\n'.join([line for line in lines if not (line.strip().startswith('## ') and not line.strip().startswith('### '))])

                # Calculate available space
                subsection_available_tokens = max_tokens - source_header_tokens - h2_header_tokens

                # Split the large section
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=subsection_available_tokens,
                    chunk_overlap=self.overlap_tokens,
                    length_function=self._token_length_function,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )

                subsections = text_splitter.split_text(content_without_h2)

                for subsection_idx, subsection in enumerate(subsections):
                    # Build chunk: source header + H2 list + H2 header + content
                    chunk_parts = []
                    if source_header:
                        chunk_parts.append(source_header_text.rstrip())

                    # Create H2 list for this subsection (will contain the one H2 header)
                    if h2_header:
                        h2_list_for_subsection = self._create_h2_list_section([section_text])
                        if h2_list_for_subsection:
                            chunk_parts.append(h2_list_for_subsection.rstrip())
                        chunk_parts.append(h2_header.rstrip())

                    chunk_parts.append(subsection)

                    chunk_text = '\n\n'.join(chunk_parts)

                    # Add overlap from previous chunk
                    if subsection_idx == 0 and previous_chunk_text and self.overlap_tokens > 0:
                        overlap_text = self._get_overlap_text(previous_chunk_text, from_end=True)
                        header_part = '\n\n'.join(chunk_parts[:-1])
                        if header_part:
                            chunk_text = header_part + '\n\n' + overlap_text + '\n\n' + subsection
                        else:
                            chunk_text = overlap_text + '\n\n' + subsection

                    chunk_tokens = self.count_tokens(chunk_text)
                    chunks.append((chunk_text, chunk_tokens))
                    previous_chunk_text = chunk_text

            else:
                # Add section to current chunk
                current_chunk_sections.append(section_text)
                current_chunk_tokens += section_with_separator_tokens

        # Finalize any remaining chunk
        if current_chunk_sections:
            chunk_content = '\n\n'.join(current_chunk_sections)
            h2_list = self._create_h2_list_section(current_chunk_sections)

            # Build chunk with source header, H2 list, and content
            if source_header:
                chunk_text = source_header_text + h2_list + "\n\n" + chunk_content
            else:
                chunk_text = h2_list + "\n\n" + chunk_content if h2_list else chunk_content

            # Add overlap from previous chunk
            if previous_chunk_text and self.overlap_tokens > 0:
                overlap_text = self._get_overlap_text(previous_chunk_text, from_end=True)
                if source_header:
                    chunk_text = source_header_text + h2_list + "\n\n" + overlap_text + "\n\n" + chunk_content
                else:
                    content_with_overlap = overlap_text + "\n\n" + chunk_content
                    chunk_text = h2_list + "\n\n" + content_with_overlap if h2_list else content_with_overlap

            chunk_tokens = self.count_tokens(chunk_text)
            chunks.append((chunk_text, chunk_tokens))

        return chunks

    def process_markdown(
        self,
        markdown_text: str,
        max_tokens: Optional[int] = None,
        source_identifier: Optional[str] = None
    ) -> List[Tuple[str, int]]:
        """Process markdown text and return chunks.

        Args:
            markdown_text: Markdown text to process
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)
            source_identifier: Optional identifier (filename/URL) to use as header

        Returns:
            List of (chunk_text, token_count) tuples
        """
        return self.chunk_markdown(markdown_text, max_tokens, source_header=source_identifier)
