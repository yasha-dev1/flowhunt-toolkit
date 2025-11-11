"""Web content processing utilities for URLs and sitemaps."""

from pathlib import Path
from typing import List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import html2text
import tiktoken
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from .markdown_processor import MarkdownProcessor


class WebProcessor:
    """Handles web content extraction and token-based chunking using markdown splitting."""

    def __init__(self, max_tokens: int = 4000, overlap_tokens: int = 100, encoding_name: str = "cl100k_base"):
        """Initialize web processor.

        Args:
            max_tokens: Maximum tokens per chunk (default: 4000)
            overlap_tokens: Number of tokens to overlap between chunks (default: 100)
            encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4/ChatGPT)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # Don't wrap text
        self.markdown_processor = MarkdownProcessor(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            encoding_name=encoding_name
        )

    def fetch_url(self, url: str, timeout: int = 30) -> str:
        """Fetch content from a URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            HTML content from the URL

        Raises:
            Exception: If URL cannot be fetched
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; FlowHunt-Toolkit/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise Exception(f"Failed to fetch URL {url}: {str(e)}")

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract text content from HTML.

        Args:
            html_content: HTML content

        Returns:
            Markdown-formatted text extracted from HTML
        """
        # Convert HTML to markdown for better formatting
        text = self.html_converter.handle(html_content)
        return text.strip()

    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse a sitemap XML and extract URLs.

        Args:
            sitemap_url: URL of the sitemap.xml

        Returns:
            List of URLs found in the sitemap

        Raises:
            Exception: If sitemap cannot be parsed
        """
        try:
            sitemap_content = self.fetch_url(sitemap_url)

            # Parse XML
            root = ET.fromstring(sitemap_content)

            # Handle namespace
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            urls = []

            # Check if this is a sitemap index (contains other sitemaps)
            sitemap_elements = root.findall('.//ns:sitemap/ns:loc', namespace)
            if sitemap_elements:
                # This is a sitemap index, recursively parse child sitemaps
                for elem in sitemap_elements:
                    child_sitemap_url = elem.text
                    if child_sitemap_url:
                        child_urls = self.parse_sitemap(child_sitemap_url)
                        urls.extend(child_urls)
            else:
                # This is a regular sitemap with URLs
                url_elements = root.findall('.//ns:url/ns:loc', namespace)
                for elem in url_elements:
                    url = elem.text
                    if url:
                        urls.append(url)

            return urls
        except Exception as e:
            raise Exception(f"Failed to parse sitemap: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = None, url: str = None) -> List[Tuple[str, int]]:
        """Split markdown text into chunks based on headers and token count.

        Args:
            text: Markdown text to chunk (already converted from HTML)
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)
            url: Optional URL to use as source identifier

        Returns:
            List of (chunk_text, token_count) tuples
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Use MarkdownProcessor to chunk the text
        return self.markdown_processor.chunk_markdown(
            markdown_text=text,
            max_tokens=max_tokens,
            source_header=url
        )

    def process_url(self, url: str, max_tokens: int = None) -> List[Tuple[str, int]]:
        """Fetch URL content and chunk it based on token count.

        Args:
            url: URL to process
            max_tokens: Maximum tokens per chunk (uses instance default if not specified)

        Returns:
            List of (chunk_text, token_count) tuples
        """
        html_content = self.fetch_url(url)
        text = self.extract_text_from_html(html_content)
        return self.chunk_text(text, max_tokens, url=url)
