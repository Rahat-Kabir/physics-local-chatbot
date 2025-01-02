import pytest
from app import get_text_chunks

class TestPDFProcessing:
    def test_text_chunking(self):
        """Test text chunking logic"""
        # Provide longer text to ensure chunking works correctly
        sample_text = "This is a test paragraph. " * 100  # Ensure the text is long enough to create multiple chunks
        chunks = get_text_chunks(sample_text)
        assert len(chunks) > 1, f"Expected more than 1 chunk, but got {len(chunks)}"
        assert isinstance(chunks, list), "Chunks should be in a list format."
