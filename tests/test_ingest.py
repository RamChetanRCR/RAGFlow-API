import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from app.services.ingestor import Ingestor


def test_chunk_text_respects_token_limit():
    ingestor = Ingestor()
    ingestor.settings.chunk_size = 5
    ingestor.settings.chunk_overlap = 2
    text = "one two three four five\n\nsix seven eight nine ten\n\neleven twelve thirteen"
    chunks = ingestor.chunk_text(text)
    assert len(chunks) >= 2


def test_chunk_text_returns_single_for_small_text():
    ingestor = Ingestor()
    text = "Short paragraph."
    chunks = ingestor.chunk_text(text)
    assert len(chunks) == 1
    assert "Short paragraph." in chunks[0]


def test_parse_pdf_mocked():
    ingestor = Ingestor()
    mock_page_text = "Test page content"
    mock_page_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"text": "Section 1", "size": 18, "font": "Helvetica-Bold"}
                        ]
                    }
                ]
            }
        ]
    }

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.side_effect = [
            mock_page_text, mock_page_dict,
            mock_page_text, mock_page_dict,
        ]
        mock_doc.__iter__.return_value = [mock_page, mock_page]
        mock_doc.__len__.return_value = 2
        mock_open.return_value = mock_doc

        pages, count = ingestor.parse_pdf("fake.pdf")
        assert count == 2
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert pages[0]["section_header"] == "Section 1"
