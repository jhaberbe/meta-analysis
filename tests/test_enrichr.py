import pytest
import pandas as pd
import requests
from unittest.mock import patch, Mock

from meta_analysis.go import GeneOntologyLibraries


# ----------------------------
# Test pull_library parsing
# ----------------------------
def test_pull_library_parses_response_correctly():
    fake_text = (
        "Pathway1\t\tGeneA\tGeneB\n"
        "Pathway2\t\tGeneC\tGeneD\n"
    )

    mock_response = Mock()
    mock_response.text = fake_text
    mock_response.raise_for_status = Mock()

    with patch("meta_analysis.go.requests.get", return_value=mock_response):
        gol = GeneOntologyLibraries()
        result = gol.pull_library("fake_library")

    assert result == {
        "Pathway1": ["GeneA", "GeneB"],
        "Pathway2": ["GeneC", "GeneD"],
    }


# ----------------------------
# Test HTTP error handling
# ----------------------------
def test_pull_library_http_error():
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("Bad request")

    with patch("meta_analysis.go.requests.get", return_value=mock_response):
        gol = GeneOntologyLibraries()

        with pytest.raises(requests.HTTPError):
            gol.pull_library("fake_library")


# ----------------------------
# Test assignment matrix output
# ----------------------------
def test_pull_assignment_matrix():
    fake_text = "Pathway1\t\tGeneA\tGeneB\n"

    mock_response = Mock()
    mock_response.text = fake_text
    mock_response.raise_for_status = Mock()

    with patch("meta_analysis.go.requests.get", return_value=mock_response):
        gol = GeneOntologyLibraries()
        df = gol.pull_assignment_matrix("fake_library")

    expected = pd.DataFrame({
        "Pathway1": {"GeneA": 1, "GeneB": 1}
    }).fillna(0)

    pd.testing.assert_frame_equal(df, expected)