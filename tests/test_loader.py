
from bliss.spectrum_data.text_spectrum_loader import load_text_spectrum

def test_loader_exists():
    assert callable(load_text_spectrum)
