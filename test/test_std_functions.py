import std_functions
import pytest


@pytest.mark.skip("testing location_config")
def test_make_outputplot_filename():
    ofile = std_functions.make_outputplot_filename(0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert ofile == ""
