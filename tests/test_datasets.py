import pytest
import pandas as pd

from datasets import get_filenames_for_this_stage

@pytest.mark.parametrize('input_data', [
    (pd.DataFrame({
        'img_filename':["a", "b", 'c'],
        'bbox_filename':["A", "B", "C"],
        'stage':[0, 1, 2]
    }), (["a", "b"], ['A', 'B'], [True, False]),  1)
])
def test_get_filenames_for_this_stage(input_data):
    input_df, expected_out, stage = input_data
    out = get_filenames_for_this_stage(input_df, stage)
    assert type(out) == tuple
    for observed, expected in zip(out, expected_out):
        assert observed == expected
