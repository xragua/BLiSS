from io import StringIO
import pandas as pd
import pytest
from bliss import read_bliss_csv, normalize_score_columns, calculate_bliss_score


def test_historical_csv_is_normalized_without_changing_values():
    old = pd.DataFrame({'cluster_probability': [0., 0.8, 1., float('nan')],
                        'pbliss_status': ['evaluated'] * 4})
    actual = read_bliss_csv(StringIO(old.to_csv(index=False)))
    pd.testing.assert_series_equal(actual.bliss_score, old.cluster_probability, check_names=False)
    assert actual.bliss_score_status.tolist() == old.pbliss_status.tolist()
    assert 'cluster_probability' in old
    assert 'cluster_probability' not in actual
    pd.testing.assert_frame_equal(normalize_score_columns(actual), actual)


def test_conflicting_names_are_not_silently_accepted():
    with pytest.raises(ValueError, match='Conflicting'):
        normalize_score_columns(pd.DataFrame({'cluster_probability': [0.8], 'bliss_score': [0.2]}))


def test_score_calculation_retains_rate_excess_definition():
    assert calculate_bliss_score(10, 2) == pytest.approx(.8)
    assert calculate_bliss_score(0, 10) == 0
    assert calculate_bliss_score(10, 0) == 1
