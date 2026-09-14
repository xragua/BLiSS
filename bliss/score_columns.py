"""Read historical candidate tables using the canonical bliss_score schema."""
import pandas as pd


def normalize_score_columns(table):
    """Return a copy with canonical names, rejecting conflicting duplicate columns."""
    result = table.copy()
    for old, new in [('cluster_probability', 'bliss_score'),
                     ('pbliss_status', 'bliss_score_status')]:
        if old not in result:
            continue
        if new in result:
            equal = result[old].eq(result[new]) | (result[old].isna() & result[new].isna())
            if not equal.all():
                raise ValueError(f'Conflicting columns: {old} and {new}')
            result = result.drop(columns=old)
        else:
            result = result.rename(columns={old: new})
    return result


def read_bliss_csv(*args, **kwargs):
    """Read a full CSV and normalize historical score columns without rewriting it."""
    return normalize_score_columns(pd.read_csv(*args, **kwargs))
