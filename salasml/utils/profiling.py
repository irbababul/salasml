import json


def get_inferred_type_from_report(report, target_col):
    """
    Ambil kolom numerik dan kategorikal dari pandas-profiling report.

    == Parameter ==
    report    : pandas_profiling ProfileReport object
    target_col: nama kolom target yang ingin dikeluarkan dari hasil
    """
    profiles = json.loads(report.to_json())
    num_vars = []
    cat_vars = []
    for col, profile in profiles["variables"].items():
        if col == target_col:
            continue
        if profile["type"] == "Numeric":
            num_vars.append(col)
        elif profile["type"] == "Categorical":
            cat_vars.append(col)
        else:
            print(f"Unassigned: kolom {col} dengan tipe {profile['type']}")
    return num_vars, cat_vars
