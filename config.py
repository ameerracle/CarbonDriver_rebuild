"""
Configuration for the carbondriver project.
Simple configuration matching the original design.
"""

# Main configuration - matches the original Old_files/config.py
default_config = {
    "num_iter": 400,
    "make_plots": False,
    "normalize": False,
}

# Optional: Add a few essential settings for the rebuild
extended_config = {
    **default_config,
    "data_file": "Characterization_data.xlsx",
    "random_seed": 2,
    "device": "cpu",
}
