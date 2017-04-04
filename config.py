"""
Configuration settings. This contains folders and subject codes etc...:
"""

# Path is relative to the directory.....
_raw = 'storage/raw_data/'
_processed = 'storage/processed_data/'
_models = 'storage/models/'

# Get the subject list
with open('amuse_datasets') as f:
    _amuse_subjects = [s.strip() for s in f.readlines()]
