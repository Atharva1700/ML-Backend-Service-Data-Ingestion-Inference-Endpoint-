import json
from typing import List, Dict

class DataPipeline:
    @staticmethod
    def load_raw_data(file_path: str) -> List[Dict]:
        """Load raw data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def preprocess(raw_data: List[Dict]) -> List[Dict]:
        """Normalize and clean data for model input."""
        processed = []
        for record in raw_data:
            normalized = {
                'features': [float(x) / 255.0 for x in record.get('features', [])],
                'label': record.get('label'),
            }
            processed.append(normalized)
        return processed

def ingest_and_prepare(file_path: str) -> List[Dict]:
    raw = DataPipeline.load_raw_data(file_path)
    return DataPipeline.preprocess(raw)
