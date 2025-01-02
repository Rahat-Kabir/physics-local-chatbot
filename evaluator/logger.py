import pandas as pd
import datetime
import time
import json
import re
from typing import Dict, List
class MetricsLogger:
    def __init__(self, log_file="metrics_log.json"):
        self.log_file = log_file
        self.metrics_history = []
        
    def log_metrics(self, metrics: Dict):
        metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.metrics_history.append(metrics)
        self._save_to_file()
        
    def _save_to_file(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_metrics_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_history)