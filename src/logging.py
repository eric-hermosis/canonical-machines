from typing import Sequence
from typing import Any
from pathlib import Path
from csv import DictWriter 
 
class Logger:
    def __init__(self, filepath: str, fieldnames: Sequence[str]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(path, "w", newline="")
        self.fieldnames = list(fieldnames)
        self.writer = DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader() 

    def log(self, **metrics: Any) -> None: 
        row = {
            key: value
            for key, value in metrics.items()
            if key in self.fieldnames
        }
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close() 