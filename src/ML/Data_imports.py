from ..data_treatment.data_loader import get_dataset
from ..config import CONFIG

DATASETS = [
    [
        symbol,
        get_dataset(
            symbol,
            StartDate=list(CONFIG.data.start_date),
            EndDate=list(CONFIG.data.end_date),
        ),
    ]
    for symbol in CONFIG.data.symbols
]