"""Official PS3 output schemas from supplied examples and Info Kits."""

OFFICIAL_FILES = {
    "door": "door_predictions.csv", "acv": "acv_predictions.csv",
    "corrugation": "rail_predictions.csv", "shm": "shm_predictions.csv",
}
OFFICIAL_COLUMNS = {
    "door": ["start_time", "end_time", "prediction"],
    "acv": ["file_id", "ranked_cars"],
    "corrugation": ["file_id", "prediction"],
    "shm": ["file_id", "prediction"],
}

