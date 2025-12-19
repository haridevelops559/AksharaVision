from tinydb import TinyDB
import pandas as pd
import datetime

db = TinyDB("feedback_db.json")
CSV_FILE = "feedback_log.csv"

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "timestamp","top1","top2","top3","correct_label"
    ]).to_csv(CSV_FILE, index=False)

def save_feedback(t1, t2, t3, correct):
    ts = datetime.datetime.now().isoformat()
    db.insert({
        "timestamp": ts,
        "top1": t1,
        "top2": t2,
        "top3": t3,
        "correct_label": correct
    })
