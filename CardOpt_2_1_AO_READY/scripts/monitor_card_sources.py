import json, re, sys
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"data"/"source_watch.json").read_text())
report={"review_required":False,"reviewed_baseline":cfg.get("reviewed"),"cards":[]}

headers={"User-Agent":"CardOptSourceMonitor/1.0 (+educational research project)"}
for card in cfg["cards"]:
    item={"name":card["name"],"url":card["url"],"status":"ok","http_status":None,"matched":[]}
    try:
        r=requests.get(card["url"],headers=headers,timeout=20,allow_redirects=True)
        item["http_status"]=r.status_code
        text=re.sub(r"\s+"," ",r.text).lower()
        matches=[term for term in card.get("expected_any",[]) if term.lower() in text]
        item["matched"]=matches
        if r.status_code>=400 or not matches:
            item["status"]="review"
            report["review_required"]=True
    except Exception as e:
        item["status"]="review"
        item["error"]=str(e)
        report["review_required"]=True
    report["cards"].append(item)

print(json.dumps(report,indent=2))
