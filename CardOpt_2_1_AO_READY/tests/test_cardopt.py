import ast
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/"app.py").read_text()
TREE=ast.parse(SOURCE)

ASSIGNMENTS={
    "VERIFIED","CATS","DB","CASHLIKE","DEFAULT",
    "EXCLUDED_PFC_PREFIXES","GROCERY_KEYWORDS","DINING_KEYWORDS","DRUGSTORE_KEYWORDS",
    "AIRLINE_KEYWORDS","HOTEL_KEYWORDS","PORTAL_KEYWORDS","MCC_MAP",
    "OCR_DATE_PATTERNS","OCR_AMOUNT_RE","OCR_SKIP_WORDS"
}

nodes=[]
for node in TREE.body:
    if isinstance(node,ast.Import):
        keep=[a for a in node.names if a.name!="streamlit"]
        if keep:
            nodes.append(ast.Import(names=keep))
    elif isinstance(node,ast.ImportFrom):
        nodes.append(node)
    elif isinstance(node,ast.Assign):
        names=[t.id for t in node.targets if isinstance(t,ast.Name)]
        if any(n in ASSIGNMENTS for n in names):
            nodes.append(node)
    elif isinstance(node,ast.FunctionDef):
        nodes.append(node)

module_ast=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
core=types.ModuleType("cardopt_core_for_tests")
exec(compile(module_ast,"cardopt_core_for_tests","exec"),core.__dict__)

def zero_spend():
    return {k:0.0 for k in core.CATS}

def cpp():
    return {n:d["cpp"] for n,d in core.DB.items()}

def benefits():
    return {n:0.0 for n in core.DB}

def test_flat_rate_known_answer():
    s=zero_spend(); s["other"]=7000
    r=core.solve(s,cpp(),benefits(),1,"First-year recurring economics")
    assert abs(r["net"]-140)<1e-8

def test_dining_fee_tradeoff():
    s=zero_spend(); s["dining"]=10000
    r=core.solve(s,cpp(),benefits(),1,"First-year recurring economics")
    assert r["selected"]==["Chase Freedom Unlimited"]
    assert abs(r["net"]-300)<1e-8

def test_gold_supermarket_cap():
    s=zero_spend(); s["us_supermarkets"]=30000
    r=core.solve(s,cpp(),benefits(),2,"First-year recurring economics")
    assert abs(r["net"]-1275)<1e-8

def test_venture_x_credit_does_not_double_count_rewards():
    s=zero_spend(); s["portal_hotels"]=5000
    r=core.solve(
        s,cpp(),benefits(),1,"Ongoing annual economics",
        allowed_cards=["Capital One Venture X"],
        required_cards=["Capital One Venture X"],
    )
    assert abs(r["net"]-760)<1e-8

def test_sapphire_credit_does_not_double_count_rewards():
    s=zero_spend(); s["airfare_direct"]=1000
    r=core.solve(
        s,cpp(),benefits(),1,"First-year recurring economics",
        allowed_cards=["Chase Sapphire Reserve"],
        required_cards=["Chase Sapphire Reserve"],
    )
    assert abs(r["net"]+453)<1e-8

def test_classifier_known_categories():
    cases=[
        (("FOOD_AND_DRINK","FOOD_AND_DRINK_RESTAURANTS","Chipotle","5812","HIGH"),"dining"),
        (("FOOD_AND_DRINK","FOOD_AND_DRINK_GROCERIES","ShopRite","5411","VERY_HIGH"),"us_supermarkets"),
        (("TRAVEL","TRAVEL_FLIGHTS","United Airlines","4511","HIGH"),"airfare_direct"),
        (("TRAVEL","TRAVEL_LODGING","Marriott","7011","HIGH"),"hotels_direct"),
        (("MEDICAL","MEDICAL_PHARMACIES","CVS","5912","HIGH"),"drugstores"),
    ]
    for args, expected in cases:
        assert core.classify_transaction(*args)[0]==expected

def test_complexity_cost_can_change_wallet():
    s=zero_spend(); s["dining"]=10000; s["other"]=10000
    base=core.solve(s,cpp(),benefits(),3,"First-year recurring economics")
    friction=core.solve(s,cpp(),benefits(),3,"First-year recurring economics",complexity_cost=100)
    assert len(friction["selected"])<=len(base["selected"])
    assert friction["decision_utility"]<=friction["net"]+1e-8


def test_plaid_hosted_link_current_response_shape():
    payload={
        "link_sessions":[{
            "link_session_id":"session-1",
            "finished_at":"2026-09-27T12:00:00Z",
            "results":{"item_add_results":[{
                "public_token":"public-sandbox-test",
                "institution":{"name":"First Platypus Bank"},
                "accounts":[],
            }]},
        }]
    }
    status=core.parse_plaid_link_status(payload)
    assert status["public_tokens"]==["public-sandbox-test"]
    assert status["finished_count"]==1
    assert status["institutions"]==["First Platypus Bank"]


def test_ocr_text_parser_extracts_reviewable_rows():
    text="""
    09/20 Starbucks Coffee 12.45
    09/21 ShopRite 86.30
    Statement Balance 1,245.77
    09/22 CVS Pharmacy 18.10
    """
    df=core.ocr_text_to_transactions(text)
    assert len(df)==3
    assert set(round(x,2) for x in df["amount"].tolist())=={12.45,86.30,18.10}
    assert "Starbucks Coffee" in set(df["merchant"])
