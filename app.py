import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import io
import json
import secrets
from datetime import datetime, timedelta
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

st.set_page_config(page_title="CardOpt", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{
  color-scheme: light;
  --navy:#071a45;
  --text:#173052;
  --muted:#667a99;
  --blue:#0b6cff;
  --blue2:#267ff7;
  --violet:#5b43eb;
  --line:#d8e6f7;
  --soft:#f5f9ff;
  --panel:#ffffff;
  --shadow:0 14px 36px rgba(31,73,127,.08);
}
html,body,.stApp,[data-testid="stAppViewContainer"]{
  color-scheme:light!important;
  background:linear-gradient(180deg,#f8fbff 0%,#ffffff 38%,#f7fbff 100%)!important;
  color:var(--text)!important;
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif!important;
}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1240px!important;padding:1.25rem 1.75rem 5rem!important}
.main .block-container{max-width:1240px!important}
h1,h2,h3,h4,h5,h6{color:var(--navy)!important;letter-spacing:-.035em!important}
p,li,label,.stMarkdown,[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"]{color:var(--text)!important}
[data-testid="stCaptionContainer"],.stCaption,small{color:var(--muted)!important}
a{color:#0b62d5!important}
hr{border-color:#e2ebf6!important}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#f6f9ff 0%,#eef5ff 100%)!important;
  border-right:1px solid var(--line)!important;
  min-width:310px!important;
}
[data-testid="stSidebar"]>div:first-child{padding:1.45rem 1.35rem!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
.sidebar-brand{font-size:1.48rem;font-weight:850;letter-spacing:-.05em;color:var(--navy);margin-bottom:1rem}.sidebar-brand span{color:var(--blue)}
.sidebar-pref{display:grid;grid-template-columns:44px 1fr;gap:.75rem;align-items:center;padding:1rem;border:1px solid var(--line);border-radius:18px;background:#fff;box-shadow:0 8px 24px rgba(30,70,120,.05);margin-bottom:1rem}
.sidebar-pref .badge{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--blue),var(--violet));color:#fff!important;font-weight:800}
.sidebar-pref b{display:block;color:var(--navy)!important;font-size:1rem}.sidebar-pref span{display:block;color:var(--muted)!important;font-size:.78rem;margin-top:.1rem}

/* Native controls: force one coherent LIGHT theme even when device/browser is dark */
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,textarea{
  background:#fff!important;color:#10284d!important;-webkit-text-fill-color:#10284d!important;
  caret-color:#0b62d5!important;border-color:#bfd3ec!important;font-size:1rem!important
}
input::placeholder,textarea::placeholder{color:#7a8ba5!important;-webkit-text-fill-color:#7a8ba5!important;opacity:1!important}
[data-testid="stNumberInput"]{overflow:visible!important}
[data-testid="stNumberInput"]>div{background:#fff!important;border:1px solid #bfd3ec!important;border-radius:12px!important;overflow:hidden!important;min-height:48px!important}
[data-testid="stNumberInput"] input{border:0!important;box-shadow:none!important;padding:.65rem .8rem!important;min-width:0!important}
[data-testid="stNumberInput"] button{
  display:flex!important;visibility:visible!important;opacity:1!important;position:relative!important;
  width:46px!important;min-width:46px!important;height:46px!important;padding:0!important;
  align-items:center!important;justify-content:center!important;background:#f3f8ff!important;
  border:0!important;border-left:1px solid #d9e6f6!important;color:#075fd8!important;font-size:0!important
}
[data-testid="stNumberInput"] button svg,[data-testid="stNumberInput"] button span{display:none!important}
[data-testid="stNumberInput"] button:first-of-type::after{content:"−";font-size:22px!important;line-height:1!important;color:#075fd8!important;font-weight:700}
[data-testid="stNumberInput"] button:last-of-type::after{content:"+";font-size:22px!important;line-height:1!important;color:#075fd8!important;font-weight:700}

/* BaseWeb select/multiselect closed state */
[data-baseweb="select"],[data-baseweb="select"]>div,
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div{
  background:#fff!important;color:#10284d!important;border-color:#bfd3ec!important;border-radius:12px!important;min-height:48px!important
}
[data-baseweb="select"] *,[data-baseweb="select"] span,[data-baseweb="select"] div,[data-baseweb="select"] input,
[data-testid="stSelectbox"] *,[data-testid="stMultiSelect"] *{
  color:#10284d!important;-webkit-text-fill-color:#10284d!important;text-overflow:clip!important
}
[data-baseweb="select"] input::placeholder,[data-testid="stMultiSelect"] input::placeholder{color:#7487a5!important;-webkit-text-fill-color:#7487a5!important;opacity:1!important}
[data-baseweb="select"] svg{fill:#47678f!important;color:#47678f!important}
[data-baseweb="tag"]{height:auto!important;min-height:31px!important;max-width:100%!important;background:#edf5ff!important}
[data-baseweb="tag"] span{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;color:#173052!important}

/* BaseWeb dropdown popup: make it light too */
[data-baseweb="popover"]>div,[role="listbox"]{background:#fff!important;border:1px solid var(--line)!important;box-shadow:0 14px 34px rgba(24,51,91,.16)!important}
[role="listbox"] *,[role="option"],[role="option"] *{color:#10284d!important;-webkit-text-fill-color:#10284d!important}
[role="option"]{background:#fff!important;white-space:normal!important;min-height:44px!important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:#edf5ff!important;color:#0b54c6!important}

/* Buttons */
div.stButton>button,div.stDownloadButton>button{
  min-height:46px!important;height:auto!important;padding:.68rem .85rem!important;border-radius:12px!important;
  white-space:normal!important;overflow:visible!important;text-overflow:clip!important;line-height:1.2!important;
  font-weight:720!important;background:#fff!important;color:#15345f!important;border:1px solid #c9dbf2!important;
  box-shadow:0 5px 14px rgba(32,74,126,.05)!important
}
div.stButton>button *,div.stDownloadButton>button *{color:inherit!important;-webkit-text-fill-color:currentColor!important}
div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:#7ba9e9!important;background:#f7fbff!important;color:#0b58ca!important}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(100deg,#0879f9 0%,#236ef1 52%,#6045eb 100%)!important;color:#fff!important;border:0!important;
  box-shadow:0 10px 24px rgba(43,94,224,.2)!important
}
div.stButton>button[kind="primary"] *,div.stButton>button[data-testid="stBaseButton-primary"] *{color:#fff!important;-webkit-text-fill-color:#fff!important}

/* Radios, sliders, tabs */
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{color:var(--text)!important}
[data-testid="stSlider"] [role="slider"]{background:var(--blue)!important}
[data-testid="stTabs"] [role="tablist"]{gap:.3rem!important;background:#f1f6fd!important;padding:.35rem!important;border:1px solid #deebf8!important;border-radius:14px!important;flex-wrap:wrap!important}
[data-testid="stTabs"] button[role="tab"]{min-height:42px!important;padding:.55rem .9rem!important;border-radius:10px!important;color:#4f6385!important;white-space:normal!important}
[data-testid="stTabs"] button[role="tab"] p{color:#4f6385!important}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:#fff!important;box-shadow:0 3px 12px rgba(35,70,115,.08)!important}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:#0b5ed7!important;font-weight:750!important}

/* Containers, alerts, expanders, dataframes */
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;border-radius:22px!important;background:linear-gradient(145deg,#fff,#fbfdff)!important;box-shadow:var(--shadow)!important}
[data-testid="stVerticalBlockBorderWrapper"]>div{padding:1.25rem 1.3rem!important}
[data-testid="stExpander"]{background:#fff!important;border:1px solid var(--line)!important;border-radius:16px!important;overflow:hidden!important}
[data-testid="stExpander"] summary *{color:var(--navy)!important}
[data-testid="stAlert"]{border-radius:16px!important}
[data-testid="stAlert"] *{color:#253a5c!important}
[data-testid="stDataFrame"]{border-radius:15px!important;overflow:hidden!important;border:1px solid var(--line)!important}

/* Metrics: no ellipses */
[data-testid="stMetric"]{background:linear-gradient(145deg,#fff,#f8fbff)!important;border:1px solid var(--line)!important;border-radius:17px!important;padding:1rem!important;min-width:0!important;box-shadow:0 8px 22px rgba(35,72,116,.05)!important}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricLabel"]>div,[data-testid="stMetricValue"]>div,[data-testid="stMetric"] p{
  white-space:normal!important;overflow:visible!important;text-overflow:clip!important;max-width:none!important;overflow-wrap:anywhere!important
}
[data-testid="stMetricLabel"] p{color:#657895!important}[data-testid="stMetricValue"]{color:var(--navy)!important;font-size:clamp(1.45rem,2.4vw,2.25rem)!important;line-height:1.08!important}

/* Brand and landing hero */
.topbrand,.brand{font-size:1.55rem;font-weight:850;letter-spacing:-.05em;color:var(--navy)}.topbrand span,.brand span{color:var(--blue)}
.hero61{position:relative;overflow:hidden;padding:3.15rem 3rem;border:1px solid #d5e5f7;border-radius:30px;background:radial-gradient(circle at 84% 23%,rgba(63,141,255,.23),transparent 29%),linear-gradient(128deg,#fff 0%,#eef7ff 60%,#e6efff 100%);box-shadow:0 20px 52px rgba(28,73,126,.1)}
.hero61 .brand{margin-bottom:3rem}.eyebrow{font-size:.74rem;letter-spacing:.15em;text-transform:uppercase;color:#315f9b!important;font-weight:800}.hero61 h1{font-size:clamp(2.8rem,5vw,4rem)!important;line-height:1.01!important;letter-spacing:-.055em!important;margin:.55rem 0 .9rem!important;max-width:700px}.hero61 h1 span{color:#2868dc}.hero61 p{font-size:1.07rem;line-height:1.7;color:#536989!important;max-width:665px}.stats{display:flex;gap:0;margin-top:2rem;flex-wrap:wrap}.stat{padding-right:1.4rem;margin-right:1.4rem;border-right:1px solid #cad9eb}.stat:last-child{border:0}.stat b{display:block;color:var(--navy);font-size:.95rem}.stat small{color:#7183a0}.orbit{position:absolute;right:2.8rem;top:6.1rem;width:270px;height:215px}.cardshape{position:absolute;width:205px;height:126px;border-radius:18px;box-shadow:0 18px 35px rgba(16,35,75,.22);border:1px solid rgba(255,255,255,.55);padding:18px;color:#fff;font-weight:750;letter-spacing:.04em}.c1{right:32px;top:0;transform:rotate(8deg);background:linear-gradient(135deg,#071d47,#1762a4)}.c2{right:72px;top:60px;transform:rotate(-7deg);background:linear-gradient(135deg,#a67f35,#eed88f);color:#17203a}.c3{right:0;top:105px;transform:rotate(7deg);background:linear-gradient(135deg,#243d60,#0b1c34)}

/* Landing and general cards */
.section61{text-align:center;padding:3rem 0 1rem}.section61 h2{font-size:2.2rem;margin:.2rem 0}.section61 p{color:var(--muted)!important}.modecard{height:100%;min-height:300px;padding:1.7rem;border:1px solid var(--line);border-radius:24px;background:linear-gradient(145deg,#fff,#f6fbff);box-shadow:var(--shadow);overflow:hidden}.modecard.advanced{background:linear-gradient(145deg,#fff,#f8f5ff);border-color:#dfdaf6}.modeicon{width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:#e8f2ff;font-size:1.2rem;font-weight:800;color:#1769e0}.advanced .modeicon{background:#eee8ff;color:#6142da}.modecard h3{font-size:1.45rem!important;margin:.85rem 0 .12rem!important}.modecard .tag{font-weight:750;color:#2665cf}.modecard p{color:#5d6f8d!important;line-height:1.55}.ticks{line-height:1.9;color:#526884;font-size:.92rem}.pagehero{padding:2.2rem 2.3rem;border:1px solid #d5e5f7;border-radius:24px;background:radial-gradient(circle at 88% 20%,rgba(78,140,255,.17),transparent 28%),linear-gradient(135deg,#fff,#eef6ff);box-shadow:0 14px 38px rgba(34,75,125,.07);margin-bottom:1.2rem}.pagehero h1{font-size:clamp(2.1rem,4vw,3.35rem)!important;margin:.35rem 0 .6rem!important}.pagehero p{font-size:1.03rem;color:#566b8b!important;max-width:780px}.infoCard,.research-card{padding:1.25rem;border:1px solid var(--line);border-radius:19px;background:linear-gradient(145deg,#fff,#f9fcff);height:100%;box-shadow:0 9px 28px rgba(35,75,123,.055)}.infoCard h3,.research-card h3{margin:.3rem 0 .45rem!important}.infoCard p,.research-card p{color:#5e718f!important;line-height:1.58}.research-card{height:auto;margin:.7rem 0}.premium-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:14px 0 24px}.premium-pill{background:linear-gradient(145deg,#fff,#f6faff);border:1px solid var(--line);border-radius:17px;padding:15px;min-width:0;overflow:visible;box-shadow:0 8px 22px rgba(35,74,120,.045)}.premium-pill b{display:block;color:var(--navy);font-size:.96rem;margin-bottom:4px;white-space:normal;overflow:visible;text-overflow:clip}.premium-pill span{display:block;color:#657895;font-size:.83rem;line-height:1.45;white-space:normal;overflow:visible;text-overflow:clip}.note{font-size:.9rem;color:#667b99;border:1px solid #d8e6f7;background:#f7fbff;border-radius:14px;padding:.9rem 1rem;margin:.2rem 0 1rem}.source{padding:.9rem 1rem;border:1px solid var(--line);border-radius:14px;margin:.45rem 0;background:#fff}.callout{padding:1rem 1.1rem;border:1px solid var(--line);border-radius:16px;background:#f7fbff}.kpi{padding:1.1rem;border:1px solid var(--line);border-radius:18px;min-height:150px;background:#fff;box-shadow:0 8px 22px rgba(35,75,120,.05)}.kpi .label{font-size:.83rem;color:#687b98;line-height:1.35}.kpi .value{font-size:2rem;font-weight:780;letter-spacing:-.04em;margin:.3rem 0;color:var(--navy)}.kpi .sub{font-size:.78rem;color:#7b8aa2}.wallet{padding:1rem 1.1rem;border:1px solid var(--line);border-radius:16px;margin:.55rem 0;background:#fff}.wallet .name{font-size:1.04rem;font-weight:740;color:var(--navy)}.wallet .meta{font-size:.84rem;color:#7485a0;margin-top:.22rem}

/* Section headers used inside the working app */
.step-head{display:grid;grid-template-columns:46px 1fr;gap:.8rem;align-items:center;margin-bottom:.75rem}.step-head .stepnum{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--blue),var(--violet));color:#fff;font-weight:850;box-shadow:0 8px 18px rgba(40,99,225,.2)}.step-head b{display:block;color:var(--navy);font-size:1.35rem;letter-spacing:-.03em}.step-head span{display:block;color:var(--muted);font-size:.9rem;margin-top:.1rem}

/* Credit card visuals */
.creditcard{height:196px;border-radius:20px;padding:1.35rem;color:#fff;box-shadow:0 20px 42px rgba(12,32,68,.2);margin-bottom:.9rem;position:relative;overflow:hidden;border:1px solid rgba(255,255,255,.3);isolation:isolate}.creditcard:before{content:"";position:absolute;inset:0;background:linear-gradient(115deg,rgba(255,255,255,.18),transparent 34%,rgba(255,255,255,.04) 55%,transparent);z-index:-1}.creditcard:after{content:"";position:absolute;width:180px;height:180px;border-radius:50%;right:-70px;top:-80px;background:rgba(255,255,255,.08);z-index:-1}.creditcard .issuer{font-size:.7rem;opacity:.78;text-transform:uppercase;letter-spacing:.14em;font-weight:700}.creditcard .cardname{font-size:1.17rem;font-weight:780;margin-top:.35rem}.creditcard .chip{width:40px;height:29px;border-radius:6px;background:linear-gradient(135deg,#d9bf72,#f6e8ae);margin-top:2rem;border:1px solid rgba(110,80,20,.25)}.creditcard .fee{position:absolute;bottom:1rem;left:1.35rem;font-size:.78rem;opacity:.82;font-weight:600}.chase{background:linear-gradient(135deg,#06152f,#0b3c70 55%,#1479bb)}.amex{background:linear-gradient(135deg,#9a752e,#e2c776 48%,#f4e4aa);color:#17203a}.capone{background:linear-gradient(145deg,#0b1729,#173658 62%,#274e74)}.citi{background:linear-gradient(145deg,#12366f,#2d7cc0)}.freedom{background:linear-gradient(145deg,#184b83,#62a8dc)}.wells{background:linear-gradient(145deg,#651426,#b62b46)}

/* Learn */
.learn-hero{padding:2.35rem;border:1px solid #d3e5fa;border-radius:25px;margin-bottom:1.05rem;background:radial-gradient(circle at 88% 30%,rgba(87,73,238,.16),transparent 28%),linear-gradient(125deg,#fff,#eef7ff);box-shadow:0 16px 42px rgba(32,77,132,.08)}.learn-hero h1{font-size:clamp(2.15rem,4vw,3.3rem)!important;margin:.35rem 0 .55rem!important}.learn-hero p{max-width:760px;color:#5d718f!important;font-size:1.04rem;line-height:1.65}.learn-path{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:1rem 0 1.35rem}.learn-step{padding:1.15rem;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#fff,#f8fbff);box-shadow:0 8px 22px rgba(36,74,118,.045)}.learn-step .num{display:inline-flex;width:34px;height:34px;border-radius:50%;align-items:center;justify-content:center;background:#eaf3ff;color:#0b67e8;font-weight:800;margin-bottom:.65rem}.learn-step b{display:block;color:var(--navy);font-size:1.02rem;margin-bottom:.25rem}.learn-step span{display:block;color:#687b99;line-height:1.45}.lesson-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:.7rem 0 1rem}.lesson{padding:1.2rem 1.25rem;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#fff,#f8fbff)}.lesson .lesson-icon{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#edf5ff;color:#0b64df;font-size:1.05rem;font-weight:850;margin-bottom:.55rem}.lesson h3{font-size:1.05rem!important;margin:.15rem 0 .35rem!important}.lesson p{font-size:.94rem;line-height:1.55;color:#607391!important;margin:0!important}.learn-example{padding:1.15rem 1.25rem;border-radius:18px;border:1px solid #cfe1f7;background:linear-gradient(120deg,#eef7ff,#f7f4ff);margin:1rem 0}

/* Responsive */
@media(max-width:1050px){.block-container{padding-left:1.1rem!important;padding-right:1.1rem!important}.premium-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.orbit{opacity:.9;right:1.3rem}.hero61 h1{max-width:580px}}
@media(max-width:850px){.orbit{display:none}.hero61{padding:2rem 1.4rem}.hero61 .brand{margin-bottom:2rem}.premium-strip,.learn-path,.lesson-grid{grid-template-columns:1fr}.pagehero,.learn-hero{padding:1.55rem}.modecard{min-height:auto}[data-testid="stMetricValue"]{font-size:1.45rem!important}}
@media(max-width:640px){.block-container{padding-left:.8rem!important;padding-right:.8rem!important}.stats{gap:.75rem}.stat{border:0;margin-right:.4rem;padding-right:.4rem}.hero61 h1{font-size:2.45rem!important}[data-testid="stSidebar"]{min-width:275px!important}}

/* CardOpt 2.0: decision engine + real-spend analysis */
.data-source-banner{padding:1.1rem 1.2rem;border:1px solid #cfe1f7;border-radius:18px;background:linear-gradient(120deg,#f4f9ff,#f8f6ff);margin:.7rem 0 1rem}
.data-source-banner b{display:block;color:var(--navy)!important;font-size:1rem;margin-bottom:.2rem}
.data-source-banner span{color:var(--muted)!important;font-size:.9rem;line-height:1.5}
.insight-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:.85rem 0 1.15rem}
.insight-card{padding:1.15rem 1.2rem;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#fff,#f8fbff);box-shadow:0 8px 22px rgba(35,72,116,.045)}
.insight-card .kicker{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#5e74a0!important;font-weight:800}
.insight-card .big{font-size:1.55rem;line-height:1.1;font-weight:820;color:var(--navy)!important;margin:.35rem 0}
.insight-card p{font-size:.9rem;line-height:1.5;color:var(--muted)!important;margin:0}
.decision-banner{padding:1.35rem 1.4rem;border-radius:20px;border:1px solid #cfe0f7;background:radial-gradient(circle at 92% 20%,rgba(91,67,235,.12),transparent 30%),linear-gradient(120deg,#edf7ff,#faf8ff);margin:1rem 0}
.decision-banner h3{margin:0 0 .4rem!important}
.decision-banner p{margin:0!important;color:#5b6f90!important;line-height:1.55}
.status-good,.status-warn{display:inline-flex;padding:.28rem .58rem;border-radius:999px;font-size:.75rem;font-weight:800}
.status-good{background:#e9f8ef;color:#167344!important}.status-warn{background:#fff4df;color:#925c00!important}
.plaid-box{padding:1.15rem;border:1px solid #d6e5f7;border-radius:18px;background:#fff;margin:.75rem 0}
.plaid-box strong{color:var(--navy)!important}
.micro-note{font-size:.82rem;color:var(--muted)!important;line-height:1.5}
@media(max-width:850px){.insight-grid{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)

VERIFIED="2026-09-20"
CATS={
"dining":"Dining","us_supermarkets":"U.S. supermarkets",
"airfare_direct":"Flights booked directly","hotels_direct":"Hotels booked directly",
"portal_flights":"Flights via issuer portal","portal_hotels":"Hotels via issuer portal",
"drugstores":"Drugstores","other":"Everything else"}

DB={
"Chase Sapphire Reserve":{"fee":795,"cpp":1.50,
"rates":{"dining":3,"us_supermarkets":1,"airfare_direct":4,"hotels_direct":4,"portal_flights":8,"portal_hotels":8,"drugstores":1,"other":1},
"caps":{},"ann":0,
"auto_credit":{"name":"Annual travel credit","cap":300,"eligible":["airfare_direct","hotels_direct","portal_flights","portal_hotels"],"note":"Modeled only against eligible travel categories in CardOpt; credited spend earns no points."},
"benefits":[],
"source":"https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve"},
"American Express Gold":{"fee":325,"cpp":1.50,
"rates":{"dining":4,"us_supermarkets":4,"airfare_direct":3,"hotels_direct":1,"portal_flights":3,"portal_hotels":5,"drugstores":1,"other":1},
"caps":{"dining":50000,"us_supermarkets":25000},"ann":0,
"benefits":[("Dining credit",120,"Eligible partners; monthly limits and enrollment/terms apply."),
("Uber Cash",120,"U.S. Uber/Uber Eats; monthly limits and terms apply."),
("Resy credit",100,"Eligible U.S. Resy purchases; semiannual limits and terms apply."),
("Dunkin' credit",84,"Eligible U.S. Dunkin' purchases; monthly limits and enrollment/terms apply.")],
"auto_credit":None,"source":"https://www.americanexpress.com/us/credit-cards/card/gold-card/"},
"Capital One Venture X":{"fee":395,"cpp":1.50,
"rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,"portal_flights":5,"portal_hotels":10,"drugstores":2,"other":2},
"caps":{},"ann":10000,
"auto_credit":{"name":"Capital One Travel credit","cap":300,"eligible":["portal_flights","portal_hotels"],"note":"Applied only to modeled Capital One Travel spend; credited spend earns no miles."},
"benefits":[],
"source":"https://www.capitalone.com/credit-cards/venture-x/"},
"Citi Double Cash":{"fee":0,"cpp":1.00,
"rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,"portal_flights":2,"portal_hotels":5,"drugstores":2,"other":2},
"caps":{},"ann":0,"benefits":[],
"auto_credit":None,"source":"https://www.citi.com/credit-cards/citi-double-cash-credit-card"},
"Chase Freedom Unlimited":{"fee":0,"cpp":1.00,
"rates":{"dining":3,"us_supermarkets":1.5,"airfare_direct":1.5,"hotels_direct":1.5,"portal_flights":5,"portal_hotels":5,"drugstores":3,"other":1.5},
"caps":{},"ann":0,"benefits":[],
"auto_credit":None,"source":"https://creditcards.chase.com/cash-back-credit-cards/freedom/unlimited"},
"Wells Fargo Active Cash":{"fee":0,"cpp":1.00,"rates":{k:2 for k in CATS},"caps":{},"ann":0,"benefits":[],
"auto_credit":None,"source":"https://www.wellsfargo.com/credit-cards/active-cash/"}}

CASHLIKE={"Citi Double Cash","Chase Freedom Unlimited","Wells Fargo Active Cash"}
DEFAULT={"dining":6000,"us_supermarkets":5000,"airfare_direct":2500,"hotels_direct":1500,"portal_flights":500,"portal_hotels":500,"drugstores":1000,"other":8000}
def money(x): return f"${x:,.0f}"


# ---------- Data, Plaid, and decision-economics helpers ----------

EXCLUDED_PFC_PREFIXES = (
    "TRANSFER_", "INCOME", "LOAN_PAYMENTS", "BANK_FEES",
)

GROCERY_KEYWORDS = (
    "shoprite","whole foods","trader joe","kroger","publix","aldi","wegmans",
    "stop & shop","safeway","h mart","lidl","food lion","giant food","acme"
)
DINING_KEYWORDS = (
    "restaurant","cafe","coffee","pizza","grill","diner","doordash","uber eats",
    "grubhub","starbucks","chipotle","panera","mcdonald","taco bell","chick-fil-a"
)
DRUGSTORE_KEYWORDS = ("cvs","walgreens","rite aid","pharmacy","drugstore")
AIRLINE_KEYWORDS = (
    "airlines","airways","united","delta","american airlines","jetblue",
    "southwest","spirit airlines","frontier airlines"
)
HOTEL_KEYWORDS = (
    "hotel","hotels","marriott","hilton","hyatt","ihg","holiday inn","sheraton",
    "westin","ritz-carlton","hampton inn"
)
PORTAL_KEYWORDS = (
    "chase travel","capital one travel","amex travel","american express travel",
    "citi travel"
)

MCC_MAP = {
    "5812":"dining","5814":"dining",
    "5411":"us_supermarkets",
    "5912":"drugstores",
    "4511":"airfare_direct",
    "7011":"hotels_direct",
}

def safe_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

def plaid_is_configured():
    return bool(safe_secret("PLAID_CLIENT_ID") and safe_secret("PLAID_SECRET"))

def plaid_base_url():
    env=str(safe_secret("PLAID_ENV","sandbox")).lower().strip()
    return "https://production.plaid.com" if env=="production" else "https://sandbox.plaid.com"

def plaid_post(path, payload):
    client_id=safe_secret("PLAID_CLIENT_ID")
    secret=safe_secret("PLAID_SECRET")
    if not client_id or not secret:
        raise RuntimeError("Plaid credentials are not configured.")
    headers={
        "Content-Type":"application/json",
        "PLAID-CLIENT-ID":str(client_id),
        "PLAID-SECRET":str(secret),
    }
    r=requests.post(plaid_base_url()+path,headers=headers,json=payload,timeout=30)
    try:
        body=r.json()
    except Exception:
        body={"error_message":r.text}
    if r.status_code>=400:
        msg=body.get("error_message") or body.get("error_code") or f"Plaid request failed ({r.status_code})."
        raise RuntimeError(msg)
    return body

def create_plaid_hosted_link(client_user_id):
    payload={
        "client_name":"CardOpt",
        "country_codes":["US"],
        "language":"en",
        "user":{"client_user_id":client_user_id},
        "products":["transactions"],
        "transactions":{"days_requested":365},
        "hosted_link":{"url_lifetime_seconds":1800},
    }
    redirect_uri=safe_secret("PLAID_REDIRECT_URI")
    app_url=safe_secret("CARDOPT_APP_URL")
    webhook=safe_secret("PLAID_WEBHOOK_URL")
    if redirect_uri:
        payload["redirect_uri"]=str(redirect_uri)
    if app_url:
        payload["hosted_link"]["completion_redirect_uri"]=str(app_url)
    if webhook:
        payload["webhook"]=str(webhook)
    return plaid_post("/link/token/create",payload)

def get_public_tokens_from_link(link_token):
    data=plaid_post("/link/token/get",{"link_token":link_token})
    tokens=[]
    results=(data.get("results") or {})
    for item in results.get("item_add_results") or []:
        t=item.get("public_token")
        if t: tokens.append(t)
    if not tokens:
        old=(data.get("on_success") or {})
        t=old.get("public_token")
        if t: tokens.append(t)
    # Deduplicate while preserving order
    return list(dict.fromkeys(tokens))

def exchange_public_token(public_token):
    data=plaid_post("/item/public_token/exchange",{"public_token":public_token})
    return data["access_token"], data.get("item_id")

def sync_plaid_transactions(access_token, cursor=None):
    all_added=[]; all_modified=[]; removed=[]
    next_cursor=cursor
    for _ in range(20):
        payload={"access_token":access_token,"count":500}
        if next_cursor:
            payload["cursor"]=next_cursor
        data=plaid_post("/transactions/sync",payload)
        all_added.extend(data.get("added") or [])
        all_modified.extend(data.get("modified") or [])
        removed.extend(data.get("removed") or [])
        next_cursor=data.get("next_cursor")
        if not data.get("has_more"):
            break
    return all_added, all_modified, removed, next_cursor

def normalize_text(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return ""
    return str(x).strip().lower()

def classify_transaction(primary="", detailed="", merchant="", mcc="", plaid_confidence=None):
    primary_u=str(primary or "").upper()
    detailed_u=str(detailed or "").upper()
    merchant_l=normalize_text(merchant)
    mcc_s=str(mcc or "").strip().replace(".0","")
    joined=(primary_u+" "+detailed_u+" "+merchant_l).strip()

    if any(primary_u.startswith(x) for x in EXCLUDED_PFC_PREFIXES):
        return None,"Excluded","Non-spend / transfer-like Plaid category"

    # Explicit issuer portals first because otherwise a portal hotel/flight may look direct.
    if any(k in merchant_l for k in PORTAL_KEYWORDS):
        if "FLIGHT" in detailed_u or "AIR" in detailed_u or any(k in merchant_l for k in AIRLINE_KEYWORDS):
            return "portal_flights","High","Recognized issuer travel portal"
        if "HOTEL" in detailed_u or "LODG" in detailed_u or any(k in merchant_l for k in HOTEL_KEYWORDS):
            return "portal_hotels","High","Recognized issuer travel portal"
        return "portal_hotels","Medium","Issuer travel portal; category inferred as travel lodging"

    if mcc_s in MCC_MAP:
        return MCC_MAP[mcc_s],"High","Merchant category code mapping"

    if "GROC" in detailed_u or "SUPERMARKET" in detailed_u:
        cat="us_supermarkets"
    elif any(k in merchant_l for k in GROCERY_KEYWORDS):
        cat="us_supermarkets"
    elif "RESTAURANT" in detailed_u or "FAST_FOOD" in detailed_u or "COFFEE" in detailed_u or "BAR" in detailed_u:
        cat="dining"
    elif primary_u=="FOOD_AND_DRINK" or any(k in merchant_l for k in DINING_KEYWORDS):
        cat="dining"
    elif "PHARM" in detailed_u or any(k in merchant_l for k in DRUGSTORE_KEYWORDS):
        cat="drugstores"
    elif "FLIGHT" in detailed_u or "AIRLINE" in detailed_u or any(k in merchant_l for k in AIRLINE_KEYWORDS):
        cat="airfare_direct"
    elif "HOTEL" in detailed_u or "LODG" in detailed_u or any(k in merchant_l for k in HOTEL_KEYWORDS):
        cat="hotels_direct"
    else:
        cat="other"

    conf=str(plaid_confidence or "").upper()
    if conf in ("VERY_HIGH","HIGH"):
        level="High"
    elif conf=="MEDIUM":
        level="Medium"
    elif conf in ("LOW","UNKNOWN"):
        level="Low"
    else:
        # Heuristic mapping from non-Plaid CSV fields
        level="Medium" if cat!="other" else "Low"
    reason="Plaid personal-finance category / merchant heuristic" if (primary_u or detailed_u) else "Merchant-description heuristic"
    return cat,level,reason

def plaid_transactions_to_frame(transactions):
    rows=[]
    for t in transactions:
        pfc=t.get("personal_finance_category") or {}
        rows.append({
            "date":t.get("date") or t.get("authorized_date"),
            "amount":t.get("amount"),
            "merchant":t.get("merchant_name") or t.get("name") or "",
            "primary":pfc.get("primary",""),
            "detailed":pfc.get("detailed",""),
            "plaid_confidence":pfc.get("confidence_level",""),
            "mcc":t.get("merchant_category_code") or "",
            "payment_channel":t.get("payment_channel") or "",
            "transaction_id":t.get("transaction_id") or "",
        })
    return pd.DataFrame(rows)

def csv_to_normalized_frame(df, purchase_sign="positive"):
    # Flexible column discovery for common bank exports and Plaid-style CSVs.
    cols={str(c).lower().strip():c for c in df.columns}
    def find_col(*names):
        for n in names:
            if n in cols: return cols[n]
        for low,orig in cols.items():
            if any(n in low for n in names): return orig
        return None

    amount_col=find_col("amount","transaction amount","debit")
    if amount_col is None:
        raise ValueError("Could not find an amount column. Include a column named Amount.")
    date_col=find_col("date","transaction date","posted date","posting date")
    merchant_col=find_col("merchant_name","merchant","description","name","memo")
    primary_col=find_col("personal_finance_category.primary","primary category","primary")
    detailed_col=find_col("personal_finance_category.detailed","detailed category","detailed","category")
    conf_col=find_col("confidence_level","confidence")
    mcc_col=find_col("merchant_category_code","mcc")

    out=pd.DataFrame()
    out["amount"]=pd.to_numeric(df[amount_col],errors="coerce").fillna(0.0)
    if purchase_sign=="negative":
        out["amount"]=-out["amount"]
    out["date"]=pd.to_datetime(df[date_col],errors="coerce") if date_col is not None else pd.NaT
    out["merchant"]=df[merchant_col].astype(str) if merchant_col is not None else ""
    out["primary"]=df[primary_col].astype(str) if primary_col is not None else ""
    out["detailed"]=df[detailed_col].astype(str) if detailed_col is not None else ""
    out["plaid_confidence"]=df[conf_col].astype(str) if conf_col is not None else ""
    out["mcc"]=df[mcc_col].astype(str) if mcc_col is not None else ""
    return out

def build_spending_profile(df, annualize=False):
    if df is None or len(df)==0:
        return {k:0.0 for k in CATS}, pd.DataFrame(), {"rows":0,"mapped":0,"low":0,"days":0,"factor":1.0}
    work=df.copy()
    records=[]
    for _,row in work.iterrows():
        amount=float(row.get("amount",0) or 0)
        # Plaid uses positive amounts for outflows. Ignore zero/negative here; refunds/payments
        # should be reviewed rather than silently treated as purchases.
        if amount<=0: continue
        cat,confidence,reason=classify_transaction(
            row.get("primary",""),row.get("detailed",""),row.get("merchant",""),
            row.get("mcc",""),row.get("plaid_confidence","")
        )
        if cat is None: continue
        records.append({
            "Date":row.get("date"),
            "Merchant":row.get("merchant",""),
            "Amount":amount,
            "CardOpt category":CATS[cat],
            "Category key":cat,
            "Confidence":confidence,
            "Reason":reason,
        })
    mapped=pd.DataFrame(records)
    spend={k:0.0 for k in CATS}
    if len(mapped):
        for key,val in mapped.groupby("Category key")["Amount"].sum().items():
            if key in spend: spend[key]=float(val)
    dates=pd.to_datetime(work.get("date"),errors="coerce") if "date" in work else pd.Series(dtype="datetime64[ns]")
    valid_dates=dates.dropna()
    days=0
    if len(valid_dates)>=2:
        days=max(1,(valid_dates.max()-valid_dates.min()).days+1)
    factor=1.0
    if annualize and 30<=days<330:
        factor=365.0/days
        spend={k:v*factor for k,v in spend.items()}
    low=int((mapped["Confidence"]=="Low").sum()) if len(mapped) else 0
    stats={"rows":len(work),"mapped":len(mapped),"low":low,"days":days,"factor":factor}
    return spend,mapped,stats

def solve(spend,cpp,bens,maxcards,horizon, allowed_cards=None, required_cards=None,
          complexity_cost=0.0, switching_cost=0.0, existing_wallet=None, fee_overrides=None):
    """
    Mixed-integer linear program.

    X[i,j] = reward-earning spend in category j on card i at category rate
    Z[i,j] = post-cap spend in category j on card i at 1x
    D[i,j] = spend covered by a modeled statement/travel credit; earns no rewards
    Y[i]   = 1 when card i is carried, otherwise 0

    Optional economics terms:
      complexity_cost: user's annual friction value per additional card after the first
      switching_cost: user's annualized friction value for each add/drop vs current wallet
    """
    names=list(DB); cats=list(CATS); n=len(names); m=len(cats); q=n*m; N=3*q+n
    allowed_cards=set(names if allowed_cards is None else allowed_cards)
    required_cards=set([] if required_cards is None else required_cards)
    existing_wallet=set(existing_wallet or [])
    fee_overrides=fee_overrides or {}

    c=np.zeros(N); ub=np.full(N,np.inf); integ=np.zeros(N)
    X=lambda i,j:i*m+j
    Z=lambda i,j:q+i*m+j
    D=lambda i,j:2*q+i*m+j
    Y=lambda i:3*q+i

    for i,nm in enumerate(names):
        d=DB[nm]
        credit=d.get("auto_credit")
        for j,cat in enumerate(cats):
            c[X(i,j)]=-d["rates"][cat]*cpp[nm]/100
            c[Z(i,j)]=-cpp[nm]/100
            if cat not in d["caps"]: ub[Z(i,j)]=0
            if not credit or cat not in credit["eligible"]:
                ub[D(i,j)]=0
            else:
                c[D(i,j)]=-1.0
        ann=d["ann"]*cpp[nm]/100 if horizon=="Ongoing annual economics" else 0
        fee=float(fee_overrides.get(nm,d["fee"]))
        # Complexity penalty on every selected card differs from "extra cards" only by
        # a constant for any positive-spend feasible portfolio, so it preserves ranking.
        choice_cost=float(complexity_cost)
        if switching_cost:
            # symmetric-difference cost up to an irrelevant constant:
            # adding a new card is +cost; retaining an existing card avoids a drop cost.
            choice_cost += (-switching_cost if nm in existing_wallet else switching_cost)
        c[Y(i)]=fee-bens[nm]-ann+choice_cost+1e-5
        ub[Y(i)]=1 if nm in allowed_cards else 0
        integ[Y(i)]=1

    rows=[]; lo=[]; hi=[]
    for j,cat in enumerate(cats):
        r={}
        for i in range(n):
            r[X(i,j)]=1; r[Z(i,j)]=1; r[D(i,j)]=1
        rows.append(r); lo.append(spend[cat]); hi.append(spend[cat])

    M=max(sum(spend.values()),1)
    for i in range(n):
        r={Y(i):-M}
        for j in range(m):
            r[X(i,j)]=1; r[Z(i,j)]=1; r[D(i,j)]=1
        rows.append(r); lo.append(-np.inf); hi.append(0)

    for i,nm in enumerate(names):
        for cat,cap in DB[nm]["caps"].items():
            rows.append({X(i,cats.index(cat)):1,Y(i):-cap}); lo.append(-np.inf); hi.append(0)

    for i,nm in enumerate(names):
        credit=DB[nm].get("auto_credit")
        if credit:
            r={Y(i):-credit["cap"]}
            for cat in credit["eligible"]:
                r[D(i,cats.index(cat))]=1
            rows.append(r); lo.append(-np.inf); hi.append(0)

    rows.append({Y(i):1 for i in range(n)}); lo.append(-np.inf); hi.append(maxcards)
    for i,nm in enumerate(names):
        if nm in required_cards:
            rows.append({Y(i):1}); lo.append(1); hi.append(1)

    A=lil_matrix((len(rows),N))
    for rr,dct in enumerate(rows):
        for col,val in dct.items(): A[rr,col]=val

    res=milp(c,integrality=integ,bounds=Bounds(np.zeros(N),ub),
             constraints=LinearConstraint(A.tocsr(),np.array(lo),np.array(hi)))
    if not res.success:return None

    v=res.x
    selected=[names[i] for i in range(n) if v[Y(i)]>.5]
    alloc=[]; gross=0; credit_value=0
    cg={x:0 for x in names}; cs={x:0 for x in names}; cc={x:0 for x in names}
    for i,nm in enumerate(names):
        for j,cat in enumerate(cats):
            pieces=[
                (max(v[X(i,j)],0),DB[nm]["rates"][cat],"Reward earning"),
                (max(v[Z(i,j)],0),1,"Post-cap"),
                (max(v[D(i,j)],0),0,"Covered by credit")
            ]
            for amt,rate,tier in pieces:
                if amt>1e-6:
                    if tier=="Covered by credit":
                        val=0; credit_value+=amt; cc[nm]+=amt
                    else:
                        val=amt*rate*cpp[nm]/100; gross+=val; cg[nm]+=val
                    cs[nm]+=amt
                    alloc.append([CATS[cat],nm,amt,rate,cpp[nm],val,tier])

    fees=sum(float(fee_overrides.get(x,DB[x]["fee"])) for x in selected)
    manual_ben=sum(bens[x] for x in selected)
    ann=sum(DB[x]["ann"]*cpp[x]/100 for x in selected) if horizon=="Ongoing annual economics" else 0

    details=[]
    for x in selected:
        av=DB[x]["ann"]*cpp[x]/100 if horizon=="Ongoing annual economics" else 0
        total_ben=bens[x]+cc[x]
        fee=float(fee_overrides.get(x,DB[x]["fee"]))
        details.append({"Card":x,"Spend":cs[x],"Rewards":cg[x],"Benefits":total_ben,
                        "Anniversary":av,"Fee":fee,
                        "Net":cg[x]+total_ben+av-fee})

    total_ben=manual_ben+credit_value
    economic_net=gross+total_ben+ann-fees
    extra_cards=max(0,len(selected)-1)
    changes=len(set(selected).symmetric_difference(existing_wallet)) if existing_wallet else len([x for x in selected if x not in existing_wallet])
    complexity_penalty=float(complexity_cost)*extra_cards
    switching_penalty=float(switching_cost)*changes
    decision_utility=economic_net-complexity_penalty-switching_penalty
    return {"selected":selected,"allocation":alloc,"gross":gross,"fees":fees,
            "benefits":total_ben,"manual_benefits":manual_ben,"credits":credit_value,
            "anniversary":ann,"net":economic_net,"details":details,
            "complexity_penalty":complexity_penalty,"switching_penalty":switching_penalty,
            "decision_utility":decision_utility,"changes":changes}

def portfolio_frontier(spend,cpp,bens,horizon,allowed_cards,complexity_cost=0,switching_cost=0,existing_wallet=None):
    rows=[]
    prev=None
    for k in range(1,len(DB)+1):
        rr=solve(spend,cpp,bens,k,horizon,allowed_cards=allowed_cards,
                 complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=existing_wallet)
        if rr is None: continue
        marginal=np.nan if prev is None else rr["decision_utility"]-prev
        rows.append({
            "Maximum cards":k,
            "Economic net value":rr["net"],
            "Decision utility":rr["decision_utility"],
            "Marginal decision value":marginal,
            "Optimal wallet":", ".join(rr["selected"])
        })
        prev=rr["decision_utility"]
    return pd.DataFrame(rows)

def marginal_card_value(base_result,spend,cpp,bens,maxcards,horizon,allowed_cards,
                        complexity_cost=0,switching_cost=0,existing_wallet=None):
    rows=[]
    for card in base_result["selected"]:
        alt_allowed=[x for x in allowed_cards if x!=card]
        rr=solve(spend,cpp,bens,maxcards,horizon,allowed_cards=alt_allowed,
                 complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=existing_wallet)
        if rr is None: continue
        rows.append({
            "Card":card,
            "Marginal decision value":base_result["decision_utility"]-rr["decision_utility"],
            "Best wallet without card":", ".join(rr["selected"]),
            "Economic value without card":rr["net"],
        })
    return pd.DataFrame(rows)

def forced_alternative_analysis(base_result,spend,cpp,bens,maxcards,horizon,allowed_cards,
                                complexity_cost=0,switching_cost=0,existing_wallet=None):
    rows=[]
    for card in allowed_cards:
        if card in base_result["selected"]: continue
        rr=solve(spend,cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,required_cards=[card],
                 complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=existing_wallet)
        if rr is None: continue
        rows.append({
            "Card forced into wallet":card,
            "Opportunity cost":base_result["decision_utility"]-rr["decision_utility"],
            "Best wallet if forced":", ".join(rr["selected"]),
            "Economic net value":rr["net"],
        })
    return pd.DataFrame(rows).sort_values("Opportunity cost") if rows else pd.DataFrame()

def robustness_analysis(base_result,spend,cpp,bens,maxcards,horizon,allowed_cards,
                        complexity_cost=0,switching_cost=0,existing_wallet=None):
    scenarios=[]
    base_set=tuple(sorted(base_result["selected"]))
    def run(label, sspend, scpp, sbens):
        rr=solve(sspend,scpp,sbens,maxcards,horizon,allowed_cards=allowed_cards,
                 complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=existing_wallet)
        if rr:
            scenarios.append({
                "Scenario":label,
                "Decision utility":rr["decision_utility"],
                "Economic net value":rr["net"],
                "Wallet":", ".join(rr["selected"]),
                "Same portfolio":tuple(sorted(rr["selected"]))==base_set
            })
    run("Base",dict(spend),dict(cpp),dict(bens))
    for label,factor in [("Points -25%",.75),("Points +25%",1.25)]:
        scpp={n:(cpp[n] if n in CASHLIKE else max(.5,cpp[n]*factor)) for n in DB}
        run(label,dict(spend),scpp,dict(bens))
    for label,factor in [("Benefits -30%",.70),("Benefits +20%",1.20)]:
        sb={n:min(sum(x[1] for x in DB[n]["benefits"]),bens[n]*factor) for n in DB}
        run(label,dict(spend),dict(cpp),sb)
    for cat in CATS:
        for sign,factor in [("down",.8),("up",1.2)]:
            ss=dict(spend); ss[cat]=spend[cat]*factor
            run(f"{CATS[cat]} {sign} 20%",ss,dict(cpp),dict(bens))
    df=pd.DataFrame(scenarios)
    score=float(df["Same portfolio"].mean()*100) if len(df) else 0.0
    return score,df

def fee_decision_boundaries(base_result,spend,cpp,bens,maxcards,horizon,allowed_cards,
                            complexity_cost=0,switching_cost=0,existing_wallet=None):
    rows=[]
    selected=set(base_result["selected"])
    for card in allowed_cards:
        current=float(DB[card]["fee"])
        if card in selected:
            # Find the highest fee (up to +$2,000) where the card still remains selected.
            low=current; high=current+2000
            test=solve(spend,cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,
                       fee_overrides={card:high},complexity_cost=complexity_cost,
                       switching_cost=switching_cost,existing_wallet=existing_wallet)
            if test and card in test["selected"]:
                rows.append({"Card":card,"Boundary":"Fee headroom", "Threshold":high,
                             "Interpretation":f"Still selected even at {money(high)} annual fee in tested range."})
                continue
            for _ in range(10):
                mid=(low+high)/2
                rr=solve(spend,cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,
                         fee_overrides={card:mid},complexity_cost=complexity_cost,
                         switching_cost=switching_cost,existing_wallet=existing_wallet)
                if rr and card in rr["selected"]: low=mid
                else: high=mid
            rows.append({"Card":card,"Boundary":"Approx. max annual fee", "Threshold":low,
                         "Interpretation":f"Above roughly {money(low)}, another modeled wallet becomes preferable."})
        elif current>0:
            rr0=solve(spend,cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,
                      fee_overrides={card:0},complexity_cost=complexity_cost,
                      switching_cost=switching_cost,existing_wallet=existing_wallet)
            if rr0 and card in rr0["selected"]:
                low=0; high=current
                for _ in range(10):
                    mid=(low+high)/2
                    rr=solve(spend,cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,
                             fee_overrides={card:mid},complexity_cost=complexity_cost,
                             switching_cost=switching_cost,existing_wallet=existing_wallet)
                    if rr and card in rr["selected"]: low=mid
                    else: high=mid
                rows.append({"Card":card,"Boundary":"Approx. entry fee", "Threshold":low,
                             "Interpretation":f"At about {money(low)} annual fee or lower, this card enters the modeled optimum."})
    return pd.DataFrame(rows)

st.markdown("""<div class="hero61">
<div class="brand">Card<span>Opt</span></div>
<div class="eyebrow">Explainable credit-card decision intelligence</div>
<h1>Optimize your wallet.<br><span>Understand every tradeoff.</span></h1>
<p>CardOpt combines real or self-reported spending, mixed-integer optimization, and decision economics to find a high-value wallet and explain why the answer changes.</p>
<div class="stats">
<div class="stat"><b>6 verified cards</b><small>official issuer sources</small></div>
<div class="stat"><b>Real-spend ready</b><small>manual, CSV, or Plaid</small></div>
<div class="stat"><b>Decision economics</b><small>marginal value + opportunity cost</small></div>
</div>
<div class="orbit">
<div class="cardshape c1">SAPPHIRE RESERVE</div>
<div class="cardshape c2">AMERICAN EXPRESS GOLD</div>
<div class="cardshape c3">VENTURE X</div>
</div></div>""",unsafe_allow_html=True)


if "experience" not in st.session_state:
    st.session_state.experience = None
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.experience is None:
    n0,n1,n2,n3,n4,n5,n6,n7=st.columns([2.15,0.9,0.62,0.72,0.62,0.98,0.62,0.86])
    with n0: st.markdown('<div class="topbrand">Card<span>Opt</span></div>',unsafe_allow_html=True)
    with n1:
        if st.button("How It Works",use_container_width=True): st.session_state.page="How It Works"
    with n2:
        if st.button("Cards",use_container_width=True): st.session_state.page="Cards"
    with n3:
        if st.button("Research",use_container_width=True): st.session_state.page="Research"
    with n4:
        if st.button("Learn",use_container_width=True): st.session_state.page="Learn"
    with n5:
        if st.button("Math & Model",use_container_width=True): st.session_state.page="Math & Model"
    with n6:
        if st.button("About",use_container_width=True): st.session_state.page="About"
    with n7:
        if st.button("Get Started",type="primary",use_container_width=True): st.session_state.page="Home"

    if st.session_state.page == "How It Works":
        st.markdown('<div class="pagehero"><div class="eyebrow">How it works</div><h1>From spending to an optimized wallet.</h1><p>CardOpt turns a complicated card comparison into a structured optimization problem.</p></div>',unsafe_allow_html=True)
        aa,bb,cc=st.columns(3)
        steps=[("01","Tell us how you spend","Enter annual spending across eight categories and choose how many cards you are comfortable carrying."),
               ("02","Value benefits realistically","Tell CardOpt what restricted credits are actually worth to you. Face value is never assumed."),
               ("03","Optimize the whole wallet","The solver chooses both which cards belong in the wallet and where each category of spending should go.")]
        for col,item in zip([aa,bb,cc],steps):
            with col:
                st.markdown('<div class="infoCard"><div class="eyebrow">'+item[0]+'</div><h3>'+item[1]+'</h3><p>'+item[2]+'</p></div>',unsafe_allow_html=True)
        st.markdown("""<div class="research-card"><div class="eyebrow">WHY CARDOPT</div><h3>What makes this different</h3><p>Most card comparisons rank one card at a time. CardOpt looks at the wallet as a system. Fees, reward rates, spending caps, point values, benefits and wallet size interact, so changing one assumption can change the best overall portfolio.</p></div>""",unsafe_allow_html=True)
        st.markdown("""<div class="premium-strip"><div class="premium-pill"><b>Portfolio first</b><span>Optimizes the combination, not a single-card ranking.</span></div><div class="premium-pill"><b>Your constraints</b><span>Respects wallet size, fees, caps and booking preferences.</span></div><div class="premium-pill"><b>Explainable results</b><span>See assumptions, allocation and why cards were selected.</span></div></div>""",unsafe_allow_html=True)
        if st.button("Choose an experience",type="primary"): st.session_state.page="Home"; st.rerun()
        st.stop()

    if st.session_state.page == "Cards":
        st.markdown('<div class="pagehero"><div class="eyebrow">Card library</div><h1>The cards currently modeled.</h1><p>A deliberately focused research set with transparent terms.</p></div>',unsafe_allow_html=True) 
        st.markdown("""<div class="premium-strip"><div class="premium-pill"><b>6 researched cards</b><span>Focused set with transparent modeled terms.</span></div><div class="premium-pill"><b>Official sources</b><span>Issuer links are available inside each card.</span></div><div class="premium-pill"><b>Last reviewed</b><span>September 20, 2026.</span></div></div>""",unsafe_allow_html=True)
        styles={"Chase Sapphire Reserve":"chase","American Express Gold":"amex","Capital One Venture X":"capone","Citi Double Cash":"citi","Chase Freedom Unlimited":"freedom","Wells Fargo Active Cash":"wells"}
        issuers={"Chase Sapphire Reserve":"Chase","American Express Gold":"American Express","Capital One Venture X":"Capital One","Citi Double Cash":"Citi","Chase Freedom Unlimited":"Chase","Wells Fargo Active Cash":"Wells Fargo"}
        cardcols=st.columns(3)
        for ii,(nm,d) in enumerate(DB.items()):
            with cardcols[ii%3]:
                html='<div class="creditcard '+styles[nm]+'"><div class="issuer">'+issuers[nm]+'</div><div class="cardname">'+nm+'</div><div class="chip"></div><div class="fee">Annual fee $'+format(d["fee"],",")+'</div></div>'
                st.markdown(html,unsafe_allow_html=True)
                with st.expander("View modeled terms"):
                    st.write("Default model point value: **"+format(d["cpp"],".2f")+"¢**")
                    if nm=="Wells Fargo Active Cash":
                        st.success("Flat-rate structure: unlimited 2% cash rewards on purchases across all modeled categories.")
                    elif nm=="Citi Double Cash":
                        st.success("Flat-rate structure: 2% total cash back on purchases (1% when you buy + 1% as you pay), plus 5% total on eligible Citi Travel hotels in CardOpt's modeled categories.")
                    elif nm=="Chase Freedom Unlimited":
                        st.info("1.5% is the base rate, not a supermarket or direct-travel bonus. Dining and drugstores earn 3%; Chase Travel earns 5%.")
                    st.dataframe(pd.DataFrame([[CATS[k],(str(v)+"x" + (" base" if nm=="Chase Freedom Unlimited" and v==1.5 else ""))] for k,v in d["rates"].items()],columns=["Category","Modeled rate"]),hide_index=True,use_container_width=True)
                    if d.get("auto_credit"):
                        ac=d["auto_credit"]
                        st.write("Modeled automatic/eligible credit: **"+ac["name"]+" up to $"+format(ac["cap"],",")+"**")
                        st.caption(ac["note"])
                    if d["caps"]: st.write("Modeled caps: "+", ".join(CATS[k]+" $"+format(v,",") for k,v in d["caps"].items()))
                    if d["benefits"]:
                        st.write("Optional user-valued recurring benefits:")
                        for bn,face,note in d["benefits"]: st.write("• "+bn+": up to $"+format(face,","))
                    st.markdown("[Official issuer source]("+d["source"]+")")
        st.caption("Terms reviewed "+VERIFIED+". Issuer terms can change.")
        st.stop()

    if st.session_state.page == "Learn":
        st.markdown("""<div class="learn-hero">
        <div class="eyebrow">CARDOPT LEARN</div>
        <h1>Build confidence with credit.</h1>
        <p>Understand how credit cards, rewards and wallet decisions work before you optimize. Learn the concepts in plain English, then practice building a wallet that matches your priorities.</p>
        </div>""",unsafe_allow_html=True)

        st.markdown("""<div class="learn-path">
          <div class="learn-step"><div class="num">1</div><b>Credit foundations</b><span>Statements, APR, utilization and responsible card use.</span></div>
          <div class="learn-step"><div class="num">2</div><b>Rewards economics</b><span>Cash back, points, annual fees, caps and credits.</span></div>
          <div class="learn-step"><div class="num">3</div><b>Build your wallet</b><span>Turn what you learned into a simple wallet strategy.</span></div>
        </div>""",unsafe_allow_html=True)

        basics,rewards,economics,wallet=st.tabs(["Credit Card Basics","Rewards 101","Decision Economics","Build Your Wallet"])

        with basics:
            st.markdown("## Start with the fundamentals")
            st.write("Rewards come after responsible card use. These four ideas matter before any points calculation.")
            st.markdown("""<div class="lesson-grid">
              <div class="lesson"><div class="lesson-icon">▣</div><h3>Statement balance</h3><p>The amount shown on your statement for the billing cycle. Paying the statement balance in full by the due date is central to avoiding purchase interest when a grace period applies.</p></div>
              <div class="lesson"><div class="lesson-icon">%</div><h3>APR</h3><p>Annual Percentage Rate expresses borrowing cost on an annualized basis. Interest can quickly outweigh the value of credit-card rewards.</p></div>
              <div class="lesson"><div class="lesson-icon">◔</div><h3>Credit utilization</h3><p>The share of available revolving credit currently being used. It is different from a spending budget and can affect credit scoring.</p></div>
              <div class="lesson"><div class="lesson-icon">$</div><h3>Annual fee</h3><p>A yearly cost for holding a card. The useful question is whether the value you realistically receive justifies that cost for your own habits.</p></div>
            </div>""",unsafe_allow_html=True)
            st.info("CardOpt does not encourage opening more accounts or spending more. The optimizer works with the spending and constraints you provide.")

        with rewards:
            st.markdown("## Learn the economics behind rewards")
            st.write("Headline multipliers are only one part of a card's value.")
            st.markdown("""<div class="lesson-grid">
              <div class="lesson"><div class="lesson-icon">$</div><h3>Cash back</h3><p>A 2% cash-back rate means $2 of rewards for each $100 of eligible purchases, subject to issuer terms.</p></div>
              <div class="lesson"><div class="lesson-icon">×</div><h3>Points and miles</h3><p>A 4X card earns four points per eligible dollar. It is not automatically 4% back because the dollar value of a point depends on redemption.</p></div>
              <div class="lesson"><div class="lesson-icon">⌁</div><h3>Reward caps</h3><p>Some bonus categories stop earning the elevated rate after a threshold. CardOpt explicitly models relevant caps in its research set.</p></div>
              <div class="lesson"><div class="lesson-icon">◎</div><h3>Credits and benefits</h3><p>A $100 restricted credit is not necessarily worth $100 to you. CardOpt separates issuer face value from the value you realistically expect to use.</p></div>
              <div class="lesson"><div class="lesson-icon">◇</div><h3>Point valuation</h3><p>Transferable points do not have one guaranteed cash value. CardOpt exposes cents-per-point as a model assumption instead of presenting it as an issuer fact.</p></div>
              <div class="lesson"><div class="lesson-icon">Σ</div><h3>Net annual value</h3><p>CardOpt combines modeled reward value, benefits and applicable anniversary value, then subtracts annual fees to estimate recurring portfolio economics.</p></div>
            </div>""",unsafe_allow_html=True)
            st.markdown("""<div class="learn-example"><b>Why 4X is not automatically better than 3%</b><br><br>
            On $10,000 of dining, 4X points valued at 1.5 cents each produces $600 of modeled reward value. Subtract a $325 annual fee and the result is $275 before other benefits. A no-fee 3% card produces $300. CardOpt evaluates the economics, not just the biggest multiplier.</div>""",unsafe_allow_html=True)

        with economics:
            st.markdown("## Decision economics, not just rewards")
            st.write("CardOpt treats a wallet as an economic choice under constraints. These concepts explain why the highest headline reward rate is not always the best decision.")
            st.markdown("""<div class="lesson-grid">
              <div class="lesson"><div class="lesson-icon">Δ</div><h3>Marginal value</h3><p>How much better is the best portfolio with a card than the best portfolio without it? This measures what that card actually adds at the margin.</p></div>
              <div class="lesson"><div class="lesson-icon">↔</div><h3>Opportunity cost</h3><p>If you insist on a different card, what value do you give up versus the modeled optimum? CardOpt can force that choice and re-optimize everything else.</p></div>
              <div class="lesson"><div class="lesson-icon">≈</div><h3>Break-even boundary</h3><p>At what annual fee or other assumption does a card enter or leave the optimal portfolio? The answer is a decision boundary, not a universal ranking.</p></div>
              <div class="lesson"><div class="lesson-icon">↓</div><h3>Diminishing returns</h3><p>The second card may add substantial value while the fourth adds almost nothing. CardOpt's wallet frontier shows the incremental gain from allowing more complexity.</p></div>
              <div class="lesson"><div class="lesson-icon">⚖</div><h3>Utility and friction</h3><p>Two wallets can have similar dollar value but very different hassle. Advanced Mode can assign an explicit cost to extra cards and switching rather than pretending convenience is free.</p></div>
              <div class="lesson"><div class="lesson-icon">?</div><h3>Uncertainty</h3><p>Point values and future spending are uncertain. A recommendation is stronger when it survives reasonable changes in those assumptions.</p></div>
            </div>""",unsafe_allow_html=True)
            st.markdown("""<div class="learn-example"><b>Example: is a third card worth carrying?</b><br><br>
            If a one-card wallet is worth $610, two cards are worth $681, and three cards are worth $696, the third card adds only $15. The mathematical maximum is three cards, but the economic question is whether $15 is worth the extra complexity to you.</div>""",unsafe_allow_html=True)

        with wallet:
            st.markdown("## Build Your Wallet")
            st.write("Use this guided exercise to understand what kind of wallet structure is worth exploring. It teaches the decision logic; the actual CardOpt optimizer does the math.")
            q1,q2=st.columns(2)
            with q1:
                annual_fee=st.radio("Annual-fee preference",["Avoid annual fees while learning","Open to a fee when the math supports it"],key="learn_fee")
                travel=st.radio("How important are travel rewards?",["Not important","Somewhat important","Very important"],key="learn_travel")
            with q2:
                simplicity=st.radio("Wallet complexity",["Keep it simple","Comfortable using cards by category"],key="learn_complex")
                payoff=st.radio("Balance plan",["Pay the statement balance in full","I may carry a balance"],key="learn_payoff")

            st.markdown("### Your learning takeaway")
            if payoff=="I may carry a balance":
                st.error("Prioritize borrowing cost before rewards. Interest can outweigh rewards quickly. CardOpt's optimization intentionally does not model or recommend carrying debt.")
            elif annual_fee=="Avoid annual fees while learning" and simplicity=="Keep it simple":
                st.success("Start by understanding a simple no-annual-fee, flat-rate structure. Then compare whether category bonuses create enough additional value to justify more complexity.")
            elif travel=="Very important" and annual_fee=="Open to a fee when the math supports it":
                st.success("Explore travel-oriented cards, but evaluate annual fees, realistic credit usage, redemption value and booking restrictions together. A headline multiplier alone is not enough.")
            else:
                st.success("A small portfolio may be worth exploring: a strong everyday card plus a complementary category card can add value without making the wallet unnecessarily complex.")

            st.markdown("""<div class="learn-example"><b>Learning mode → optimization mode</b><br>
            Build Your Wallet gives you the framework. Simple Mode and Advanced Mode use CardOpt's mixed-integer optimization model to evaluate your actual modeled spending scenario.</div>""",unsafe_allow_html=True)
            if st.button("Use the CardOpt optimizer",type="primary",use_container_width=True,key="learn_to_home"):
                st.session_state.page="Home"; st.rerun()
        st.stop()

    if st.session_state.page == "Math & Model":
        st.markdown('<div class="pagehero"><div class="eyebrow">Math & Model</div><h1>The mathematics behind CardOpt.</h1><p>See the decision variables, objective function, constraints and assumptions used to turn a wallet decision into a mixed-integer linear program.</p></div>',unsafe_allow_html=True)
        st.markdown("""<div class="premium-strip">
        <div class="premium-pill"><b>Binary decisions</b><span>Should a card be in the wallet?</span></div>
        <div class="premium-pill"><b>Continuous decisions</b><span>How much category spend goes to each card?</span></div>
        <div class="premium-pill"><b>Linear optimization</b><span>Maximize modeled recurring net value under explicit constraints.</span></div>
        </div>""",unsafe_allow_html=True)

        st.markdown("### 1. Decision variables")
        st.latex(r"y_i \in \{0,1\}")
        st.write("**yᵢ** equals 1 when card *i* is selected and 0 otherwise.")
        st.latex(r"x_{ij} \ge 0")
        st.write("**xᵢⱼ** is reward-earning spending from category *j* assigned to card *i*.")
        st.latex(r"z_{ij} \ge 0")
        st.write("**zᵢⱼ** is spending above a modeled category cap. CardOpt models that overflow at the card's post-cap rate, currently 1x for the capped categories in this research set.")
        st.latex(r"d_{ij} \ge 0")
        st.write("**dᵢⱼ** is eligible spend covered by a modeled statement/travel credit. It contributes credit value but earns no rewards when issuer terms exclude rewards on credited spend.")

        st.markdown("### 2. Objective function")
        st.latex(r"\max \left[\sum_{i,j} r_{ij}v_i x_{ij}+\sum_{i,j} v_i z_{ij}+\sum_{i,j} d_{ij}+\sum_i b_i y_i+\sum_i a_i y_i-\sum_i f_i y_i\right]")
        st.write("Here **rᵢⱼ** is the reward multiplier, **vᵢ** is cents-per-point converted to dollars, **bᵢ** is user-valued recurring benefit value, **aᵢ** is applicable anniversary value, and **fᵢ** is the annual fee.")

        st.markdown("### 2A. Optional practical-utility layer")
        st.write("Advanced Mode can add explicit user-defined friction costs without mixing them into the issuer economics.")
        st.latex(r"U = V - \lambda\left(\sum_i y_i-1\right) - \gamma(A+D)")
        st.write("**V** is core economic net value, **λ** is the annual hassle value assigned to each additional card, and **γ** is the friction value assigned to each card added or removed relative to the current wallet. **A** and **D** count additions and drops. With positive spending, at least one card must be selected, so the complexity term remains linear up to an irrelevant constant.")

        st.markdown("### 2B. Real-spending data layer")
        st.write("Manual inputs can be replaced by transaction-derived category totals. Plaid or CSV records are classified into CardOpt categories using available personal-finance categories, merchant category codes, merchant names, and confidence labels. The user can review and override category totals before they enter the optimization model.")
        st.latex(r"\text{transactions} \rightarrow \text{classification} \rightarrow S_j \rightarrow \text{MILP}")

        st.markdown("### 3. Core constraints")
        st.write("**Spend conservation**: every modeled dollar must be assigned exactly once.")
        st.latex(r"\sum_i (x_{ij}+z_{ij}+d_{ij}) = S_j \quad \forall j")
        st.write("**Card activation**: spending can flow only to a selected card.")
        st.latex(r"\sum_j (x_{ij}+z_{ij}+d_{ij}) \le M y_i \quad \forall i")
        st.write("**Wallet size**: the number of selected cards cannot exceed the user's limit.")
        st.latex(r"\sum_i y_i \le K")
        st.write("**Reward caps**: bonus-rate spending is bounded by each issuer cap.")
        st.latex(r"x_{ij} \le C_{ij}y_i")
        st.write("**Credit caps**: credit-covered eligible spend cannot exceed the issuer's modeled annual credit.")
        st.latex(r"\sum_{j \in E_i}d_{ij} \le T_i y_i")

        st.markdown("### 4. Why this is a MILP")
        st.write("The model mixes binary card-selection variables with continuous spending-allocation variables, while keeping the objective and constraints linear. That is the defining structure of mixed-integer linear programming.")
        st.markdown("### 5. What is fact vs assumption?")
        st.markdown("""<div class="research-card"><p><b>Issuer facts</b>: annual fees, published earning rates, reward caps, anniversary rewards and eligible credits.<br><br>
        <b>User inputs</b>: annual spending, maximum wallet size and personal value assigned to restricted recurring benefits.<br><br>
        <b>Model assumptions</b>: cents-per-point values, the 2% comparison benchmark, excluded features and category mapping.<br><br>
        <b>Calculated outputs</b>: selected portfolio, spending allocation, reward value, modeled credits, fees and estimated net annual value.</p></div>""",unsafe_allow_html=True)
        st.info("CardOpt is a recurring-value model, not a forecast of realized returns. Welcome offers, interest, approval odds, taxes, credit-score effects, merchant coding uncertainty and transfer-partner availability are outside the optimization.")
        st.stop()

    if st.session_state.page == "Research":
        st.markdown('<div class="pagehero"><div class="eyebrow">Research</div><h1>Transparent by design.</h1><p>The optimization is useful only if its assumptions, constraints and data can be inspected.</p></div>',unsafe_allow_html=True)
        st.markdown("""<div class="premium-strip"><div class="premium-pill"><b>Issuer facts</b><span>Rates and fees sourced from official issuer materials.</span></div><div class="premium-pill"><b>Visible assumptions</b><span>Point values and user benefit values are never disguised as issuer facts.</span></div><div class="premium-pill"><b>Reproducible model</b><span>Objective, variables and constraints are documented in Math & Model.</span></div></div>""",unsafe_allow_html=True)
        st.markdown("""<div class="research-card"><h3>Optimization model</h3><p>CardOpt uses mixed-integer linear programming. Binary variables represent whether a card is selected. Continuous variables represent category-level spending allocated to each card.</p></div>""",unsafe_allow_html=True)
        st.markdown("""<div class="insight-grid">
        <div class="insight-card"><div class="kicker">Real behavior</div><div class="big">Manual + CSV + Plaid</div><p>Users can optimize estimates or transaction-derived spending, then review category mapping before it reaches the model.</p></div>
        <div class="insight-card"><div class="kicker">Decision economics</div><div class="big">Beyond rewards</div><p>Marginal value, opportunity cost, diminishing returns, decision boundaries, and explicit convenience frictions explain the recommendation.</p></div>
        <div class="insight-card"><div class="kicker">Uncertainty</div><div class="big">Robustness tested</div><p>CardOpt asks whether the same wallet survives changes in point values, benefit use, and category spending.</p></div>
        </div>""",unsafe_allow_html=True)
        st.markdown("""<div class="research-card"><h3>Data freshness architecture</h3><p>Card terms are versioned by review date and tied to official issuer sources. Automated source monitoring should flag possible term changes for human review rather than silently overwriting the live optimization database.</p></div>""",unsafe_allow_html=True)
        st.caption("Plaid transaction integration uses Hosted Link and the Transactions product when deployment credentials are configured. CardOpt's classifier does not claim that a Plaid category is identical to an issuer's final merchant coding.")
        st.latex(r"\max\; \text{reward value} + \text{user-valued benefits} + \text{anniversary value} - \text{annual fees}")
        r1,r2=st.columns(2)
        with r1:
            st.markdown("""<div class="research-card"><h3>Constraints</h3><p>Every modeled dollar is allocated. Spending flows only to selected cards. Reward caps are enforced. The portfolio respects the user's maximum wallet size.</p></div>""",unsafe_allow_html=True)
            st.markdown("""<div class="research-card"><h3>Assumption taxonomy</h3><p><b>Issuer Fact:</b> card terms and fees.<br><b>User Input:</b> spending and benefit values.<br><b>Model Assumption:</b> point valuations.<br><b>Calculated Result:</b> optimized wallet and estimated net value.</p></div>""",unsafe_allow_html=True)
        with r2:
            st.markdown("""<div class="research-card"><h3>Validation</h3><p>Known-answer tests cover cash-back arithmetic, annual-fee tradeoffs, multi-card routing, reward caps and overflow, benefit valuation, and first-year versus ongoing anniversary treatment.</p></div>""",unsafe_allow_html=True)
            st.markdown("""<div class="research-card"><h3>Deliberate exclusions</h3><p>Welcome offers, APR and interest, approval odds, credit-score effects, taxes, merchant-coding uncertainty, transfer award availability and unpriced qualitative perks are outside the recurring model.</p></div>""",unsafe_allow_html=True)
        st.stop()

    if st.session_state.page == "About":
        st.markdown('<div class="pagehero"><div class="eyebrow">About CardOpt</div><h1>What should actually be in your wallet?</h1><p>CardOpt is an explainable credit-card decision engine combining portfolio optimization, real-spending analysis, and economic tradeoff modeling.</p></div>',unsafe_allow_html=True)
        a1,a2=st.columns([1.25,1])
        with a1:
            st.markdown("""<div class="research-card"><h3>Why CardOpt exists</h3><p>Comparing individual rewards is relatively easy. The harder problem is deciding which cards belong in a wallet, how spending should be routed, what each additional card is actually worth, what convenience costs, and when the answer changes. CardOpt models those decisions together.</p></div>""",unsafe_allow_html=True)
            st.markdown("""<div class="research-card"><h3>Built by Sri Nihal Tammana</h3><p>CardOpt is an independent quantitative finance and optimization project exploring how mathematical modeling can make an everyday financial decision more transparent and easier to understand.</p></div>""",unsafe_allow_html=True)
        with a2:
            st.markdown('<div class="infoCard"><div class="eyebrow">Principles</div><h3>Useful, inspectable, honest.</h3><p>Keep the consumer experience simple. Expose the model for people who want depth. Separate issuer facts from assumptions. Validate the math with known-answer tests. Never treat restricted benefits as automatic cash value.</p></div>',unsafe_allow_html=True)
        st.info("CardOpt is an educational and research prototype, not individualized financial advice.")
        st.stop()

    st.markdown("""<div class="section61">
    <div class="eyebrow">Choose your experience</div>
    <h2>How would you like to use CardOpt?</h2>
    <p>The same optimization model, designed for different needs.</p></div>""", unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown("""<div class="modecard"><div class="modeicon">S</div><h3>Simple Mode</h3>
        <div class="tag">Just tell me what to use.</div>
        <p>Built for everyday users. Enter your spending and preferences, then get an optimized wallet and a clear estimate of your additional annual value.</p>
        <div class="ticks">✓ Quick and easy setup<br>✓ Clear card recommendations<br>✓ Estimated extra value</div></div>""",unsafe_allow_html=True)
        if st.button("Start Simple  →",type="primary",use_container_width=True):
            st.session_state.experience="Simple"; st.rerun()
    with c2:
        st.markdown("""<div class="modecard advanced"><div class="modeicon">A</div><h3>Advanced Mode</h3>
        <div class="tag">Show me the analysis.</div>
        <p>Explore portfolio economics, reward valuations, fees, benefit utilization, spending caps, sensitivity analysis, wallet complexity, assumptions, and methodology.</p>
        <div class="ticks">✓ Detailed results and charts<br>✓ Full model breakdown<br>✓ Assumptions and data sources</div></div>""",unsafe_allow_html=True)
        if st.button("Open Advanced  →",use_container_width=True):
            st.session_state.experience="Advanced"; st.rerun()
    st.info("Not sure which to choose? You can switch between Simple and Advanced Mode at any time.")
    st.markdown("""<div class="section61" style="padding-top:2rem">
    <div class="eyebrow">Beyond a rewards calculator</div>
    <h2>CardOpt explains the decision.</h2>
    <p>It is designed to answer not only which wallet wins, but what each card adds, what you give up, and when the answer changes.</p></div>""",unsafe_allow_html=True)
    st.markdown("""<div class="insight-grid">
    <div class="insight-card"><div class="kicker">Real spending</div><div class="big">Manual, CSV, or Plaid</div><p>Start from estimates or transaction-derived behavior, then review the category mapping before optimization.</p></div>
    <div class="insight-card"><div class="kicker">Economic tradeoffs</div><div class="big">Marginal value + opportunity cost</div><p>Measure what another card actually contributes and how much value is lost by forcing a different choice.</p></div>
    <div class="insight-card"><div class="kicker">Uncertainty</div><div class="big">Robustness + boundaries</div><p>Stress point values and spending, then estimate where annual-fee changes would flip the optimal portfolio.</p></div>
    </div>""",unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Card<span>Opt</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-pref"><div class="badge">1</div><div><b>Your preferences</b><span>Customize your analysis</span></div></div>',unsafe_allow_html=True)
    st.caption("Switch modes anytime without losing your inputs.")
    mode=st.radio("Experience",["Simple","Advanced"],index=0 if st.session_state.experience=="Simple" else 1,horizontal=True)
    if mode != st.session_state.experience:
        st.session_state.experience=mode
        st.rerun()
    if st.button("Back to welcome",use_container_width=True):
        st.session_state.experience=None
        st.rerun()
    st.caption("Simple is for everyday decisions. Advanced exposes the economics and model diagnostics.")
    st.divider()
    maxcards=st.slider("Maximum cards",1,6,3)
    benchmark=st.number_input("Cash-back comparison (%)",0.0,10.0,2.0,.1)/100
    horizon=st.radio("Time horizon",["Ongoing annual economics","First-year recurring economics"])
    st.caption("Ongoing includes applicable anniversary rewards. First-year recurring excludes rewards that begin after the first anniversary.")

with st.container(border=True):
    st.markdown('<div class="step-head"><div class="stepnum">2</div><div><b>Your analysis</b><span>Build a spending profile, then choose the assumptions CardOpt should use.</span></div></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Spending data","Benefits","Reward assumptions","Economics","Methodology"])

    with tabs[0]:
        st.subheader("Build your spending profile")
        st.markdown('<div class="data-source-banner"><b>Use estimates or real transactions.</b><span>Manual entry always remains available. Transaction imports are converted into CardOpt reward categories, then you can review and edit the totals before optimization.</span></div>',unsafe_allow_html=True)
        source_mode=st.radio(
            "Spending data source",
            ["Enter manually","Upload transactions","Connect with Plaid (Beta)"],
            horizontal=True,
            key="spending_source_mode"
        )

        spend={k:0.0 for k in CATS}

        if source_mode=="Enter manually":
            cols=st.columns(2)
            for i,(k,label) in enumerate(CATS.items()):
                with cols[i%2]:
                    spend[k]=st.number_input(label,0.0,value=float(DEFAULT[k]),step=500.0,format="%.0f",key="manual_"+k)
            st.caption("Manual mode is private and requires no account connection.")

        elif source_mode=="Upload transactions":
            st.write("Upload a CSV exported from a bank, credit-card account, budgeting tool, or Plaid-style dataset.")
            up=st.file_uploader("Transaction CSV",type=["csv"],key="transaction_csv")
            sign=st.radio("How are purchases shown in the Amount column?",["Positive amounts","Negative amounts"],horizontal=True,key="csv_sign")
            annualize_csv=st.checkbox("Annualize a partial-year history",value=True,key="csv_annualize",
                                     help="If the file covers 30 to 329 days, CardOpt scales observed category spend to a 365-day estimate.")
            if up is not None:
                try:
                    raw=pd.read_csv(up)
                    normalized=csv_to_normalized_frame(raw,"positive" if sign=="Positive amounts" else "negative")
                    profile,mapped,stats=build_spending_profile(normalized,annualize=annualize_csv)
                    st.markdown(f"""<div class="insight-grid">
                    <div class="insight-card"><div class="kicker">Transactions read</div><div class="big">{stats['rows']:,}</div><p>Rows found in the uploaded file.</p></div>
                    <div class="insight-card"><div class="kicker">Mapped purchases</div><div class="big">{stats['mapped']:,}</div><p>Positive purchase-like transactions used in the profile.</p></div>
                    <div class="insight-card"><div class="kicker">History window</div><div class="big">{stats['days'] or 'Unknown'} days</div><p>{'Annualized ×'+format(stats['factor'],'.2f') if stats['factor']!=1 else 'No annualization applied.'}</p></div>
                    </div>""",unsafe_allow_html=True)
                    if len(mapped):
                        low_pct=100*stats["low"]/max(1,len(mapped))
                        st.caption(f"Classification review: {stats['low']} low-confidence transactions ({low_pct:.1f}%). Merchant coding can differ from CardOpt's inferred reward category.")
                        with st.expander("Review classified transactions"):
                            preview=mapped.copy()
                            preview["Amount"]=preview["Amount"].map(lambda x:f"${x:,.2f}")
                            st.dataframe(preview[["Date","Merchant","Amount","CardOpt category","Confidence","Reason"]],use_container_width=True,hide_index=True)
                    st.markdown("#### Review annual category totals")
                    cols=st.columns(2)
                    for i,(k,label) in enumerate(CATS.items()):
                        with cols[i%2]:
                            spend[k]=st.number_input(label,0.0,value=float(round(profile[k],2)),step=100.0,format="%.0f",key="csv_"+k)
                except Exception as e:
                    st.error(f"Could not read that CSV: {e}")
            else:
                st.info("Upload a CSV to create a spending profile. You can switch to Manual at any time.")

        else:
            st.markdown("""<div class="plaid-box"><strong>Plaid connection</strong><br>
            CardOpt can use Plaid Hosted Link to let you authorize transaction access without entering bank credentials into CardOpt. The connection is optional and Manual/CSV modes remain available.</div>""",unsafe_allow_html=True)

            if "plaid_user_id" not in st.session_state:
                st.session_state.plaid_user_id="cardopt_"+secrets.token_hex(8)
            if "plaid_access_tokens" not in st.session_state:
                st.session_state.plaid_access_tokens=[]
            if "plaid_cursors" not in st.session_state:
                st.session_state.plaid_cursors={}
            if "plaid_transactions" not in st.session_state:
                st.session_state.plaid_transactions={}

            if not plaid_is_configured():
                st.warning("Plaid is not configured on this deployment yet. Add PLAID_CLIENT_ID and PLAID_SECRET to Streamlit Secrets to activate it.")
                st.code("""# .streamlit/secrets.toml
PLAID_CLIENT_ID = "..."
PLAID_SECRET = "..."
PLAID_ENV = "sandbox"   # change to production after approval
CARDOPT_APP_URL = "https://your-app.streamlit.app"
# Optional for production OAuth institutions:
PLAID_REDIRECT_URI = "https://your-approved-redirect.example.com"
""",language="toml")
                st.caption("For development, Plaid Sandbox uses mock data. Production uses real financial data and requires the appropriate Plaid access and configuration.")
            else:
                pc1,pc2=st.columns(2)
                with pc1:
                    if st.button("Create secure Plaid connection",use_container_width=True,key="plaid_create"):
                        try:
                            link=create_plaid_hosted_link(st.session_state.plaid_user_id)
                            st.session_state.plaid_link_token=link["link_token"]
                            st.session_state.plaid_hosted_url=link.get("hosted_link_url")
                        except Exception as e:
                            st.error(f"Plaid connection could not be created: {e}")
                with pc2:
                    if st.session_state.get("plaid_access_tokens"):
                        st.success(f"{len(st.session_state.plaid_access_tokens)} institution connection(s) active in this session.")

                if st.session_state.get("plaid_hosted_url"):
                    st.link_button("Open Plaid to connect an account",st.session_state.plaid_hosted_url,use_container_width=True)
                    st.caption("Complete the Plaid flow, return to CardOpt, then import the connection below.")
                    if st.button("I finished connecting. Import transactions",type="primary",use_container_width=True,key="plaid_finish"):
                        try:
                            tokens=get_public_tokens_from_link(st.session_state.plaid_link_token)
                            if not tokens:
                                st.info("Plaid has not reported a completed connection yet. Finish Link, then try again.")
                            else:
                                for pt in tokens:
                                    at,item_id=exchange_public_token(pt)
                                    if at not in st.session_state.plaid_access_tokens:
                                        st.session_state.plaid_access_tokens.append(at)
                                st.success("Connection authorized. Importing available transaction history.")
                        except Exception as e:
                            st.error(f"Could not finish Plaid connection: {e}")

                if st.session_state.get("plaid_access_tokens"):
                    if st.button("Refresh connected transactions",use_container_width=True,key="plaid_refresh"):
                        try:
                            for idx,at in enumerate(st.session_state.plaid_access_tokens):
                                cursor=st.session_state.plaid_cursors.get(str(idx))
                                added,modified,removed,next_cursor=sync_plaid_transactions(at,cursor)
                                for t in added+modified:
                                    if t.get("transaction_id"):
                                        st.session_state.plaid_transactions[t["transaction_id"]]=t
                                for rem in removed:
                                    rid=rem.get("transaction_id")
                                    if rid: st.session_state.plaid_transactions.pop(rid,None)
                                st.session_state.plaid_cursors[str(idx)]=next_cursor
                            if st.session_state.plaid_transactions:
                                st.success(f"Loaded {len(st.session_state.plaid_transactions):,} transaction records into this session.")
                            else:
                                st.info("Plaid has not returned historical transactions yet. Transaction history can become available after the institution finishes preparing it; use Refresh again later.")
                        except Exception as e:
                            st.error(f"Could not refresh Plaid transactions: {e}")

                    plaid_df=plaid_transactions_to_frame(list(st.session_state.plaid_transactions.values()))
                    annualize_plaid=st.checkbox("Annualize partial Plaid history",value=True,key="plaid_annualize")
                    profile,mapped,stats=build_spending_profile(plaid_df,annualize=annualize_plaid)
                    if len(mapped):
                        st.markdown(f"""<div class="insight-grid">
                        <div class="insight-card"><div class="kicker">Connected purchases</div><div class="big">{stats['mapped']:,}</div><p>Purchase-like transactions currently analyzed.</p></div>
                        <div class="insight-card"><div class="kicker">History window</div><div class="big">{stats['days'] or 'Unknown'} days</div><p>{'Annualized ×'+format(stats['factor'],'.2f') if stats['factor']!=1 else 'Observed totals used directly.'}</p></div>
                        <div class="insight-card"><div class="kicker">Low confidence</div><div class="big">{stats['low']:,}</div><p>Transactions worth reviewing before relying on category routing.</p></div>
                        </div>""",unsafe_allow_html=True)
                        with st.expander("Review Plaid classification"):
                            preview=mapped.copy(); preview["Amount"]=preview["Amount"].map(lambda x:f"${x:,.2f}")
                            st.dataframe(preview[["Date","Merchant","Amount","CardOpt category","Confidence","Reason"]],use_container_width=True,hide_index=True)
                    st.markdown("#### Review annual category totals")
                    cols=st.columns(2)
                    for i,(k,label) in enumerate(CATS.items()):
                        with cols[i%2]:
                            spend[k]=st.number_input(label,0.0,value=float(round(profile[k],2)),step=100.0,format="%.0f",key="plaid_"+k)

                    if st.button("Disconnect Plaid data from this session",key="plaid_disconnect"):
                        st.session_state.plaid_access_tokens=[]
                        st.session_state.plaid_cursors={}
                        st.session_state.plaid_transactions={}
                        st.session_state.pop("plaid_hosted_url",None)
                        st.session_state.pop("plaid_link_token",None)
                        st.rerun()

            st.caption("Privacy note: this prototype keeps Plaid access tokens only in the active Streamlit server session and does not intentionally write them to disk. A production service should use encrypted persistent token storage, deletion controls, webhook handling, and formal security review.")

        st.markdown(f"""<div class="learn-example" style="margin-top:1rem"><span style="color:#647696;font-size:.9rem">ANNUAL SPENDING USED BY THE MODEL</span><br><strong style="font-size:2.25rem;color:#0b55d9">{money(sum(spend.values()))}</strong></div>""",unsafe_allow_html=True)

    bens={}
    with tabs[1]:
        st.subheader("What are the benefits worth to you?")
        st.markdown('<div class="note">Restricted lifestyle credits are not automatically worth face value. Start at $0 and add only value you realistically expect to use. The Sapphire Reserve and Venture X travel credits are modeled separately against eligible planned travel spend so credited dollars do not also earn rewards.</div>',unsafe_allow_html=True)
        for nm,d in DB.items():
            with st.expander(nm,expanded=(nm=="Capital One Venture X")):
                total=0
                if not d["benefits"]: st.caption("No recurring lifestyle credit explicitly modeled here.")
                for bn,face,note in d["benefits"]:
                    val=st.slider(f"{bn} · up to {money(face)}",0,int(face),0,5,key=nm+bn)
                    st.caption(note); total+=val
                bens[nm]=total
                if d["benefits"]: st.write(f"Your modeled annual benefit value: **{money(total)}**")
    for n in DB:bens.setdefault(n,0)

    cpp={}
    with tabs[2]:
        st.subheader("Reward-value assumptions")
        st.markdown('<div class="note">Transferable points do not have one guaranteed cash value. CardOpt makes that judgment visible instead of hiding it.</div>',unsafe_allow_html=True)
        for nm,d in DB.items():
            cpp[nm]=st.number_input(f"{nm} · cents per point",.50,3.00,float(d["cpp"]),.05,key="c"+nm)
        st.info("Cash-like cards default to 1.00¢. Transferable currencies default to 1.50¢ as a model assumption, not an issuer guarantee.")

    with tabs[3]:
        st.subheader("Decision economics")
        st.markdown('<div class="data-source-banner"><b>Optimize value, not just rewards.</b><span>Advanced users can assign a small dollar cost to wallet complexity and switching. CardOpt then maximizes a practical decision-utility score while still reporting the underlying card economics separately.</span></div>',unsafe_allow_html=True)
        if mode=="Advanced":
            ec1,ec2=st.columns(2)
            with ec1:
                complexity_cost=st.number_input("Annual hassle cost per additional card",0.0,500.0,0.0,5.0,
                    help="A subjective dollar value for carrying/managing each card after the first.")
            with ec2:
                switching_cost=st.number_input("Friction cost per card added or removed",0.0,500.0,0.0,5.0,
                    help="A subjective annualized cost for changing your current wallet.")
            macro_spend_shock=st.slider("Broad nominal spending scenario for stress testing (%)",-20,20,0,1,
                help="Used only in Advanced stress analysis. This is not an inflation forecast.")
        else:
            complexity_cost=0.0; switching_cost=0.0; macro_spend_shock=0
            st.info("Simple Mode keeps these frictions at $0 so the recommendation stays easy to interpret. Advanced Mode lets you price convenience and switching.")
        st.markdown("""<div class="insight-grid">
        <div class="insight-card"><div class="kicker">Opportunity cost</div><div class="big">What you give up</div><p>Compare the optimal portfolio with the best portfolio forced to include another card.</p></div>
        <div class="insight-card"><div class="kicker">Marginal value</div><div class="big">What one card adds</div><p>Remove one selected card and re-optimize the rest of the wallet.</p></div>
        <div class="insight-card"><div class="kicker">Decision boundary</div><div class="big">When the answer flips</div><p>Estimate the annual-fee level at which a card enters or leaves the optimum.</p></div>
        </div>""",unsafe_allow_html=True)

    with tabs[4]:
        st.subheader("How CardOpt works")
        st.caption("For the full mathematical formulation, return to the welcome screen and open Math & Model.")
        st.write("CardOpt uses mixed-integer linear programming. Binary variables decide which cards enter the wallet; continuous variables decide how much spending in each category goes to each selected card.")
        st.markdown("**Core objective** · Maximize recurring economic value: reward value + user-valued recurring benefits + applicable anniversary value − annual fees.")
        st.markdown("**Optional decision-economics layer** · In Advanced Mode, user-defined complexity and switching costs can be added to the optimization objective without being confused with issuer economics.")
        st.markdown("**Constraints** · Allocate every dollar, route spend only to selected cards, enforce modeled category caps, credit eligibility, and maximum wallet size.")
        st.markdown("**Real-spend layer** · CSV and Plaid transactions are mapped into CardOpt reward categories using Plaid PFC/MCC fields and merchant heuristics. Users can review/edit the resulting category totals.")
        st.markdown("**Benchmark** · Compare the optimized wallet with putting the same spending on a simple cash-back card.")
        st.subheader("Validation record")
        st.write("Known-answer tests cover cash-back arithmetic, annual-fee tradeoffs, multi-card routing, category caps and post-cap overflow, travel-credit treatment, benefit valuation, first-year versus ongoing anniversary treatment, and decision-economics penalties.")
        st.subheader("Scope")
        st.write("Welcome offers, APR/interest, approval odds, credit-score effects, taxes, exact issuer merchant coding, transfer-partner award availability, and unpriced qualitative perks are intentionally excluded from the recurring optimization.")
        st.subheader("Official issuer sources")
        st.caption(f"Card terms reviewed {VERIFIED}. Terms can change.")
        for nm,d in DB.items():
            st.markdown(f'<div class="source"><b>{nm}</b><br><a href="{d["source"]}" target="_blank">Official issuer page ↗</a></div>',unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="step-head"><div class="stepnum">3</div><div><b>Make it yours</b><span>Choose how you actually want to use credit cards.</span></div></div>',unsafe_allow_html=True)
    q1,q2,q3=st.columns(3)
    with q1:
        fee_choice=st.selectbox("Annual fees",["Any fee if the math justifies it","No annual-fee cards only"])
    with q2:
        portal_choice=st.selectbox("Travel portals",["I am willing to use portals","I prefer booking direct"])
    with q3:
        reward_choice=st.selectbox("Reward style",["Maximize estimated value","Keep rewards simple"])

    st.markdown("#### CardOpt Compare")
    st.caption("Already have cards? Compare your current wallet with CardOpt's optimized wallet.")
    existing_wallet=st.multiselect("Cards I already carry",list(DB.keys()),default=[])

allowed_cards=list(DB.keys())
if fee_choice=="No annual-fee cards only":
    allowed_cards=[n for n in allowed_cards if DB[n]["fee"]==0]

effective_spend=dict(spend)
if portal_choice=="I prefer booking direct":
    effective_spend["airfare_direct"] += effective_spend["portal_flights"]
    effective_spend["hotels_direct"] += effective_spend["portal_hotels"]
    effective_spend["portal_flights"] = 0
    effective_spend["portal_hotels"] = 0

effective_cpp=dict(cpp)
if reward_choice=="Keep rewards simple":
    # Conservative cash-equivalent treatment for transferable currencies.
    for n in effective_cpp:
        if n not in CASHLIKE:
            effective_cpp[n]=1.0

with st.container(border=True):
    st.markdown('<div class="step-head"><div class="stepnum">4</div><div><b>Trust & transparency</b><span>See the data quality and assumptions behind the result.</span></div></div>',unsafe_allow_html=True)
    st.markdown(f"""<div class="premium-strip">
    <div class="premium-pill"><b>6 / 6 official sources</b><span>Every modeled card links to an official issuer source.</span></div>
    <div class="premium-pill"><b>$0 default benefit value</b><span>Restricted lifestyle benefits are valued only when you say you use them.</span></div>
    <div class="premium-pill"><b>{benchmark*100:.0f}% cash-back baseline</b><span>The comparison benchmark stays visible and editable.</span></div>
    <div class="premium-pill"><b>Reviewed {VERIFIED}</b><span>Issuer terms can change, so CardOpt shows its verification date.</span></div>
    </div>""",unsafe_allow_html=True)
    with st.expander("How CardOpt earns your trust"):
        st.write("• Card terms link to official issuer sources.")
        st.write("• Restricted credits start at $0 unless you say you would naturally use them.")
        st.write("• Point values are visible model assumptions.")
        st.write("• Welcome bonuses are excluded from recurring economics.")
        st.write("• Advanced Mode exposes allocations, assumptions, sensitivity and methodology.")
        st.write("• Estimates are not guarantees. Issuer terms, merchant coding and redemption values can change.")

with st.container(border=True):
    st.markdown('<div class="step-head"><div class="stepnum">5</div><div><b>Find your optimal wallet</b><span>Run the mathematical optimization model with your choices.</span></div></div>',unsafe_allow_html=True)
    run_optimizer=st.button("Optimize my wallet",type="primary",use_container_width=True)

if run_optimizer:
    total=sum(effective_spend.values())
    if total<=0: st.error("Enter at least some annual spending."); st.stop()
    r=solve(effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,
            complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=existing_wallet)
    if r is None:
        st.error("No feasible wallet matches those preferences. Try relaxing a filter.")
        st.stop()

    base=total*benchmark; adv=r["net"]-base

    if existing_wallet:
        current_allowed=[n for n in existing_wallet if n in DB]
        current=solve(
            effective_spend,effective_cpp,bens,len(current_allowed),horizon,
            allowed_cards=current_allowed,required_cards=current_allowed,
            complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=current_allowed
        )
        if current is not None:
            economic_improvement=r["net"]-current["net"]
            decision_improvement=r["decision_utility"]-current["decision_utility"]
            st.subheader("CardOpt Compare")
            ca,cb,cc=st.columns(3)
            ca.metric("Your current wallet",money(current["net"]))
            cb.metric("CardOpt economic value",money(r["net"]))
            if complexity_cost or switching_cost:
                cc.metric("Decision advantage",("+" if decision_improvement>=0 else "")+money(decision_improvement))
                st.caption(f"Raw economic-value difference: {('+' if economic_improvement>=0 else '')+money(economic_improvement)}. Decision advantage also reflects the convenience and switching costs you chose.")
                if decision_improvement>1:
                    st.success("After the friction values you entered, CardOpt still prefers the optimized portfolio.")
                elif decision_improvement>=-1:
                    st.info("Your current wallet and the optimized portfolio are effectively tied after your friction assumptions.")
                else:
                    st.info("Your current wallet has the higher practical decision utility under the friction assumptions you entered.")
            else:
                cc.metric("Potential improvement",("+" if economic_improvement>=0 else "")+money(economic_improvement))
                if economic_improvement>1:
                    st.success("Under your assumptions, CardOpt estimates about "+money(economic_improvement)+" more annual value than your current wallet.")
                elif economic_improvement>=-1:
                    st.info("Your current wallet is already very close to CardOpt's optimized result.")
                else:
                    st.info("Your current wallet performs strongly under these assumptions.")

    st.divider()
    if mode=="Simple":
        st.header("Your recommendation")
        st.markdown('<div class="note">The best modeled wallet for the spending and preferences you entered.</div>',unsafe_allow_html=True)
        a,b,c=st.columns(3)
        a.markdown(f'<div class="kpi"><div class="label">Your estimated annual value</div><div class="value">{money(r["net"])}</div><div class="sub">Rewards + benefits you value − annual fees</div></div>',unsafe_allow_html=True)
        b.markdown(f'<div class="kpi"><div class="label">If you just used a {benchmark*100:.0f}% cash-back card</div><div class="value">{money(base)}</div><div class="sub">Same spending, simpler strategy</div></div>',unsafe_allow_html=True)
        c.markdown(f'<div class="kpi"><div class="label">Estimated extra value from your optimized wallet</div><div class="value">{"+" if adv>=0 else ""}{money(adv)}</div><div class="sub">per year under your assumptions</div></div>',unsafe_allow_html=True)
        st.caption("Estimates include reward value, annual fees, and benefits you said you would use. Point values are assumptions, not guaranteed cash values.")
        st.subheader("Cards to carry")
        for d in r["details"]:
            st.markdown(f'<div class="wallet"><div class="name">{d["Card"]}</div><div class="meta">Annual fee {money(d["Fee"])} · estimated contribution {money(d["Net"])}</div></div>',unsafe_allow_html=True)
        st.subheader("Where to use each card")
        df=pd.DataFrame(r["allocation"],columns=["Spending category","Card","Annual spend","Multiplier","Point value (¢)","Estimated reward value","Rate tier"])
        simp=df.groupby(["Spending category","Card"],as_index=False)[["Annual spend","Estimated reward value"]].sum()
        simp["Annual spend"]=simp["Annual spend"].map(money); simp["Estimated reward value"]=simp["Estimated reward value"].map(money)
        st.dataframe(simp,use_container_width=True,hide_index=True)
        st.subheader("Why this wallet?")
        st.write("CardOpt compared rewards, annual fees, the benefits you said you would actually use, spending caps, and the number of cards you are comfortable carrying. It then selected the combination with the highest estimated net annual value.")
        with st.expander("Why not another card?"):
            omitted=[n for n in allowed_cards if n not in r["selected"]]
            if omitted:
                st.write("CardOpt can test this as an opportunity-cost question: what happens if an omitted card is forced into the wallet?")
                simple_alt=forced_alternative_analysis(
                    r,effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards,
                    complexity_cost,switching_cost,existing_wallet
                )
                if len(simple_alt):
                    for _,row in simple_alt.iterrows():
                        st.write(f"• **{row['Card forced into wallet']}**: forcing it into the portfolio reduces modeled decision value by about **{money(row['Opportunity cost'])}** versus the optimum.")
                else:
                    for n in omitted:
                        st.write("• **"+n+"** did not improve the modeled optimum.")
            else:
                st.write("All eligible modeled cards are in the optimized wallet.")
        st.info("For the assumptions, stress tests, allocation details and model diagnostics, switch to Advanced in the sidebar and run the same inputs.")
    else:
        st.header("Advanced decision analysis")
        if complexity_cost or switching_cost:
            m=st.columns(4)
            m[0].metric("Economic net value",money(r["net"]))
            m[1].metric("Decision utility",money(r["decision_utility"]))
            m[2].metric(f"{benchmark*100:.1f}% benchmark",money(base))
            m[3].metric("Economic advantage",f"{'+' if adv>=0 else ''}{money(adv)}")
            st.caption("Decision utility = card economics minus the convenience/switching costs you explicitly chose. It is a preference score, not an issuer reward value.")
        else:
            m=st.columns(4)
            m[0].metric("Estimated net value",money(r["net"]))
            m[1].metric("Net-value rate",f"{r['net']/total*100:.2f}%")
            m[2].metric(f"{benchmark*100:.1f}% benchmark",money(base))
            m[3].metric("Modeled advantage",f"{'+' if adv>=0 else ''}{money(adv)}")
            st.caption("Net-value rate includes modeled benefits and fees. It is not an issuer-advertised reward rate.")

        st.markdown("""<div class="decision-banner"><h3>CardOpt Decision Engine</h3>
        <p>The optimizer is only the first layer. The analysis below asks what each card adds at the margin, what you give up by forcing a different card, how much value another card creates, and whether the recommendation survives changes in assumptions.</p></div>""",unsafe_allow_html=True)

        st.subheader("Portfolio economics")
        ddf=pd.DataFrame(r["details"])
        display=ddf.copy()
        for col in ["Spend","Rewards","Benefits","Anniversary","Fee","Net"]:
            display[col]=display[col].map(money)
        st.dataframe(display,use_container_width=True,hide_index=True)

        df=pd.DataFrame(r["allocation"],columns=["Spending category","Card","Annual spend","Multiplier","Point value (¢)","Estimated reward value","Rate tier"])
        st.subheader("Category allocation")
        st.dataframe(df,use_container_width=True,hide_index=True)

        st.subheader("Economic bridge")
        bridge=pd.DataFrame({
            "Component":["Reward value","Modeled benefits / credits","Anniversary value","Annual fees","Economic net value"],
            "Value":[r["gross"],r["benefits"],r["anniversary"],-r["fees"],r["net"]]
        })
        if complexity_cost or switching_cost:
            bridge=pd.concat([bridge,pd.DataFrame({
                "Component":["Complexity friction","Switching friction","Decision utility"],
                "Value":[-r["complexity_penalty"],-r["switching_penalty"],r["decision_utility"]]
            })],ignore_index=True)
        st.dataframe(bridge.assign(Value=bridge["Value"].map(money)),use_container_width=True,hide_index=True)
        st.bar_chart(bridge.set_index("Component")["Value"],horizontal=True)

        st.subheader("Economics of your wallet")
        marg=marginal_card_value(
            r,effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards,
            complexity_cost,switching_cost,existing_wallet
        )
        if len(marg):
            st.markdown("#### Marginal value of each selected card")
            st.caption("CardOpt removes one selected card, re-optimizes the remaining eligible cards, and measures what the original portfolio loses.")
            shown=marg.copy()
            shown["Marginal decision value"]=shown["Marginal decision value"].map(money)
            shown["Economic value without card"]=shown["Economic value without card"].map(money)
            st.dataframe(shown,use_container_width=True,hide_index=True)
            weakest=marg.sort_values("Marginal decision value").iloc[0]
            st.info(f"**Diminishing-return check:** {weakest['Card']} contributes about {money(weakest['Marginal decision value'])} of marginal decision value versus the best modeled wallet without it.")

        alt=forced_alternative_analysis(
            r,effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards,
            complexity_cost,switching_cost,existing_wallet
        )
        if len(alt):
            st.markdown("#### Opportunity cost: what if you insist on another card?")
            st.caption("For each omitted card, CardOpt forces that card into the wallet and re-optimizes everything else. The loss versus the optimum is the modeled opportunity cost of that constraint.")
            shown=alt.copy()
            shown["Opportunity cost"]=shown["Opportunity cost"].map(money)
            shown["Economic net value"]=shown["Economic net value"].map(money)
            st.dataframe(shown,use_container_width=True,hide_index=True)

        frontier=portfolio_frontier(
            effective_spend,effective_cpp,bens,horizon,allowed_cards,
            complexity_cost,switching_cost,existing_wallet
        )
        st.markdown("#### Wallet complexity frontier")
        st.caption("This shows diminishing returns from allowing more cards. A larger wallet is not automatically better if another card adds very little incremental value.")
        if len(frontier):
            shown=frontier.copy()
            shown["Economic net value"]=shown["Economic net value"].map(money)
            shown["Decision utility"]=shown["Decision utility"].map(money)
            shown["Marginal decision value"]=shown["Marginal decision value"].map(lambda x:"—" if pd.isna(x) else money(x))
            st.dataframe(shown,use_container_width=True,hide_index=True)
            st.line_chart(frontier.set_index("Maximum cards")["Decision utility"])

        st.markdown("#### Decision boundaries")
        boundaries=fee_decision_boundaries(
            r,effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards,
            complexity_cost,switching_cost,existing_wallet
        )
        if len(boundaries):
            bshow=boundaries.copy()
            bshow["Threshold"]=bshow["Threshold"].map(money)
            st.dataframe(bshow,use_container_width=True,hide_index=True)
            st.caption("These are model-specific thresholds, not issuer predictions. They answer: at approximately what annual fee does the optimal portfolio change, holding the other assumptions fixed?")
        else:
            st.caption("No meaningful annual-fee boundary was identified among the fee-bearing eligible cards under this scenario.")

        st.subheader("Robustness: does the answer survive uncertainty?")
        score,rdf=robustness_analysis(
            r,effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards,
            complexity_cost,switching_cost,existing_wallet
        )
        rc1,rc2,rc3=st.columns(3)
        rc1.metric("Portfolio robustness",f"{score:.0f}%")
        rc2.metric("Scenarios tested",str(len(rdf)))
        if len(rdf):
            most_common=rdf["Wallet"].value_counts().index[0]
            rc3.metric("Most common wallet",most_common)
        if score>=80:
            st.success("This recommendation is relatively robust across the tested point-value, benefit, and category-spending changes.")
        elif score>=50:
            st.warning("This recommendation is moderately sensitive. Reasonable changes in assumptions can change the optimal wallet.")
        else:
            st.warning("This recommendation is highly sensitive. Treat the exact portfolio as scenario-dependent rather than a stable answer.")
        if len(rdf):
            shown=rdf.copy()
            shown["Decision utility"]=shown["Decision utility"].map(money)
            shown["Economic net value"]=shown["Economic net value"].map(money)
            st.dataframe(shown,use_container_width=True,hide_index=True)

        st.subheader("Macroeconomic lens")
        st.caption("CardOpt's core problem is microeconomic. This optional layer asks how a broad nominal spending shift could change the portfolio. It is a scenario, not an inflation forecast.")
        if macro_spend_shock==0:
            st.info("Set a broad spending scenario in the Economics tab to test whether a price-level / nominal-spending shift changes the optimal wallet.")
        else:
            factor=1+macro_spend_shock/100
            shocked={k:v*factor for k,v in effective_spend.items()}
            mr=solve(shocked,effective_cpp,bens,maxcards,horizon,allowed_cards=allowed_cards,
                     complexity_cost=complexity_cost,switching_cost=switching_cost,existing_wallet=existing_wallet)
            if mr:
                mc1,mc2,mc3=st.columns(3)
                mc1.metric("Nominal spending scenario",f"{macro_spend_shock:+d}%")
                mc2.metric("Scenario economic value",money(mr["net"]))
                mc3.metric("Value change",f"{'+' if mr['net']-r['net']>=0 else ''}{money(mr['net']-r['net'])}")
                st.write("Scenario wallet: **"+", ".join(mr["selected"])+"**")
                if set(mr["selected"])==set(r["selected"]):
                    st.caption("The selected portfolio remains unchanged under this broad nominal-spending scenario.")
                else:
                    st.caption("The portfolio changes because higher/lower nominal category spend alters the value of reward rates, caps, and fixed annual fees.")

        st.subheader("Assumption audit")
        audit=pd.DataFrame(
            [[n,f"{effective_cpp[n]:.2f}¢",money(bens[n]),money(DB[n]["fee"])] for n in r["selected"]],
            columns=["Card","Point value assumption","User-valued restricted benefits","Annual fee"]
        )
        st.dataframe(audit,use_container_width=True,hide_index=True)

        summary=f"""CardOpt 2.0 Decision Analysis
Annual spending used by model: {money(total)}
Spending source: {source_mode}
Horizon: {horizon}
Recommended wallet: {", ".join(r["selected"])}
Economic net annual value: {money(r["net"])}
Decision utility: {money(r["decision_utility"])}
Cash-back benchmark: {money(base)}
Economic advantage vs benchmark: {money(adv)}
Complexity friction: {money(r["complexity_penalty"])}
Switching friction: {money(r["switching_penalty"])}
Robustness score: {score:.0f}% across {len(rdf)} deterministic stress scenarios
Issuer-data review date: {VERIFIED}

CardOpt is an educational/research decision model, not individualized financial advice.
"""
        st.download_button("Download decision-analysis summary",summary,file_name="cardopt_decision_analysis.txt")
        st.download_button("Download allocation CSV",df.to_csv(index=False),file_name="cardopt_allocation.csv")

st.caption("CardOpt is an educational and research prototype, not individualized financial advice. Card terms change; verify current issuer terms before acting.")


st.markdown("---")
st.caption("CardOpt is an educational optimization tool, not individualized financial advice. Verify current issuer terms before applying or making a financial decision. Estimated value depends on your inputs and model assumptions.")
