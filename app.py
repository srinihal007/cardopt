import streamlit as st
import pandas as pd
import numpy as np
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

def solve(spend,cpp,bens,maxcards,horizon, allowed_cards=None, required_cards=None):
    """
    MILP variables
      X[i,j] = reward-earning spend in category j on card i at the category rate
      Z[i,j] = post-cap spend in category j on card i at 1x
      D[i,j] = spend covered by a modeled statement/travel credit; earns no rewards
      Y[i]   = 1 when card i is carried, otherwise 0

    D prevents double counting: a dollar covered by a modeled credit contributes $1
    of credit value but earns no points when issuer terms say credited spend is ineligible.
    """
    names=list(DB); cats=list(CATS); n=len(names); m=len(cats); q=n*m; N=3*q+n
    allowed_cards=set(names if allowed_cards is None else allowed_cards)
    required_cards=set([] if required_cards is None else required_cards)
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
        c[Y(i)]=d["fee"]-bens[nm]-ann+1e-5
        ub[Y(i)]=1 if nm in allowed_cards else 0
        integ[Y(i)]=1

    rows=[]; lo=[]; hi=[]
    # Every dollar of category spend is assigned exactly once.
    for j,cat in enumerate(cats):
        r={}
        for i in range(n):
            r[X(i,j)]=1; r[Z(i,j)]=1; r[D(i,j)]=1
        rows.append(r); lo.append(spend[cat]); hi.append(spend[cat])

    M=max(sum(spend.values()),1)
    # No spend can flow to an unselected card.
    for i in range(n):
        r={Y(i):-M}
        for j in range(m):
            r[X(i,j)]=1; r[Z(i,j)]=1; r[D(i,j)]=1
        rows.append(r); lo.append(-np.inf); hi.append(0)

    # Category bonus caps. Overflow goes to Z at 1x.
    for i,nm in enumerate(names):
        for cat,cap in DB[nm]["caps"].items():
            rows.append({X(i,cats.index(cat)):1,Y(i):-cap}); lo.append(-np.inf); hi.append(0)

    # Statement/travel credit caps. D is eligible covered spend and earns no rewards.
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

    fees=sum(DB[x]["fee"] for x in selected)
    manual_ben=sum(bens[x] for x in selected)
    ann=sum(DB[x]["ann"]*cpp[x]/100 for x in selected) if horizon=="Ongoing annual economics" else 0
    details=[]
    for x in selected:
        av=DB[x]["ann"]*cpp[x]/100 if horizon=="Ongoing annual economics" else 0
        total_ben=bens[x]+cc[x]
        details.append({"Card":x,"Spend":cs[x],"Rewards":cg[x],"Benefits":total_ben,
                        "Anniversary":av,"Fee":DB[x]["fee"],
                        "Net":cg[x]+total_ben+av-DB[x]["fee"]})
    total_ben=manual_ben+credit_value
    return {"selected":selected,"allocation":alloc,"gross":gross,"fees":fees,
            "benefits":total_ben,"manual_benefits":manual_ben,"credits":credit_value,
            "anniversary":ann,"net":gross+total_ben+ann-fees,"details":details}

st.markdown("""<div class="hero61">
<div class="brand">Card<span>Opt</span></div>
<div class="eyebrow">Smarter spending. Clearer decisions.</div>
<h1>Optimize your wallet.<br><span>Maximize your value.</span></h1>
<p>CardOpt uses mathematical optimization to find the credit card combination that best fits your spending and the value you place on card benefits.</p>
<div class="stats">
<div class="stat"><b>6 cards</b><small>carefully modeled</small></div>
<div class="stat"><b>8 spending categories</b><small>optimized together</small></div>
<div class="stat"><b>Thousands</b><small>of possible allocations</small></div>
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

        basics,rewards,wallet=st.tabs(["Credit Card Basics","Rewards 101","Build Your Wallet"])

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
        st.markdown('<div class="pagehero"><div class="eyebrow">About CardOpt</div><h1>What should actually be in your wallet?</h1><p>CardOpt is an independent quantitative finance and optimization project built to explore that question.</p></div>',unsafe_allow_html=True)
        a1,a2=st.columns([1.25,1])
        with a1:
            st.markdown("""<div class="research-card"><h3>Why CardOpt exists</h3><p>Comparing individual cards is relatively easy. Comparing a portfolio is harder because rewards, annual fees, spending caps, point values, benefits and overlapping categories interact. CardOpt models those interactions together.</p></div>""",unsafe_allow_html=True)
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
    st.markdown('<div class="step-head"><div class="stepnum">2</div><div><b>Your analysis</b><span>Enter spending, benefit values and reward assumptions.</span></div></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Spending","Benefits","Reward assumptions","Methodology"])

    with tabs[0]:
        st.subheader("Your annual spending")
        st.markdown('<div class="note">Use a typical year. CardOpt will decide where each dollar should go.</div>',unsafe_allow_html=True)
        spend={}; cols=st.columns(2)
        for i,(k,label) in enumerate(CATS.items()):
            with cols[i%2]: spend[k]=st.number_input(label,0.0,value=float(DEFAULT[k]),step=500.0,format="%.0f",key="s"+k)
        st.markdown(f"""<div class="learn-example" style="margin-top:1rem"><span style="color:#647696;font-size:.9rem">TOTAL ANNUAL CARD SPENDING</span><br><strong style="font-size:2.25rem;color:#0b55d9">{money(sum(spend.values()))}</strong></div>""",unsafe_allow_html=True)

    bens={}
    with tabs[1]:
        st.subheader("What are the benefits worth to you?")
        st.markdown('<div class="note">Restricted lifestyle credits are not automatically worth face value. Start at $0 and add only value you realistically expect to use. The Sapphire Reserve and Venture X travel credits are modeled separately against eligible planned travel spend so credited dollars do not also earn rewards.</div>',unsafe_allow_html=True)
        for nm,d in DB.items():
            with st.expander(nm,expanded=(nm=="Capital One Venture X")):
                total=0
                if not d["benefits"]: st.caption("No recurring credit explicitly modeled.")
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
        for nm,d in DB.items(): cpp[nm]=st.number_input(f"{nm} · cents per point",.50,3.00,float(d["cpp"]),.05,key="c"+nm)
        st.info("Cash-like cards default to 1.00¢. Transferable currencies default to 1.50¢ as an analyst assumption, not an issuer guarantee.")

    with tabs[3]:
        st.subheader("How CardOpt works")
        st.caption("For the full mathematical formulation, return to the welcome screen and open Math & Model.")
        st.write("CardOpt uses mixed-integer linear programming. Binary variables decide which cards enter the wallet; continuous variables decide how much spending in each category goes to each selected card.")
        st.markdown("**Objective** · Maximize estimated recurring economic value: reward value + user-valued recurring benefits + applicable anniversary value − annual fees.")
        st.markdown("**Constraints** · Allocate every dollar, route spend only to selected cards, enforce modeled category caps, and respect the user's maximum wallet size.")
        st.markdown("**Benchmark** · Compare the optimized wallet with putting the same spending on a simple cash-back card.")
        st.markdown("**Sensitivity** · Stress transferable-point values and benefit utilization while keeping cash-like values anchored.")
        st.subheader("Validation record")
        st.write("Known-answer tests have covered basic cash-back arithmetic, annual-fee tradeoffs, multi-card routing, category caps and post-cap overflow, user-valued benefits, and first-year versus ongoing anniversary treatment.")
        st.subheader("Scope")
        st.write("Welcome offers, APR/interest, approval odds, credit-score effects, taxes, merchant-coding uncertainty, transfer-partner award availability, and unpriced qualitative perks are intentionally excluded from the recurring optimization.")
        st.subheader("Official issuer sources")
        st.caption(f"Prototype terms reviewed {VERIFIED}. Terms can change.")
        for nm,d in DB.items(): st.markdown(f'<div class="source"><b>{nm}</b><br><a href="{d["source"]}" target="_blank">Official issuer page ↗</a></div>',unsafe_allow_html=True)

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
    r=solve(effective_spend,effective_cpp,bens,maxcards,horizon,allowed_cards=allowed_cards)
    if r is None:
        st.error("No feasible wallet matches those preferences. Try relaxing a filter.")
        st.stop()

    base=total*benchmark; adv=r["net"]-base

    if existing_wallet:
        current_allowed=[n for n in existing_wallet if n in DB]
        current=solve(effective_spend,effective_cpp,bens,len(current_allowed),horizon,
                      allowed_cards=current_allowed,required_cards=current_allowed)
        if current is not None:
            improvement=r["net"]-current["net"]
            st.subheader("CardOpt Compare")
            ca,cb,cc=st.columns(3)
            ca.metric("Your current wallet",money(current["net"]))
            cb.metric("CardOpt optimized",money(r["net"]))
            cc.metric("Potential improvement",("+" if improvement>=0 else "")+money(improvement))
            if improvement>1:
                st.success("Under your assumptions, CardOpt estimates about "+money(improvement)+" more annual value than your current wallet.")
            elif improvement>=-1:
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
                st.write("These cards were eligible but did not improve the highest-value modeled portfolio under your current assumptions:")
                for n in omitted:
                    st.write("• **"+n+"**")
            else:
                st.write("All eligible modeled cards are in the optimized wallet.")
        st.info("For the assumptions, stress tests, allocation details and model diagnostics, switch to Advanced in the sidebar and run the same inputs.")
    else:
        st.header("Advanced analysis")
        m=st.columns(4)
        m[0].metric("Estimated net value",money(r["net"]))
        m[1].metric("Net-value rate",f"{r['net']/total*100:.2f}%")
        m[2].metric(f"{benchmark*100:.1f}% benchmark",money(base))
        m[3].metric("Modeled advantage",f"{'+' if adv>=0 else ''}{money(adv)}")
        st.caption("Net-value rate includes modeled benefits and fees. It is not an issuer-advertised reward rate.")

        st.subheader("Portfolio economics")
        ddf=pd.DataFrame(r["details"])
        display=ddf.copy()
        for col in ["Spend","Rewards","Benefits","Anniversary","Fee","Net"]: display[col]=display[col].map(money)
        st.dataframe(display,use_container_width=True,hide_index=True)

        df=pd.DataFrame(r["allocation"],columns=["Spending category","Card","Annual spend","Multiplier","Point value (¢)","Estimated reward value","Rate tier"])
        st.subheader("Category allocation")
        st.dataframe(df,use_container_width=True,hide_index=True)

        st.subheader("Economic bridge")
        bridge=pd.DataFrame({"Component":["Reward value","User-valued benefits","Anniversary value","Annual fees","Net value"],
                             "Value":[r["gross"],r["benefits"],r["anniversary"],-r["fees"],r["net"]]})
        st.dataframe(bridge.assign(Value=bridge["Value"].map(money)),use_container_width=True,hide_index=True)
        st.bar_chart(bridge.set_index("Component")["Value"],horizontal=True)

        st.subheader("Robustness")
        st.caption("Transferable-point values are stressed; cash-like reward values stay anchored.")
        rows=[]
        for label,pf,bf in [("Conservative",.75,.70),("Base",1,1),("Upside",1.25,1)]:
            scpp={n:(cpp[n] if n in CASHLIKE else max(.5,cpp[n]*pf)) for n in DB}
            sb={n:min(sum(x[1] for x in DB[n]["benefits"]),bens[n]*bf) for n in DB}
            rr=solve(spend,scpp,sb,maxcards,horizon)
            rows.append([label,rr["net"],", ".join(rr["selected"])])
        sdf=pd.DataFrame(rows,columns=["Scenario","Estimated net value","Optimal wallet"])
        shown=sdf.copy(); shown["Estimated net value"]=shown["Estimated net value"].map(money)
        st.dataframe(shown,use_container_width=True,hide_index=True)
        st.line_chart(sdf.set_index("Scenario")["Estimated net value"])

        st.subheader("Wallet complexity")
        rows=[]; prev=None; flat=0
        for k in range(1,7):
            rr=solve(spend,cpp,bens,k,horizon); marginal=np.nan if prev is None else rr["net"]-prev
            rows.append([k,rr["net"],marginal,", ".join(rr["selected"])])
            flat=flat+1 if prev is not None and abs(marginal)<1 else 0; prev=rr["net"]
            if flat>=2: break
        kdf=pd.DataFrame(rows,columns=["Maximum cards","Net value","Marginal value","Optimal wallet"])
        shown=kdf.copy(); shown["Net value"]=shown["Net value"].map(money); shown["Marginal value"]=shown["Marginal value"].map(lambda x:"to" if pd.isna(x) else money(x))
        st.dataframe(shown,use_container_width=True,hide_index=True)
        st.line_chart(kdf.set_index("Maximum cards")["Net value"])

        st.subheader("Assumption audit")
        audit=pd.DataFrame([[n,f"{cpp[n]:.2f}¢",money(bens[n]),money(DB[n]["fee"])] for n in r["selected"]],
                           columns=["Card","Point value assumption","Benefit value assumption","Annual fee"])
        st.dataframe(audit,use_container_width=True,hide_index=True)

        summary=f"""CardOpt V6 analysis
Annual spending: {money(total)}
Horizon: {horizon}
Recommended wallet: {", ".join(r["selected"])}
Estimated net annual value: {money(r["net"])}
Cash-back benchmark: {money(base)}
Estimated modeled advantage: {money(adv)}
Issuer-data review date: {VERIFIED}

Educational/research model; not individualized financial advice.
"""
        st.download_button("Download analysis summary",summary,file_name="cardopt_summary.txt")
        st.download_button("Download allocation CSV",df.to_csv(index=False),file_name="cardopt_allocation.csv")

st.caption("CardOpt is an educational and research prototype, not individualized financial advice. Card terms change; verify current issuer terms before acting.")


st.markdown("---")
st.caption("CardOpt is an educational optimization tool, not individualized financial advice. Verify current issuer terms before applying or making a financial decision. Estimated value depends on your inputs and model assumptions.")
