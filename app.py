import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from datetime import date

st.set_page_config(page_title="CardOpt V5 | Portfolio Optimizer", page_icon="💳", layout="wide")

# ----------------------------- DESIGN -----------------------------
st.markdown("""
<style>
.block-container {max-width: 1220px; padding-top: 1.6rem; padding-bottom: 4rem;}
.hero {
  padding: 1.55rem 1.7rem; border:1px solid rgba(99,102,241,.22);
  border-radius:22px; background:linear-gradient(135deg,#111827 0%,#172554 55%,#1e1b4b 100%);
  box-shadow:0 12px 35px rgba(0,0,0,.18); margin-bottom:1.15rem;
}
.hero h1 {margin:0; font-size:2.45rem; letter-spacing:-.04em;}
.hero p {margin:.45rem 0 0; color:#cbd5e1; font-size:1.02rem;}
.pill {display:inline-block; margin-top:.8rem; padding:.3rem .65rem; border-radius:999px;
       background:rgba(99,102,241,.18); border:1px solid rgba(129,140,248,.3); font-size:.78rem; color:#c7d2fe;}
.kpi {border:1px solid rgba(148,163,184,.18); border-radius:18px; padding:1rem 1.05rem;
      background:rgba(30,41,59,.20); min-height:112px;}
.kpi-label {font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; opacity:.64;}
.kpi-value {font-size:1.8rem; font-weight:800; margin:.15rem 0;}
.kpi-note {font-size:.78rem; opacity:.62;}
.rec-card {border:1px solid rgba(99,102,241,.28); border-radius:18px; padding:1rem 1.1rem;
           background:linear-gradient(145deg,rgba(49,46,129,.20),rgba(15,23,42,.12)); min-height:142px;}
.rec-card h4 {margin:0 0 .35rem 0;}
.section-note {font-size:.88rem; opacity:.70; margin-top:-.4rem;}
.good {color:#22c55e; font-weight:700;}
.warnbox {border-left:4px solid #f59e0b; padding:.7rem 1rem; background:rgba(245,158,11,.08); border-radius:8px;}
[data-testid="stSidebar"] {border-right:1px solid rgba(148,163,184,.15);}
div.stButton > button {border-radius:13px; min-height:48px; font-weight:750;}
[data-testid="stMetric"] {border:1px solid rgba(148,163,184,.16); padding:.7rem .8rem; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- VERIFIED PRODUCT DATA -----------------------------
# Product terms are separated from subjective economic assumptions.
# Last manually reviewed against issuer/official product pages: 2026-09-20.
VERIFIED_DATE = "2026-09-20"

CARD_DB = {
    "Chase Sapphire Reserve": {
        "issuer":"Chase", "annual_fee":795.0, "reward_type":"points", "default_cpp":1.50,
        "rates":{"dining":3,"us_supermarkets":1,"airfare_direct":4,"hotels_direct":4,
                 "portal_flights":8,"portal_hotels":8,"drugstores":1,"other":1},
        "caps":{}, "anniversary_points":0,
        "benefits":[
            ("Annual travel credit",300.0,"Travel purchases; qualifying credited purchases do not earn points."),
            ("Sapphire Exclusive Tables dining credit",300.0,"Select restaurants; split into Jan–Jun and Jul–Dec periods.")
        ],
        "source":"https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve"
    },
    "American Express Gold": {
        "issuer":"American Express", "annual_fee":325.0, "reward_type":"points", "default_cpp":1.50,
        "rates":{"dining":4,"us_supermarkets":4,"airfare_direct":3,"hotels_direct":1,
                 "portal_flights":3,"portal_hotels":5,"drugstores":1,"other":1},
        "caps":{"dining":50000.0,"us_supermarkets":25000.0},
        "anniversary_points":0,
        "benefits":[
            ("Dining credit",120.0,"Up to $10 monthly at eligible partners; enrollment required."),
            ("Uber Cash",120.0,"Monthly Uber Cash benefit; terms and eligible U.S. use apply."),
            ("Resy credit",100.0,"Up to $50 Jan–Jun and $50 Jul–Dec at eligible U.S. Resy restaurants."),
            ("Dunkin' credit",84.0,"Up to $7 monthly; enrollment/terms apply.")
        ],
        "source":"https://www.americanexpress.com/us/credit-cards/card/gold-card/"
    },
    "Capital One Venture X": {
        "issuer":"Capital One", "annual_fee":395.0, "reward_type":"miles", "default_cpp":1.50,
        "rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,
                 "portal_flights":5,"portal_hotels":10,"drugstores":2,"other":2},
        "caps":{}, "anniversary_points":10000,
        "benefits":[
            ("Capital One Travel credit",300.0,"Annual credit for eligible bookings through Capital One Travel.")
        ],
        "source":"https://www.capitalone.com/credit-cards/venture-x/"
    },
    "Citi Double Cash": {
        "issuer":"Citi", "annual_fee":0.0, "reward_type":"cash-like points", "default_cpp":1.00,
        "rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,
                 "portal_flights":2,"portal_hotels":5,"drugstores":2,"other":2},
        "caps":{}, "anniversary_points":0, "benefits":[],
        "source":"https://www.citi.com/credit-cards/citi-double-cash-credit-card"
    },
    "Chase Freedom Unlimited": {
        "issuer":"Chase", "annual_fee":0.0, "reward_type":"cash back", "default_cpp":1.00,
        "rates":{"dining":3,"us_supermarkets":1.5,"airfare_direct":1.5,"hotels_direct":1.5,
                 "portal_flights":5,"portal_hotels":5,"drugstores":3,"other":1.5},
        "caps":{}, "anniversary_points":0, "benefits":[],
        "source":"https://creditcards.chase.com/cash-back-credit-cards/freedom/unlimited"
    },
    "Wells Fargo Active Cash": {
        "issuer":"Wells Fargo", "annual_fee":0.0, "reward_type":"cash back", "default_cpp":1.00,
        "rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,
                 "portal_flights":2,"portal_hotels":2,"drugstores":2,"other":2},
        "caps":{}, "anniversary_points":0, "benefits":[],
        "source":"https://www.wellsfargo.com/credit-cards/active-cash/"
    }
}

CATEGORIES = {
    "dining":"Dining",
    "us_supermarkets":"U.S. supermarkets",
    "airfare_direct":"Flights booked direct",
    "hotels_direct":"Hotels booked direct",
    "portal_flights":"Flights via issuer portal",
    "portal_hotels":"Hotels via issuer portal",
    "drugstores":"Drugstores",
    "other":"Everything else"
}

DEFAULT_SPEND = {
    "dining":6000, "us_supermarkets":5000, "airfare_direct":2500, "hotels_direct":1500,
    "portal_flights":500, "portal_hotels":500, "drugstores":1000, "other":8000
}

# ----------------------------- SOLVER -----------------------------
def solve(spend, cpp, benefit_values, max_cards, horizon):
    names=list(CARD_DB); cats=list(spend); n=len(names); m=len(cats)
    n_bonus=n*m; n_over=n*m; y0=n_bonus+n_over; N=y0+n
    c=np.zeros(N)

    for i,name in enumerate(names):
        card=CARD_DB[name]
        pv=cpp[name]/100.0
        for j,cat in enumerate(cats):
            c[i*m+j] = -(card["rates"][cat]*pv)
            c[n_bonus+i*m+j] = -(1.0*pv)
        anniversary = card["anniversary_points"]*pv if horizon=="Ongoing annual economics" else 0.0
        c[y0+i] = card["annual_fee"] - benefit_values[name] - anniversary

    constraints=[]
    for j,cat in enumerate(cats):
        row=np.zeros(N)
        for i in range(n):
            row[i*m+j]=1
            row[n_bonus+i*m+j]=1
        constraints.append(LinearConstraint(row,spend[cat],spend[cat]))

    M=max(1.0,sum(spend.values()))
    for i in range(n):
        row=np.zeros(N)
        for j in range(m):
            row[i*m+j]=1
            row[n_bonus+i*m+j]=1
        row[y0+i]=-M
        constraints.append(LinearConstraint(row,-np.inf,0))

    row=np.zeros(N); row[y0:]=1
    constraints.append(LinearConstraint(row,-np.inf,max_cards))

    upper=np.full(N,np.inf)
    for i,name in enumerate(names):
        for j,cat in enumerate(cats):
            if cat in CARD_DB[name]["caps"]:
                row=np.zeros(N); row[i*m+j]=1
                constraints.append(LinearConstraint(row,-np.inf,CARD_DB[name]["caps"][cat]))
            else:
                upper[n_bonus+i*m+j]=0.0
    upper[y0:]=1
    integrality=np.zeros(N,dtype=int); integrality[y0:]=1

    res=milp(c=c,integrality=integrality,bounds=Bounds(np.zeros(N),upper),constraints=constraints)
    if not res.success:
        raise RuntimeError(res.message)

    selected=[names[i] for i in range(n) if res.x[y0+i]>.5]
    rows=[]; gross=0.0
    card_reward={n:0.0 for n in names}
    card_spend={n:0.0 for n in names}

    for i,name in enumerate(names):
        pv=cpp[name]/100.0
        for j,cat in enumerate(cats):
            b=float(res.x[i*m+j]); o=float(res.x[n_bonus+i*m+j])
            if b>.01:
                val=b*CARD_DB[name]["rates"][cat]*pv
                gross+=val; card_reward[name]+=val; card_spend[name]+=b
                rows.append([CATEGORIES[cat],name,b,CARD_DB[name]["rates"][cat],val,"Bonus"])
            if o>.01:
                val=o*pv
                gross+=val; card_reward[name]+=val; card_spend[name]+=o
                rows.append([CATEGORIES[cat],name,o,1.0,val,"Post-cap"])

    fees=sum(CARD_DB[n]["annual_fee"] for n in selected)
    benefits=sum(benefit_values[n] for n in selected)
    anniversary=sum(CARD_DB[n]["anniversary_points"]*(cpp[n]/100.0) for n in selected) if horizon=="Ongoing annual economics" else 0.0
    net=gross+benefits+anniversary-fees

    alloc=pd.DataFrame(rows,columns=["Category","Card","Spend","Multiplier","Reward value","Tranche"])
    details=[]
    for name in selected:
        ann=CARD_DB[name]["anniversary_points"]*(cpp[name]/100.0) if horizon=="Ongoing annual economics" else 0
        details.append({
            "Card":name, "Spend routed":card_spend[name], "Reward value":card_reward[name],
            "Benefit value":benefit_values[name], "Anniversary value":ann,
            "Annual fee":CARD_DB[name]["annual_fee"],
            "Net contribution":card_reward[name]+benefit_values[name]+ann-CARD_DB[name]["annual_fee"]
        })
    return {
        "selected":selected, "allocation":alloc, "gross":gross, "fees":fees,
        "benefits":benefits, "anniversary":anniversary, "net":net,
        "details":pd.DataFrame(details)
    }

def scenario_solve(spend, cpp, benefit_values, max_cards, horizon, point_factor, benefit_factor):
    scpp={k:max(.5,v*point_factor) for k,v in cpp.items()}
    sb={k:min(sum(x[1] for x in CARD_DB[k]["benefits"]), v*benefit_factor) for k,v in benefit_values.items()}
    return solve(spend,scpp,sb,max_cards,horizon)

# ----------------------------- HEADER -----------------------------
st.markdown("""
<div class="hero">
  <h1>CardOpt <span style="font-size:1rem;opacity:.55">V5</span></h1>
  <p>Optimize the wallet, not just the card — recurring rewards, fees, usable benefits and complexity in one model.</p>
  <span class="pill">Mixed-Integer Linear Programming • scenario-tested • source-aware</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.header("Optimization controls")
    max_cards=st.slider("Maximum wallet size",1,len(CARD_DB),3,
                        help="The optimizer may use fewer cards if another card adds no economic value.")
    baseline=st.number_input("Cash-back benchmark (%)",0.0,10.0,2.0,.25)
    horizon=st.radio("Analysis horizon",["Ongoing annual economics","First-year recurring economics"],
                     help="First-year recurring mode excludes anniversary points and still excludes welcome bonuses.")
    st.divider()
    st.caption("Model outputs are estimates. Point values and benefit utilization are assumptions, not issuer guarantees.")

# ----------------------------- INPUT TABS -----------------------------
tabs=st.tabs(["💸 Spending","🎁 Benefit utilization","📈 Point values","🧪 Model & sources"])

with tabs[0]:
    st.subheader("Annual spending profile")
    st.markdown('<div class="section-note">Enter expected annual spend. Direct travel and issuer-portal travel are separated because earn rates differ.</div>',unsafe_allow_html=True)
    spend={}
    cols=st.columns(2)
    for idx,(key,label) in enumerate(CATEGORIES.items()):
        with cols[idx%2]:
            spend[key]=st.number_input(label,min_value=0.0,value=float(DEFAULT_SPEND[key]),
                                       step=250.0,format="%.0f",key="sp_"+key)
    st.metric("Total modeled annual spend",f"${sum(spend.values()):,.0f}")

with tabs[1]:
    st.subheader("Value benefits at what they're actually worth to you")
    st.markdown('<div class="warnbox">Do not automatically value a $100 credit at $100. Enter $0 if you would not naturally use it. This prevents advertised credits from artificially inflating a card.</div>',unsafe_allow_html=True)
    benefit_component_values={}
    for name,card in CARD_DB.items():
        if card["benefits"]:
            with st.expander(name,expanded=name in ["American Express Gold","Capital One Venture X"]):
                for benefit,maxv,note in card["benefits"]:
                    default = maxv if name=="Capital One Venture X" and benefit=="Capital One Travel credit" else (maxv*.60 if name=="American Express Gold" else 0.0)
                    benefit_component_values[(name,benefit)] = st.slider(
                        benefit,0.0,float(maxv),float(round(default/5)*5),5.0,
                        key="benefit_"+name+benefit,
                        help=note
                    )
                    st.caption(note)

with tabs[2]:
    st.subheader("Reward valuation assumptions")
    st.write("Cash-back cards default to 1.00¢. Transferable point currencies are user-adjustable because realized redemption value varies.")
    cpp={}
    for name,card in CARD_DB.items():
        c1,c2=st.columns([2,1])
        with c1: st.write(f"**{name}**")
        with c2:
            cpp[name]=st.number_input("¢ per point/mile",0.50,3.00,float(card["default_cpp"]),0.05,
                                      key="cpp_"+name,label_visibility="collapsed")
    st.caption("CardOpt does not claim that an issuer guarantees the selected cents-per-point value.")

with tabs[3]:
    st.subheader("What the model includes")
    st.write("CardOpt jointly selects a portfolio and routes category spending to maximize estimated recurring net economic value.")
    st.markdown("""
**Included:** published category earn rates, annual fees, published category caps, user-valued recurring benefits, applicable anniversary points, wallet-size constraints, and a cash-back benchmark.

**Excluded:** welcome offers, APR/interest, approval odds, credit-score effects, taxes, merchant coding uncertainty, transfer-partner award availability, lounge value, insurance value, and other qualitative perks unless explicitly modeled.

**Interpretation:** the result is an optimization under assumptions—not a guarantee of savings or a recommendation to borrow.
""")
    st.subheader("Issuer sources")
    for name,card in CARD_DB.items():
        st.markdown(f"- [{name}]({card['source']}) — terms reviewed {VERIFIED_DATE}")
    st.caption("Card products change. Re-verify terms before publication or reliance.")

benefit_values={name:0.0 for name in CARD_DB}
for (name,benefit),v in benefit_component_values.items():
    benefit_values[name]+=v

run=st.button("Run optimization",type="primary",use_container_width=True)

# ----------------------------- REPORT -----------------------------
if run or "v5_ran" not in st.session_state:
    st.session_state.v5_ran=True
    total=sum(spend.values())
    if total<=0:
        st.error("Enter at least some annual spending.")
        st.stop()

    result=solve(spend,cpp,benefit_values,max_cards,horizon)
    base=total*baseline/100.0
    uplift=result["net"]-base
    value_rate=result["net"]/total

    st.divider()
    st.subheader("Optimization report")
    st.caption("Estimated recurring economics under the assumptions entered above.")

    k1,k2,k3,k4=st.columns(4)
    kpis=[
        ("Estimated net value",f"${result['net']:,.0f}","Rewards + valued benefits − fees"),
        ("Net-value rate",f"{value_rate:.2%}","Net value ÷ modeled spend"),
        (f"{baseline:.2f}% benchmark",f"${base:,.0f}","Simple no-fee cash-back comparator"),
        ("Modeled advantage",f"{uplift:+,.0f}","Difference vs benchmark")
    ]
    for col,(lab,val,note) in zip([k1,k2,k3,k4],kpis):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

    st.markdown("### Recommended wallet")
    rec_cols=st.columns(max(1,len(result["selected"])))
    for col,name in zip(rec_cols,result["selected"]):
        row=result["details"].set_index("Card").loc[name]
        with col:
            st.markdown(f"""
            <div class="rec-card">
              <h4>{name}</h4>
              <div style="opacity:.72">{CARD_DB[name]['issuer']} • ${CARD_DB[name]['annual_fee']:,.0f} annual fee</div>
              <div style="margin-top:.65rem"><b>${row['Spend routed']:,.0f}</b> routed spend</div>
              <div><b>${row['Net contribution']:,.0f}</b> modeled net contribution</div>
            </div>
            """,unsafe_allow_html=True)

    # Why these cards
    st.markdown("### Why the optimizer chose this wallet")
    reasons=[]
    for name in result["selected"]:
        sub=result["allocation"][result["allocation"]["Card"]==name]
        if not sub.empty:
            top=sub.sort_values("Reward value",ascending=False).iloc[0]
            reasons.append(f"**{name}** captures strong value from **{top['Category']}** ({top['Multiplier']:g}× in the modeled tranche) and contributes about **${result['details'].set_index('Card').loc[name,'Net contribution']:,.0f}** after its modeled fee/benefit economics.")
    st.markdown("\n\n".join(reasons))

    # Allocation
    st.markdown("### Best card by spending category")
    show=result["allocation"].copy()
    show["Spend"]=show["Spend"].map(lambda x:f"${x:,.0f}")
    show["Multiplier"]=show["Multiplier"].map(lambda x:f"{x:g}×")
    show["Reward value"]=show["Reward value"].map(lambda x:f"${x:,.0f}")
    st.dataframe(show,use_container_width=True,hide_index=True)

    # Economics decomposition - cleaner table + chart
    st.markdown("### Economic bridge")
    econ=pd.DataFrame({
        "Component":["Reward value","User-valued recurring benefits","Anniversary value","Annual fees","Estimated net value"],
        "Amount":[result["gross"],result["benefits"],result["anniversary"],-result["fees"],result["net"]]
    })
    e1,e2=st.columns([1,1.4])
    with e1:
        st.dataframe(econ.style.format({"Amount":"${:,.0f}"}),use_container_width=True,hide_index=True)
    with e2:
        st.bar_chart(econ.iloc[:4].set_index("Component")["Amount"],horizontal=True)

    # Scenario robustness
    st.markdown("### Robustness check")
    st.caption("Tests whether the recommended economics survive lower/higher point values and benefit realization.")
    scenarios=[
        ("Conservative",0.75,0.70),
        ("Base",1.00,1.00),
        ("Upside",1.25,1.00)
    ]
    srows=[]
    for label,pf,bf in scenarios:
        r=scenario_solve(spend,cpp,benefit_values,max_cards,horizon,pf,bf)
        srows.append({
            "Scenario":label,
            "Point-value factor":pf,
            "Benefit-value factor":bf,
            "Estimated net value":r["net"],
            "Vs benchmark":r["net"]-base,
            "Optimal wallet":" + ".join(r["selected"])
        })
    sdf=pd.DataFrame(srows)
    st.dataframe(
        sdf.style.format({"Point-value factor":"{:.0%}","Benefit-value factor":"{:.0%}",
                          "Estimated net value":"${:,.0f}","Vs benchmark":"${:+,.0f}"}),
        use_container_width=True,hide_index=True
    )
    st.line_chart(sdf.set_index("Scenario")["Estimated net value"])

    # Complexity frontier
    st.markdown("### Wallet complexity frontier")
    frontier=[]
    for k in range(1,len(CARD_DB)+1):
        r=solve(spend,cpp,benefit_values,k,horizon)
        frontier.append({"Maximum cards":k,"Estimated net value":r["net"],"Wallet":" + ".join(r["selected"])})
    fdf=pd.DataFrame(frontier)
    fdf["Marginal value"]=fdf["Estimated net value"].diff()
    # stop visual after 2 consecutive zero-ish marginal gains, but keep full table downloadable
    last=len(fdf)
    zeros=0
    for i in range(1,len(fdf)):
        if abs(fdf.loc[i,"Marginal value"])<1:
            zeros+=1
            if zeros>=2:
                last=i+1
                break
        else:
            zeros=0
    vis=fdf.iloc[:last]
    st.line_chart(vis.set_index("Maximum cards")["Estimated net value"])
    st.dataframe(
        vis.style.format({"Estimated net value":"${:,.0f}","Marginal value":"${:,.0f}"}),
        use_container_width=True,hide_index=True
    )
    positive=fdf[fdf["Marginal value"].fillna(999)>1]
    efficient_k=int(positive["Maximum cards"].max()) if not positive.empty else 1
    st.info(f"Complexity insight: under these assumptions, material modeled value stops increasing at about **{efficient_k} card(s)**. More allowed cards do not necessarily mean the optimizer will use them.")

    # Audit trail
    st.markdown("### Assumption audit")
    audit=[]
    for name in result["selected"]:
        audit.append({
            "Card":name,
            "Annual fee (issuer term)":CARD_DB[name]["annual_fee"],
            "Point value (assumption)":cpp[name],
            "Recurring benefits valued by user":benefit_values[name],
            "Published max of modeled benefits":sum(x[1] for x in CARD_DB[name]["benefits"])
        })
    adf=pd.DataFrame(audit)
    st.dataframe(adf.style.format({
        "Annual fee (issuer term)":"${:,.0f}",
        "Point value (assumption)":"{:.2f}¢",
        "Recurring benefits valued by user":"${:,.0f}",
        "Published max of modeled benefits":"${:,.0f}"
    }),use_container_width=True,hide_index=True)

    # Downloadable summary
    summary_lines=[
        "CardOpt V5 Optimization Summary",
        f"Generated: {date.today().isoformat()}",
        f"Modeled annual spend: ${total:,.2f}",
        f"Recommended wallet: {' + '.join(result['selected'])}",
        f"Estimated recurring net value: ${result['net']:,.2f}",
        f"Net-value rate: {value_rate:.2%}",
        f"Benchmark ({baseline:.2f}%): ${base:,.2f}",
        f"Modeled advantage vs benchmark: ${uplift:,.2f}",
        "",
        "Important: Results depend on user-entered point valuations and benefit values. They are estimates, not guaranteed returns.",
        f"Issuer terms last reviewed: {VERIFIED_DATE}"
    ]
    d1,d2=st.columns(2)
    with d1:
        st.download_button("Download summary (.txt)","\n".join(summary_lines),
                           file_name="cardopt_v5_summary.txt",mime="text/plain",use_container_width=True)
    with d2:
        st.download_button("Download spending strategy (.csv)",
                           result["allocation"].to_csv(index=False),
                           file_name="cardopt_v5_allocation.csv",mime="text/csv",use_container_width=True)

    st.markdown("### Interpretation")
    st.write(
        f"With **${total:,.0f}** of modeled annual spend, CardOpt estimates **${result['net']:,.0f}** "
        f"of recurring net economic value under the selected assumptions. That equals a **{value_rate:.2%} net-value rate** "
        f"and is **${uplift:,.0f}** {'above' if uplift>=0 else 'below'} the {baseline:.2f}% cash-back benchmark."
    )
    st.caption("Net-value rate includes rewards, user-valued recurring benefits, applicable anniversary value, and annual fees; it is not the same as an issuer-advertised rewards rate.")

st.divider()
st.caption("CardOpt V5 • Research prototype • MILP portfolio optimization • Pay statement balances in full; rewards do not justify interest-bearing debt.")
