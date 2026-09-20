import streamlit as st
import numpy as np
import pandas as pd
import copy
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="CardOpt | Rewards Optimizer", page_icon="💳", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(135deg,#101827,#182235); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.15rem;}
.hero p {margin:.35rem 0 0; color:#cbd5e1;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:16px; padding:1rem; min-height:105px;}
.metric-label {font-size:.8rem; opacity:.68; text-transform:uppercase; letter-spacing:.04em;}
.metric-value {font-size:1.65rem; font-weight:750; margin-top:.2rem;}
.small {font-size:.82rem; opacity:.72;}
[data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.2);}
div.stButton > button {border-radius:12px; min-height:46px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DATA LAYER
# Published card terms are kept separate from user valuation assumptions.
# Verify issuer terms before production use; products change over time.
# ------------------------------------------------------------------
CARD_DB = {
    "Chase Sapphire Reserve": {
        "issuer":"Chase", "annual_fee":795.0, "default_cpp":1.50,
        "rates":{"dining":3,"us_supermarkets":1,"airfare_direct":4,"hotels_direct":4,
                 "portal_flights":8,"portal_hotels":8,"drugstores":1,"other":1},
        "caps":{}, "credit_max":300.0, "anniversary_points":0
    },
    "American Express Gold": {
        "issuer":"American Express", "annual_fee":325.0, "default_cpp":1.50,
        "rates":{"dining":4,"us_supermarkets":4,"airfare_direct":3,"hotels_direct":1,
                 "portal_flights":3,"portal_hotels":5,"drugstores":1,"other":1},
        "caps":{"dining":50000.0,"us_supermarkets":25000.0},
        "credit_max":424.0, "anniversary_points":0
    },
    "Capital One Venture X": {
        "issuer":"Capital One", "annual_fee":395.0, "default_cpp":1.50,
        "rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,
                 "portal_flights":5,"portal_hotels":10,"drugstores":2,"other":2},
        "caps":{}, "credit_max":300.0, "anniversary_points":10000
    },
    "Citi Double Cash": {
        "issuer":"Citi", "annual_fee":0.0, "default_cpp":1.00,
        "rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,
                 "portal_flights":2,"portal_hotels":5,"drugstores":2,"other":2},
        "caps":{}, "credit_max":0.0, "anniversary_points":0
    },
    "Chase Freedom Unlimited": {
        "issuer":"Chase", "annual_fee":0.0, "default_cpp":1.00,
        "rates":{"dining":3,"us_supermarkets":1.5,"airfare_direct":1.5,"hotels_direct":1.5,
                 "portal_flights":5,"portal_hotels":5,"drugstores":3,"other":1.5},
        "caps":{}, "credit_max":0.0, "anniversary_points":0
    },
    "Wells Fargo Active Cash": {
        "issuer":"Wells Fargo", "annual_fee":0.0, "default_cpp":1.00,
        "rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,
                 "portal_flights":2,"portal_hotels":2,"drugstores":2,"other":2},
        "caps":{}, "credit_max":0.0, "anniversary_points":0
    },
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
DEFAULT_SPEND = {"dining":6000,"us_supermarkets":5000,"airfare_direct":2500,"hotels_direct":1500,
                 "portal_flights":500,"portal_hotels":500,"drugstores":1000,"other":8000}

def solve(spend, db, cpp, credit_values, max_cards, mode):
    names=list(db); cats=list(spend); n=len(names); m=len(cats)
    # Two spend tranches per card/category: bonus and overflow.
    # Overflow earns 1x only when a published bonus cap exists.
    n_bonus=n*m; n_over=n*m; y0=n_bonus+n_over; N=y0+n
    c=np.zeros(N)

    for i,name in enumerate(names):
        card=db[name]
        pv=cpp[name]/100.0
        for j,cat in enumerate(cats):
            c[i*m+j] = -(card["rates"][cat]*pv)
            c[n_bonus+i*m+j] = -(1.0*pv)
        ann = card["anniversary_points"]*pv if mode=="Ongoing-year economics" else 0.0
        c[y0+i] = card["annual_fee"] - credit_values[name] - ann

    cons=[]
    # Allocate every category dollar across bonus + overflow variables.
    for j,cat in enumerate(cats):
        row=np.zeros(N)
        for i in range(n):
            row[i*m+j]=1
            row[n_bonus+i*m+j]=1
        cons.append(LinearConstraint(row, spend[cat], spend[cat]))

    M=max(1.0,sum(spend.values()))
    for i in range(n):
        row=np.zeros(N)
        for j in range(m):
            row[i*m+j]=1; row[n_bonus+i*m+j]=1
        row[y0+i]=-M
        cons.append(LinearConstraint(row,-np.inf,0))

    row=np.zeros(N); row[y0:]=1
    cons.append(LinearConstraint(row,-np.inf,max_cards))

    # Bonus caps; cards/categories without caps cannot use overflow.
    upper=np.full(N,np.inf)
    for i,name in enumerate(names):
        for j,cat in enumerate(cats):
            if cat in db[name]["caps"]:
                row=np.zeros(N); row[i*m+j]=1
                cons.append(LinearConstraint(row,-np.inf,db[name]["caps"][cat]))
            else:
                upper[n_bonus+i*m+j]=0.0
    upper[y0:]=1
    integ=np.zeros(N,dtype=int); integ[y0:]=1
    res=milp(c=c,integrality=integ,bounds=Bounds(np.zeros(N),upper),constraints=cons)
    if not res.success: raise RuntimeError(res.message)

    selected=[names[i] for i in range(n) if res.x[y0+i]>.5]
    rows=[]; gross=0
    for i,name in enumerate(names):
        pv=cpp[name]/100.0
        for j,cat in enumerate(cats):
            b=res.x[i*m+j]; o=res.x[n_bonus+i*m+j]
            if b>.01:
                val=b*db[name]["rates"][cat]*pv; gross+=val
                rows.append([CATEGORIES[cat],name,b,db[name]["rates"][cat],val,"Bonus"])
            if o>.01:
                val=o*pv; gross+=val
                rows.append([CATEGORIES[cat],name,o,1.0,val,"Post-cap"])
    fees=sum(db[n]["annual_fee"] for n in selected)
    credits=sum(credit_values[n] for n in selected)
    ann=sum(db[n]["anniversary_points"]*(cpp[n]/100) for n in selected) if mode=="Ongoing-year economics" else 0
    net=gross+credits+ann-fees
    return selected,pd.DataFrame(rows,columns=["Category","Card","Spend","Multiplier","Reward value","Tranche"]),gross,fees,credits,ann,net

st.markdown('<div class="hero"><h1>CardOpt</h1><p>Quantitative credit-card portfolio optimization • recurring rewards, fees, credits and wallet complexity</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Model settings")
    max_cards=st.slider("Maximum cards",1,len(CARD_DB),3)
    baseline=st.number_input("Comparison baseline (%)",0.0,10.0,2.0,.25)
    mode=st.radio("Analysis horizon",["Ongoing-year economics","First-year recurring economics"],
                  help="First-year mode excludes anniversary points. Sign-up bonuses are intentionally excluded.")
    st.caption("Point valuations and benefit values are assumptions—not issuer guarantees.")

tabs=st.tabs(["Spending profile","Benefit values","Point assumptions","Methodology"])

with tabs[0]:
    st.subheader("Projected annual spending")
    st.caption("Use annual amounts. Splitting direct travel from portal travel matters because reward rates differ.")
    spend={}
    cols=st.columns(2)
    for k,(key,label) in enumerate(CATEGORIES.items()):
        with cols[k%2]:
            spend[key]=st.number_input(label,min_value=0.0,value=float(DEFAULT_SPEND[key]),step=500.0,key="sp_"+key)

with tabs[1]:
    st.subheader("What are the credits worth to you?")
    st.caption("Enter only value you would receive naturally. A $100 advertised credit is not automatically worth $100.")
    credit_values={}
    defaults={"Chase Sapphire Reserve":300.0,"American Express Gold":250.0,"Capital One Venture X":300.0}
    for name,card in CARD_DB.items():
        if card["credit_max"]>0:
            credit_values[name]=st.slider(name,0.0,float(card["credit_max"]),
                                          min(defaults.get(name,0.0),float(card["credit_max"])),10.0)
        else: credit_values[name]=0.0

with tabs[2]:
    st.subheader("Point valuation assumptions")
    st.caption("Cents per point/mile. Change these to stress-test the result.")
    cpp={}
    for name,card in CARD_DB.items():
        cpp[name]=st.slider(name,0.50,2.50,float(card["default_cpp"]),0.05,key="cpp_"+name)

with tabs[3]:
    st.subheader("Model scope")
    st.write("The optimizer jointly chooses a card portfolio and routes category spending to maximize modeled annual net economic value.")
    st.markdown("""
**Included:** category rewards, annual fees, user-valued recurring credits, anniversary points in ongoing-year mode, bonus-category caps, wallet-size constraint, and a cash-back comparison baseline.

**Excluded:** sign-up bonuses, APR/interest, balance transfers, credit-score effects, approval odds, taxes, merchant coding uncertainty, transfer-partner award availability, and qualitative perks unless the user assigns them a value.

**Important:** rewards should never justify carrying interest-bearing debt. This is an educational/research model, not individualized financial advice.
""")

run=st.button("Optimize portfolio",type="primary",use_container_width=True)

if run or "ran" not in st.session_state:
    st.session_state.ran=True
    if sum(spend.values())<=0:
        st.error("Enter at least some annual spending.")
    else:
        selected,alloc,gross,fees,credits,ann,net=solve(spend,copy.deepcopy(CARD_DB),cpp,credit_values,max_cards,mode)
        total=sum(spend.values()); base=total*baseline/100; uplift=net-base
        st.divider()
        st.subheader("Optimization report")
        st.write("**Recommended portfolio:** " + " + ".join(selected))

        a,b,c,d=st.columns(4)
        cards=[("Net annual value",f"${net:,.0f}"),("Effective yield",f"{net/total:.2%}"),
               (f"{baseline:.2f}% baseline",f"${base:,.0f}"),("Incremental value",f"{uplift:+,.0f}")]
        for col,(lab,val) in zip([a,b,c,d],cards):
            with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)

        st.subheader("Spending strategy")
        show=alloc.copy()
        show["Spend"]=show["Spend"].map(lambda x:f"${x:,.0f}")
        show["Multiplier"]=show["Multiplier"].map(lambda x:f"{x:g}x")
        show["Reward value"]=show["Reward value"].map(lambda x:f"${x:,.0f}")
        st.dataframe(show,use_container_width=True,hide_index=True)

        st.subheader("Portfolio economics")
        econ=pd.DataFrame({"Component":["Gross reward value","User-valued recurring credits","Anniversary value","Annual fees","Net annual value"],
                           "Amount":[gross,credits,ann,-fees,net]})
        st.bar_chart(econ.set_index("Component"))

        st.subheader("Wallet complexity frontier")
        frontier=[]
        for k in range(1,len(CARD_DB)+1):
            sel,_,_,_,_,_,nval=solve(spend,copy.deepcopy(CARD_DB),cpp,credit_values,k,mode)
            frontier.append({"Maximum cards":k,"Net annual value":nval,"Portfolio":" + ".join(sel)})
        fdf=pd.DataFrame(frontier)
        fdf["Marginal value of extra complexity"]=fdf["Net annual value"].diff()
        st.line_chart(fdf.set_index("Maximum cards")["Net annual value"])
        st.dataframe(fdf.style.format({"Net annual value":"${:,.0f}","Marginal value of extra complexity":"${:,.0f}"}),
                     use_container_width=True,hide_index=True)

        st.subheader("Interpretation")
        st.write(f"Under these assumptions, the model estimates **${net:,.0f}** of recurring annual net value, "
                 f"or **{net/total:.2%}** of the ${total:,.0f} spending profile. "
                 f"That is **${uplift:,.0f}** relative to the selected {baseline:.2f}% comparison baseline.")
        st.caption("Results are model outputs, not guaranteed realized returns. Card terms and rewards programs can change.")

st.divider()
st.caption("CardOpt research prototype • Mixed-Integer Linear Programming • Verify current issuer terms before relying on results.")
