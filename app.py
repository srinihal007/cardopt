import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

st.set_page_config(page_title="CardOpt", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--ink:#0b1736;--muted:#64718c;--blue:#1769e0;--violet:#673bd0;--line:#dfe8f5;--soft:#f5f9ff}
.stApp{background:linear-gradient(180deg,#fbfdff 0%,#ffffff 42%)}
.block-container{max-width:1180px;padding-top:1.25rem;padding-bottom:4rem}
[data-testid="stSidebar"]{border-right:1px solid #e6edf7;background:#fbfdff}
#MainMenu,footer{visibility:hidden}
.hero61{position:relative;overflow:hidden;padding:3.2rem 3.1rem;border:1px solid #dce8f7;border-radius:30px;
background:radial-gradient(circle at 88% 20%,rgba(73,148,255,.20),transparent 30%),
linear-gradient(135deg,#f9fcff 0%,#edf6ff 55%,#f6f3ff 100%);box-shadow:0 20px 60px rgba(30,73,125,.08)}
.brand{font-size:1.22rem;font-weight:800;letter-spacing:-.04em;color:var(--ink);margin-bottom:3.4rem}
.brand span{color:#2866d8}
.eyebrow{font-size:.74rem;letter-spacing:.15em;text-transform:uppercase;color:#51678f;font-weight:800}
.hero61 h1{font-size:3.65rem;line-height:1.01;letter-spacing:-.055em;color:var(--ink);margin:.55rem 0 .9rem;max-width:690px}
.hero61 h1 span{color:#2e62d6}
.hero61 p{font-size:1.08rem;line-height:1.7;color:#52617d;max-width:660px}
.stats{display:flex;gap:0;margin-top:2.1rem;flex-wrap:wrap}
.stat{padding-right:1.45rem;margin-right:1.45rem;border-right:1px solid #cfdbeb}
.stat:last-child{border:0}.stat b{display:block;color:var(--ink);font-size:.95rem}.stat small{color:#70809c}
.orbit{position:absolute;right:2.8rem;top:6.3rem;width:270px;height:210px}
.cardshape{position:absolute;width:205px;height:126px;border-radius:17px;box-shadow:0 18px 35px rgba(16,35,75,.20);
border:1px solid rgba(255,255,255,.55);padding:18px;color:white;font-weight:750;letter-spacing:.04em}
.c1{right:32px;top:0;transform:rotate(8deg);background:linear-gradient(135deg,#071d47,#17589a)}
.c2{right:72px;top:60px;transform:rotate(-7deg);background:linear-gradient(135deg,#b58a3b,#f0d791);color:#17203a}
.c3{right:0;top:105px;transform:rotate(7deg);background:linear-gradient(135deg,#243650,#0e1e36)}
.section61{text-align:center;padding:3rem 0 1rem}.section61 .eyebrow{margin-bottom:.65rem}
.section61 h2{font-size:2.15rem;letter-spacing:-.045em;color:var(--ink);margin:0}.section61 p{color:var(--muted)}
.modecard{height:100%;padding:1.7rem;border:1px solid #dce7f5;border-radius:22px;background:rgba(255,255,255,.86);
box-shadow:0 12px 32px rgba(40,75,120,.055)}
.modecard.advanced{background:linear-gradient(145deg,#fff,#f7f4ff);border-color:#e4dcf8}
.modeicon{width:44px;height:44px;border-radius:13px;display:flex;align-items:center;justify-content:center;background:#eaf3ff;
font-size:1.25rem;font-weight:800;color:#1769e0}.advanced .modeicon{background:#eee8ff;color:#673bd0}
.modecard h3{color:var(--ink);font-size:1.4rem;margin:.85rem 0 .15rem}.modecard .tag{font-weight:700;color:#2665cf}
.modecard p{color:#5d6a84;line-height:1.55}.ticks{line-height:1.9;color:#52617d;font-size:.91rem}
.kpi{padding:1.15rem;border:1px solid #dfe8f5;border-radius:18px;min-height:145px;background:#fff}
.kpi .label{font-size:.84rem;color:#6b7890;line-height:1.3}.kpi .value{font-size:2.1rem;font-weight:760;letter-spacing:-.04em;margin:.3rem 0;color:var(--ink)}.kpi .sub{font-size:.78rem;color:#7c889e}
.wallet{padding:1rem 1.1rem;border:1px solid #dfe8f5;border-radius:16px;margin:.55rem 0;background:#fff}
.wallet .name{font-size:1.05rem;font-weight:720;color:var(--ink)}.wallet .meta{font-size:.84rem;color:#758199;margin-top:.22rem}
.note{font-size:.88rem;color:#738099;margin-top:-.3rem;margin-bottom:1rem}
.source{padding:.75rem .9rem;border:1px solid #e0e8f3;border-radius:13px;margin:.4rem 0;background:#fff}
div[data-testid="stMetric"]{border:1px solid #dfe8f5;padding:12px;border-radius:15px;background:#fff}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden}
div.stButton > button{border-radius:12px;font-weight:700;min-height:46px}

.topbrand{font-size:1.55rem;font-weight:850;letter-spacing:-.05em;color:#0b1736;padding:.35rem 0}.topbrand span{color:#1769e0}
.pagehero{padding:2rem 0 1.2rem}.pagehero h1{font-size:2.8rem;letter-spacing:-.05em;color:#0b1736;margin:.2rem 0}.pagehero p{font-size:1.02rem;color:#68758e;max-width:760px}
.infoCard{padding:1.3rem;border:1px solid #dfe8f5;border-radius:18px;background:#fff;height:100%;box-shadow:0 8px 25px rgba(40,75,120,.04)}
.infoCard h3{color:#0b1736;margin:.25rem 0 .45rem}.infoCard p{color:#65728a;line-height:1.55}
.creditcard{height:178px;border-radius:18px;padding:1.25rem;color:white;box-shadow:0 16px 34px rgba(18,38,76,.16);margin-bottom:.9rem;position:relative;overflow:hidden}
.creditcard:after{content:"";position:absolute;width:180px;height:180px;border-radius:50%;right:-70px;top:-80px;background:rgba(255,255,255,.09)}
.creditcard .issuer{font-size:.73rem;opacity:.75;text-transform:uppercase;letter-spacing:.1em}.creditcard .cardname{font-size:1.15rem;font-weight:780;margin-top:.35rem}.creditcard .chip{width:35px;height:26px;border-radius:6px;background:linear-gradient(135deg,#d9bf72,#f6e8ae);margin-top:2rem}.creditcard .fee{position:absolute;bottom:1rem;left:1.25rem;font-size:.78rem;opacity:.78}
.chase{background:linear-gradient(135deg,#071c43,#1765a8)}.amex{background:linear-gradient(135deg,#a27a2f,#ead28d);color:#17203a}.capone{background:linear-gradient(135deg,#263c5b,#0b1b32)}.citi{background:linear-gradient(135deg,#163c78,#2875bb)}.freedom{background:linear-gradient(135deg,#235f9f,#61a3d8)}.wells{background:linear-gradient(135deg,#76192b,#b2263e)}
@media(max-width:850px){.hero61{padding:2rem 1.4rem}.hero61 h1{font-size:2.65rem}.orbit{display:none}.brand{margin-bottom:2rem}}
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
"benefits":[("Annual travel credit",300,"Eligible travel purchases; credited purchases do not earn points.")],
"source":"https://creditcards.chase.com/rewards-credit-cards/sapphire/reserve"},
"American Express Gold":{"fee":325,"cpp":1.50,
"rates":{"dining":4,"us_supermarkets":4,"airfare_direct":3,"hotels_direct":1,"portal_flights":3,"portal_hotels":5,"drugstores":1,"other":1},
"caps":{"dining":50000,"us_supermarkets":25000},"ann":0,
"benefits":[("Dining credit",120,"Eligible partners; monthly limits and enrollment/terms apply."),
("Uber Cash",120,"U.S. Uber/Uber Eats; monthly limits and terms apply."),
("Resy credit",100,"Eligible U.S. Resy purchases; semiannual limits and terms apply."),
("Dunkin' credit",84,"Eligible U.S. Dunkin' purchases; monthly limits and enrollment/terms apply.")],
"source":"https://www.americanexpress.com/us/credit-cards/card/gold-card/"},
"Capital One Venture X":{"fee":395,"cpp":1.50,
"rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,"portal_flights":5,"portal_hotels":10,"drugstores":2,"other":2},
"caps":{},"ann":10000,
"benefits":[("Capital One Travel credit",300,"Annual credit for eligible Capital One Travel purchases; rewards are not earned on the credit amount.")],
"source":"https://www.capitalone.com/credit-cards/venture-x/"},
"Citi Double Cash":{"fee":0,"cpp":1.00,
"rates":{"dining":2,"us_supermarkets":2,"airfare_direct":2,"hotels_direct":2,"portal_flights":2,"portal_hotels":5,"drugstores":2,"other":2},
"caps":{},"ann":0,"benefits":[],
"source":"https://www.citi.com/credit-cards/citi-double-cash-credit-card"},
"Chase Freedom Unlimited":{"fee":0,"cpp":1.00,
"rates":{"dining":3,"us_supermarkets":1.5,"airfare_direct":1.5,"hotels_direct":1.5,"portal_flights":5,"portal_hotels":5,"drugstores":3,"other":1.5},
"caps":{},"ann":0,"benefits":[],
"source":"https://creditcards.chase.com/cash-back-credit-cards/freedom/unlimited"},
"Wells Fargo Active Cash":{"fee":0,"cpp":1.00,"rates":{k:2 for k in CATS},"caps":{},"ann":0,"benefits":[],
"source":"https://www.wellsfargo.com/credit-cards/active-cash/"}}

CASHLIKE={"Citi Double Cash","Chase Freedom Unlimited","Wells Fargo Active Cash"}
DEFAULT={"dining":6000,"us_supermarkets":5000,"airfare_direct":2500,"hotels_direct":1500,"portal_flights":500,"portal_hotels":500,"drugstores":1000,"other":8000}
def money(x): return f"${x:,.0f}"

def solve(spend,cpp,bens,maxcards,horizon, allowed_cards=None, required_cards=None):
    names=list(DB); cats=list(CATS); n=len(names); m=len(cats); q=n*m; N=2*q+n
    allowed_cards=set(names if allowed_cards is None else allowed_cards)
    required_cards=set([] if required_cards is None else required_cards)
    c=np.zeros(N); ub=np.full(N,np.inf); integ=np.zeros(N)
    X=lambda i,j:i*m+j; Z=lambda i,j:q+i*m+j; Y=lambda i:2*q+i
    for i,nm in enumerate(names):
        d=DB[nm]
        for j,cat in enumerate(cats):
            c[X(i,j)]=-d["rates"][cat]*cpp[nm]/100
            c[Z(i,j)]=-cpp[nm]/100
            if cat not in d["caps"]: ub[Z(i,j)]=0
        ann=d["ann"]*cpp[nm]/100 if horizon=="Ongoing annual economics" else 0
        c[Y(i)]=d["fee"]-bens[nm]-ann+1e-5
        ub[Y(i)]=1 if nm in allowed_cards else 0
        integ[Y(i)]=1
    rows=[]; lo=[]; hi=[]
    for j,cat in enumerate(cats):
        r={}
        for i in range(n): r[X(i,j)]=1; r[Z(i,j)]=1
        rows.append(r); lo.append(spend[cat]); hi.append(spend[cat])
    M=max(sum(spend.values()),1)
    for i in range(n):
        r={Y(i):-M}
        for j in range(m): r[X(i,j)]=1; r[Z(i,j)]=1
        rows.append(r); lo.append(-np.inf); hi.append(0)
    for i,nm in enumerate(names):
        for cat,cap in DB[nm]["caps"].items():
            rows.append({X(i,cats.index(cat)):1,Y(i):-cap}); lo.append(-np.inf); hi.append(0)
    rows.append({Y(i):1 for i in range(n)}); lo.append(-np.inf); hi.append(maxcards)
    for i,nm in enumerate(names):
        if nm in required_cards:
            rows.append({Y(i):1}); lo.append(1); hi.append(1)
    A=lil_matrix((len(rows),N))
    for rr,d in enumerate(rows):
        for col,val in d.items(): A[rr,col]=val
    res=milp(c,integrality=integ,bounds=Bounds(np.zeros(N),ub),
             constraints=LinearConstraint(A.tocsr(),np.array(lo),np.array(hi)))
    if not res.success:return None
    v=res.x; selected=[names[i] for i in range(n) if v[Y(i)]>.5]
    alloc=[]; gross=0; cg={x:0 for x in names}; cs={x:0 for x in names}
    for i,nm in enumerate(names):
        for j,cat in enumerate(cats):
            for amt,rate,tier in [(max(v[X(i,j)],0),DB[nm]["rates"][cat],"Bonus"),(max(v[Z(i,j)],0),1,"Post-cap")]:
                if amt>1e-6:
                    val=amt*rate*cpp[nm]/100; gross+=val; cg[nm]+=val; cs[nm]+=amt
                    alloc.append([CATS[cat],nm,amt,rate,cpp[nm],val,tier])
    fees=sum(DB[x]["fee"] for x in selected); ben=sum(bens[x] for x in selected)
    ann=sum(DB[x]["ann"]*cpp[x]/100 for x in selected) if horizon=="Ongoing annual economics" else 0
    details=[]
    for x in selected:
        av=DB[x]["ann"]*cpp[x]/100 if horizon=="Ongoing annual economics" else 0
        details.append({"Card":x,"Spend":cs[x],"Rewards":cg[x],"Benefits":bens[x],"Anniversary":av,
                        "Fee":DB[x]["fee"],"Net":cg[x]+bens[x]+av-DB[x]["fee"]})
    return {"selected":selected,"allocation":alloc,"gross":gross,"fees":fees,"benefits":ben,
            "anniversary":ann,"net":gross+ben+ann-fees,"details":details}

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
    n0,n1,n2,n3,n4,n5=st.columns([3.2,1,0.75,0.85,0.75,1])
    with n0: st.markdown('<div class="topbrand">Card<span>Opt</span></div>',unsafe_allow_html=True)
    with n1:
        if st.button("How It Works",use_container_width=True): st.session_state.page="How It Works"
    with n2:
        if st.button("Cards",use_container_width=True): st.session_state.page="Cards"
    with n3:
        if st.button("Research",use_container_width=True): st.session_state.page="Research"
    with n4:
        if st.button("About",use_container_width=True): st.session_state.page="About"
    with n5:
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
        st.markdown("### What makes this different")
        st.write("CardOpt compares portfolio economics rather than ranking cards one at a time. Fees, reward rates, caps, point values, benefits and wallet size can all change the optimal result.")
        if st.button("Choose an experience",type="primary"): st.session_state.page="Home"; st.rerun()
        st.stop()

    if st.session_state.page == "Cards":
        st.markdown('<div class="pagehero"><div class="eyebrow">Card library</div><h1>The cards currently modeled.</h1><p>A deliberately focused research set with transparent terms.</p></div>',unsafe_allow_html=True)
        styles={"Chase Sapphire Reserve":"chase","American Express Gold":"amex","Capital One Venture X":"capone","Citi Double Cash":"citi","Chase Freedom Unlimited":"freedom","Wells Fargo Active Cash":"wells"}
        issuers={"Chase Sapphire Reserve":"Chase","American Express Gold":"American Express","Capital One Venture X":"Capital One","Citi Double Cash":"Citi","Chase Freedom Unlimited":"Chase","Wells Fargo Active Cash":"Wells Fargo"}
        cardcols=st.columns(3)
        for ii,(nm,d) in enumerate(DB.items()):
            with cardcols[ii%3]:
                html='<div class="creditcard '+styles[nm]+'"><div class="issuer">'+issuers[nm]+'</div><div class="cardname">'+nm+'</div><div class="chip"></div><div class="fee">Annual fee $'+format(d["fee"],",")+'</div></div>'
                st.markdown(html,unsafe_allow_html=True)
                with st.expander("View modeled terms"):
                    st.write("Default model point value: **"+format(d["cpp"],".2f")+"¢**")
                    st.dataframe(pd.DataFrame([[CATS[k],str(v)+"x"] for k,v in d["rates"].items()],columns=["Category","Rate"]),hide_index=True,use_container_width=True)
                    if d["caps"]: st.write("Modeled caps: "+", ".join(CATS[k]+" $"+format(v,",") for k,v in d["caps"].items()))
                    if d["benefits"]:
                        st.write("Optional user-valued recurring benefits:")
                        for bn,face,note in d["benefits"]: st.write("• "+bn+": up to $"+format(face,","))
                    st.markdown("[Official issuer source]("+d["source"]+")")
        st.caption("Terms reviewed "+VERIFIED+". Issuer terms can change.")
        st.stop()

    if st.session_state.page == "Research":
        st.markdown('<div class="pagehero"><div class="eyebrow">Research</div><h1>Transparent by design.</h1><p>The optimization is useful only if its assumptions, constraints and data can be inspected.</p></div>',unsafe_allow_html=True)
        st.markdown("### Optimization model")
        st.write("CardOpt uses mixed-integer linear programming. Binary variables represent whether a card is selected. Continuous variables represent category-level spend allocated to each card.")
        st.latex(r"\max\; \text{reward value} + \text{user-valued benefits} + \text{anniversary value} - \text{annual fees}")
        st.markdown("### Constraints")
        st.write("Every dollar is allocated, spending flows only to selected cards, modeled reward caps are enforced, and the portfolio respects the user's wallet-size limit.")
        st.markdown("### Validation")
        st.write("Known-answer tests cover cash-back arithmetic, annual-fee tradeoffs, multi-card routing, reward caps and overflow, benefit valuation, and first-year versus ongoing anniversary treatment.")
        st.markdown("### Assumption taxonomy")
        st.write("Issuer facts come from card terms. User inputs include spending and benefit values. Model assumptions include point valuations. Calculated results include the optimized wallet and estimated net value.")
        st.markdown("### Deliberate exclusions")
        st.write("Welcome offers, APR and interest, approval odds, credit-score effects, taxes, merchant-coding uncertainty, transfer award availability and unpriced qualitative perks are outside the recurring model.")
        st.stop()

    if st.session_state.page == "About":
        st.markdown('<div class="pagehero"><div class="eyebrow">About CardOpt</div><h1>What should actually be in your wallet?</h1><p>CardOpt is an independent quantitative finance and optimization project built to explore that question.</p></div>',unsafe_allow_html=True)
        a1,a2=st.columns([1.25,1])
        with a1:
            st.markdown("### Why CardOpt exists")
            st.write("Comparing individual cards is relatively easy. Comparing a portfolio is harder because rewards, annual fees, spending caps, point values, benefits and overlapping categories interact. CardOpt models those interactions together.")
            st.markdown("### Built by Sri Nihal Tammana")
            st.write("CardOpt explores how mathematical optimization can make an everyday financial decision more transparent and easier to understand.")
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
    st.markdown("### CardOpt")
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

tabs=st.tabs(["Spending","Benefits","Reward assumptions","Research"])

with tabs[0]:
    st.subheader("Your annual spending")
    st.markdown('<div class="note">Use a typical year. CardOpt will decide where each dollar should go.</div>',unsafe_allow_html=True)
    spend={}; cols=st.columns(2)
    for i,(k,label) in enumerate(CATS.items()):
        with cols[i%2]: spend[k]=st.number_input(label,0.0,value=float(DEFAULT[k]),step=500.0,format="%.0f",key="s"+k)
    st.metric("Total annual card spending",money(sum(spend.values())))

bens={}
with tabs[1]:
    st.subheader("What are the benefits worth to you?")
    st.markdown('<div class="note">Restricted credits are not automatically worth face value. Start at $0 and add only value you realistically expect to use on purchases you would otherwise make.</div>',unsafe_allow_html=True)
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


st.markdown("### Make it yours")
st.caption("These controls let CardOpt reflect how you actually want to use credit cards.")
q1,q2,q3=st.columns(3)
with q1:
    fee_choice=st.selectbox("Annual fees",["Any fee if the math justifies it","No annual-fee cards only"])
with q2:
    portal_choice=st.selectbox("Travel portals",["I am willing to use portals","I prefer booking direct"])
with q3:
    reward_choice=st.selectbox("Reward style",["Maximize estimated value","Keep rewards simple"])

st.markdown("### CardOpt Compare")
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

st.markdown("### Trust & transparency")
tr1,tr2,tr3,tr4=st.columns(4)
tr1.metric("Official issuer sources","6 / 6")
tr2.metric("Restricted benefits","$0 default")
tr3.metric("Comparison baseline",f"{benchmark*100:.0f}% cash back")
tr4.metric("Terms reviewed",VERIFIED)
with st.expander("How CardOpt earns your trust"):
    st.write("• Card terms link to official issuer sources.")
    st.write("• Restricted credits start at $0 unless you say you would naturally use them.")
    st.write("• Point values are visible model assumptions.")
    st.write("• Welcome bonuses are excluded from recurring economics.")
    st.write("• Advanced Mode exposes allocations, assumptions, sensitivity and methodology.")
    st.write("• Estimates are not guarantees. Issuer terms, merchant coding and redemption values can change.")

if st.button("Optimize my wallet",type="primary",use_container_width=True):
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
