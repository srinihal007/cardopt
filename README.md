# CardOpt V5

CardOpt is a mixed-integer linear programming (MILP) research prototype that jointly selects a credit-card portfolio and routes category spending to maximize estimated recurring net economic value.

## V5 improvements
- separates verified issuer terms from user/model assumptions
- itemizes recurring benefit utilization instead of assuming face value
- distinguishes first-year recurring vs ongoing annual economics
- adds conservative/base/upside robustness testing
- adds an assumption audit and issuer-source links
- improves wallet, category-routing, economic-bridge, and complexity-frontier presentation
- supports downloadable summary and allocation files

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Card terms change. Re-verify issuer terms before publication or reliance. Welcome offers, APR/interest, approval odds, credit-score effects, taxes, merchant coding uncertainty, and transfer-partner award availability are outside the recurring optimization.
