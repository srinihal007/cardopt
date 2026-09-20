# CardOpt — Credit Card Rewards Portfolio Optimizer

CardOpt is a research prototype that uses mixed-integer linear programming (MILP) to jointly:
1. select a credit-card portfolio, and
2. route category-level spending across the selected cards.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model design
The objective maximizes recurring annual modeled economic value:

reward value + user-valued recurring credits + applicable anniversary value - annual fees.

The model includes a maximum-card constraint and piecewise handling for published bonus-category caps. It separates issuer terms from subjective assumptions such as cents-per-point and the personal value of statement credits.

## Scope
Sign-up bonuses, APR/interest, approval odds, credit-score effects, taxes, merchant coding uncertainty, and transfer-partner award availability are intentionally excluded from the recurring optimization.

Card terms change. Verify the database against official issuer terms before publishing or relying on results.
