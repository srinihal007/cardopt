# CardOpt 2.1 — Explainable Credit-Card Decision Engine

CardOpt is an educational/research application that treats credit-card wallet construction as an explainable optimization problem. The project is intentionally broader than a rewards lookup: it connects spending behavior to a mixed-integer portfolio model, then analyzes the economics of the decision.

## What differentiates CardOpt

CardOpt integrates six layers in one workflow:

1. **Observe** — manual spending, bank/card CSV, PNG/JPG transaction screenshot OCR, or optional Plaid Transactions.
2. **Classify** — map transaction data into CardOpt reward categories with confidence/review before optimization.
3. **Optimize** — jointly choose cards and category-level spending allocation with a MILP.
4. **Explain** — calculate marginal card value, opportunity cost, current-wallet comparisons, and economic bridges.
5. **Stress-test** — evaluate robustness, wallet-complexity frontiers, and decision boundaries.
6. **Monitor** — scheduled source monitoring flags possible issuer-term changes for human review rather than silently changing the live model.

CardOpt does **not** claim that each individual technique is unprecedented. The project's differentiation is the integrated, transparent decision workflow and the explicit separation of issuer facts, user inputs, model assumptions, and calculated outputs.

## Mobile screenshot import

The spending-data screen accepts `CSV`, `PNG`, `JPG`, and `JPEG` files. Image uploads are processed with Tesseract OCR on the app server. CardOpt displays the extracted rows in an editable table so the user can correct or remove OCR mistakes before the data reaches the classifier or optimizer.

Streamlit Community Cloud needs the included `packages.txt` file so `tesseract-ocr` is installed.

## Plaid Hosted Link

CardOpt uses Plaid Hosted Link because Streamlit does not own a conventional frontend callback lifecycle.

### Sandbox setup

Add these values in **Streamlit Community Cloud → App settings → Secrets**:

```toml
PLAID_CLIENT_ID = "..."
PLAID_SECRET = "..."
PLAID_ENV = "sandbox"
CARDOPT_APP_URL = "https://your-app.streamlit.app"
```

Then reboot the app.

In Sandbox, CardOpt offers two paths:

- **Start secure Plaid connection** — opens Hosted Link in a new tab. After completing Link, return to the original CardOpt tab and press **Check Plaid connection**.
- **Load Plaid Sandbox demo** — creates a Sandbox Transactions Item directly so you can demo/test the backend without walking through Link every time.

Standard Plaid Sandbox Link credentials are `user_good` / `pass_good`. For transaction-rich test data, Plaid also provides `user_transactions_dynamic` with any nonblank password.

### Important fix in 2.1

Plaid's current `/link/token/get` response stores Hosted Link results inside `link_sessions[].results.item_add_results`. CardOpt 2.1 parses that current response shape and retains a backwards-compatible fallback. The previous build incorrectly looked primarily at top-level `results`, which could make a completed Link session look unfinished.

### Production setup

Before Production:

- obtain the appropriate Plaid Production/Transactions access;
- set `PLAID_ENV = "production"`;
- register the exact OAuth redirect URL in the Plaid Dashboard;
- set `PLAID_REDIRECT_URI` to that approved URL;
- use encrypted persistent access-token storage and authenticated users;
- add webhook processing for Link/Transactions updates;
- provide deletion/revocation controls and perform a formal security/privacy review.

The current prototype keeps access tokens only in the active Streamlit server session. Its disconnect action also calls Plaid `/item/remove` before clearing local session data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For screenshot OCR on macOS, install Tesseract if needed:

```bash
brew install tesseract
```

## Tests

```bash
pytest -q
```

The test suite covers core optimization arithmetic, capped rewards, travel-credit treatment, complexity cost, transaction classification, current Hosted Link result parsing, and OCR text parsing.

## Research scope

CardOpt is a recurring-value model. Welcome offers, APR/interest, approval odds, credit-score effects, taxes, merchant-coding uncertainty, transfer-partner availability, and unpriced qualitative perks are outside the optimization unless explicitly represented as user assumptions.

This project is educational/research software, not individualized financial advice. Issuer terms can change; verify current terms before acting.
