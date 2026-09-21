# CardOpt 2.0 — Explainable Credit-Card Decision Engine

CardOpt is an educational/research credit-card portfolio decision engine. It combines a mixed-integer linear programming (MILP) optimizer with real-spending ingestion and an economic explanation layer.

## What is different in 2.0

CardOpt does not stop at “which card earns the most.”

It can analyze:

- **Portfolio construction:** which modeled cards belong in the wallet.
- **Spend routing:** where each spending category should go.
- **Real-spending inputs:** manual annual estimates, CSV transaction imports, or Plaid Hosted Link.
- **Marginal value:** how much a selected card adds after the rest of the wallet is re-optimized.
- **Opportunity cost:** what value is lost when an omitted card is forced into the wallet.
- **Diminishing returns:** how much value is added by allowing card 2, 3, 4, etc.
- **Decision boundaries:** approximate annual-fee thresholds where the modeled optimum changes.
- **Robustness:** whether the portfolio survives changes in point values, benefits, and category spending.
- **Practical utility:** optional user-defined complexity and switching costs.
- **Macroeconomic lens:** a broad nominal-spending stress scenario, explicitly labeled as a scenario rather than an inflation forecast.
- **CardOpt Learn:** credit basics, rewards, and decision economics.

## Core model

CardOpt's core recurring economic value is:

`reward value + user-valued recurring benefits + applicable anniversary value - annual fees`

The MILP jointly chooses card-selection binaries and category-spend allocation variables under wallet-size, card-activation, reward-cap, and modeled-credit constraints.

Advanced Mode can optionally optimize a practical decision-utility score by subtracting explicit user-defined wallet-complexity and switching frictions. The app still reports underlying card economics separately.

## Spending data

### Manual
No account connection required.

### CSV
Upload a transaction CSV. CardOpt looks for common fields such as Amount, Date, Merchant/Description, Plaid PFC fields, and MCC. It maps transactions into CardOpt reward categories and shows the mapping for review.

**Important:** inferred categories are not guaranteed to match an issuer's final merchant coding. Users can review and edit annual category totals before optimization.

### Plaid (Beta)
CardOpt uses Plaid **Hosted Link** when credentials are configured. The app:

1. calls `/link/token/create` with the Transactions product and Hosted Link;
2. sends the user to Plaid's hosted connection flow;
3. retrieves a completed public token with `/link/token/get`;
4. exchanges it with `/item/public_token/exchange`;
5. imports transaction updates with `/transactions/sync`.

The prototype intentionally keeps Plaid access tokens in the active Streamlit server session only. A production deployment should use encrypted persistent token storage, update-mode handling, deletion/revocation workflows, webhook validation, audit logging, and a formal security/privacy review.

Plaid Sandbox uses mock data. Production uses real financial data and requires appropriate Plaid access/configuration.

## Plaid setup

In Streamlit Community Cloud, open **App settings → Secrets** and add:

```toml
PLAID_CLIENT_ID = "..."
PLAID_SECRET = "..."
PLAID_ENV = "sandbox"
CARDOPT_APP_URL = "https://your-app.streamlit.app"
```

For OAuth-capable production institutions, configure an approved redirect URI in Plaid and add:

```toml
PLAID_REDIRECT_URI = "https://your-approved-redirect.example.com"
```

Do not commit real secrets.

## Data freshness

Card terms are still human-approved. The included GitHub Action runs a daily **source monitor** against official issuer pages and can open a review issue when a source no longer matches the watch terms. It intentionally does **not** auto-edit live card terms.

This design avoids silently corrupting the optimizer when an issuer page changes layout, runs a temporary offer, or changes language.

## Testing

The package includes regression tests for:

- flat-rate cash-back arithmetic;
- annual-fee tradeoffs;
- Amex Gold supermarket cap behavior;
- Venture X travel-credit reward treatment;
- Sapphire Reserve travel-credit reward treatment;
- transaction classification examples;
- complexity-cost behavior.

Run:

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

## Scope and limitations

CardOpt is a recurring-value educational/research model, not individualized financial advice.

The current core model intentionally excludes welcome offers, APR/interest, approval odds, credit-score effects, taxes, exact issuer merchant coding, transfer-partner award availability, and unpriced qualitative perks.

If a user expects to carry interest-bearing balances, borrowing cost should be considered before rewards optimization.

Card terms and issuer rules change. Verify current issuer terms before acting.
