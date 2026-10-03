# Plaid setup for CardOpt

CardOpt uses Plaid Hosted Link plus the Transactions product.

## Why Hosted Link

Streamlit does not own a conventional browser frontend callback lifecycle. Hosted Link lets Plaid host the connection experience and lets CardOpt retrieve the completed Link result from the backend using `/link/token/get`.

## 1. Start in Sandbox

Create or open your Plaid developer account and copy your Sandbox `client_id` and secret.

In Streamlit Community Cloud:

1. Open the deployed CardOpt app.
2. Open **App settings → Secrets**.
3. Add:

```toml
PLAID_CLIENT_ID = "your_client_id"
PLAID_SECRET = "your_sandbox_secret"
PLAID_ENV = "sandbox"
CARDOPT_APP_URL = "https://your-cardopt-subdomain.streamlit.app"
```

4. Save the secrets and reboot the app.
5. Open **Spending data → Connect with Plaid (Beta)**.
6. Either:
   - press **Start secure Plaid connection**, or
   - press **Load Plaid Sandbox demo** to test the backend quickly.

Standard Sandbox Link credentials:

- username: `user_good`
- password: `pass_good`

For transaction-rich Sandbox data:

- username: `user_transactions_dynamic`
- password: any nonblank value

## 2. Hosted Link workflow

When you press **Start secure Plaid connection**:

1. CardOpt calls `/link/token/create` with `products=["transactions"]` and a Hosted Link configuration.
2. Plaid returns a `hosted_link_url` and `link_token`.
3. CardOpt opens Hosted Link in a new browser tab so the original Streamlit session remains alive.
4. After completing Link, return to the original CardOpt tab.
5. Press **Check Plaid connection**.
6. CardOpt calls `/link/token/get`, reads `link_sessions[].results.item_add_results[].public_token`, exchanges the public token via `/item/public_token/exchange`, then can call `/transactions/sync`.

The current-response parsing in step 6 is important. Older CardOpt builds primarily checked top-level `results`, but current Hosted Link responses place session results under `link_sessions`.

## 3. Production

Before switching to real financial data:

1. Obtain the required Plaid Production access for Transactions.
2. Register the exact OAuth redirect URL in Plaid Dashboard.
3. Update Streamlit Secrets:

```toml
PLAID_CLIENT_ID = "your_client_id"
PLAID_SECRET = "your_production_secret"
PLAID_ENV = "production"
CARDOPT_APP_URL = "https://your-cardopt-subdomain.streamlit.app"
PLAID_REDIRECT_URI = "https://your-cardopt-subdomain.streamlit.app"
```

4. Test OAuth and non-OAuth institutions.
5. Add a dedicated backend before treating this as production financial infrastructure.

## Production security work still required

The research prototype intentionally does not pretend Streamlit session memory is a production token vault. A production deployment should add:

- authenticated users;
- encrypted access-token storage;
- Plaid webhooks for Link and Transactions updates;
- update mode for Items that need re-authentication;
- duplicate-Item handling;
- durable transaction storage and reconciliation;
- deletion/revocation controls;
- privacy policy and data-retention policy;
- security review and secrets rotation.

CardOpt's in-app **Disconnect and revoke Plaid access** action calls Plaid `/item/remove` and then clears the active Streamlit session data.
