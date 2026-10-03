# CardOpt — Project Overview

## 30-second summary

CardOpt is an explainable credit-card portfolio decision engine. It combines real or user-estimated spending with mixed-integer linear programming, then goes beyond the optimized answer by measuring marginal value, opportunity cost, wallet complexity, decision boundaries, and robustness.

The central question is not simply *Which card earns the highest rewards?* It is:

> Given a person's spending behavior, card constraints, annual fees, reward caps, benefit utilization, point-value assumptions, and desired wallet complexity, what portfolio is economically preferable, why, and what would cause that decision to change?

## Six-layer decision stack

1. **Observe** — manual spending, CSV exports, PNG/JPG transaction screenshots, or optional Plaid Transactions.
2. **Classify** — convert transaction data into CardOpt reward categories with user review and confidence labels.
3. **Optimize** — jointly select cards and allocate spending using a MILP.
4. **Explain** — measure marginal card value, opportunity cost, current-wallet improvement, and the economic bridge from rewards to net value.
5. **Stress-test** — evaluate robustness, diminishing returns, wallet-complexity frontiers, and decision boundaries.
6. **Monitor** — check official issuer sources for possible changes and require human review before live card data is changed.

## Why the math matters

CardOpt uses binary variables to decide whether a card belongs in the wallet and continuous variables to allocate spending. Constraints enforce spending conservation, selected-card routing, reward caps, credit caps, and wallet-size limits.

The model explicitly separates:

- **Issuer facts** — fees, published rates, caps, eligible credits, anniversary rewards;
- **User inputs** — spending, wallet size, benefit values, preferences;
- **Model assumptions** — cents-per-point values, benchmark rate, transaction-category inference;
- **Calculated results** — portfolio, routing, reward value, modeled net value, tradeoff metrics.

## What changed after competitive feedback

The project started as a rewards-optimization idea. Competitive research showed that reward lookup and card recommendations already exist in the market. Instead of abandoning the project, CardOpt moved deeper into decision science:

- real-spend ingestion instead of only self-reported estimates;
- portfolio construction instead of one-card ranking;
- marginal contribution instead of headline rewards;
- counterfactual opportunity cost instead of a single recommendation;
- robustness and decision boundaries instead of pretending assumptions are certain;
- dynamic source monitoring instead of treating card terms as permanent.

That evolution is part of the project's research story.

## Validation

The repository includes regression tests for:

- flat-rate cash-back arithmetic;
- annual-fee tradeoffs;
- capped grocery rewards and overflow;
- Venture X credited-spend treatment;
- Sapphire Reserve credited-spend treatment;
- transaction classification;
- wallet-complexity costs;
- current Plaid Hosted Link response parsing;
- OCR transaction-row extraction.

## Honest limitations

CardOpt is not a complete financial-advice system. The recurring model intentionally excludes welcome offers, APR/interest, approval odds, credit-score effects, taxes, merchant-coding uncertainty, transfer-partner award availability, and unpriced qualitative perks unless they are explicitly represented as assumptions.

Screenshot OCR is probabilistic and always requires review. Plaid transaction categories and CardOpt merchant heuristics are not guaranteed to match an issuer's ultimate rewards coding.

CardOpt is educational/research software, not individualized financial advice.
