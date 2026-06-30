# Dataset Registry

This project uses 3 realistic domains for analytics coverage.

## Domain 1: Retail / Sales
- Source: UCI Online Retail II + curated product hierarchy enrichments.
- Size target: 50k+ rows, 8-15 columns.
- Key fields: invoice date, quantity, unit price, country, customer id, product description.
- Google Sheet layout:
  - `sales_transactions`
  - `product_master`

## Domain 2: HR Analytics
- Source: IBM HR Analytics Attrition dataset + compensation enrichments.
- Size target: 1k+ rows, 30+ columns.
- Key fields: department, attrition, monthly income, years at company, job role.
- Google Sheet layout:
  - `employee_master`
  - `department_targets`

## Domain 3: Finance / Marketing Mix
- Source: public campaign spend + revenue conversion dataset.
- Size target: 10k+ rows, 12+ columns.
- Key fields: date, channel, spend, clicks, conversions, revenue.
- Google Sheet layout:
  - `channel_performance`
  - `monthly_targets`

## Provenance and Documentation Rules
For each ingested sheet, store metadata in `data/artifacts/dataset_catalog.json`:
- Source URL
- License
- Ingestion timestamp
- Row/column profile
- Hash signature

## Preparing Google Sheets
Use `scripts/prepare_datasets.py` to normalize local datasets before upload to Google Sheets.
