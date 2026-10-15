# Banking Loan Portfolio Analytics

Team 2 capstone, Accenture GTGH AI & Data programme (Code.Hub, Sep–Oct 2026).

A bank's loan data arrives as four messy delivery batches. This project cleans it in Databricks,
publishes a governed Gold layer, and serves it three ways: a Power BI report, a REST API with a web
dashboard, and an assistant that answers questions in plain language from the same Gold numbers.

## Architecture

```
CSV / JSON batches (Unity Catalog volume)
        │
        ▼
Databricks Lakeflow Declarative Pipeline (SQL)            pipelines/
  Bronze   7 streaming tables, one per source file
  Silver   standardisation, newest version wins, 37 quality rules, quarantine
  Gold     star schema, data-quality exceptions, reconciliation, KPIs, knowledge graph
        │
        ├──────────────► Power BI report (semantic model on Gold)          powerbi/
        │
        ▼
FastAPI on Azure Container Apps (dev and prod)                api/
  /analytics, /branches, /exceptions, /graph, /metadata     read-only Gold queries
  /assistant/query                                          LangGraph assistant, Azure OpenAI gpt-4.1-mini
  /app                                                      web dashboard (EN / ΕΛ)
```

- **Gold** (`data2026acc123456.team2`): dimensions (`dim_date`, `dim_branch`, `dim_product`,
  `dim_customer_segment`, `dim_risk_band`, `dim_application_status`), facts (`fact_loan_application`,
  `fact_repayment`, `fact_dq_exception`), KPI tables (`kpi_portfolio_monthly`, `kpi_portfolio_summary`,
  `kpi_open_installment_ageing`), `reconciliation`, and the knowledge graph (`kg_nodes`, `kg_edges`).
- **API:** every route validates its input and calls one method of `GoldRepository`, so the
  dashboard, the assistant and Postman all get the same numbers. Rates are summed from numerators
  and denominators, never averaged.
- **Assistant:** a LangGraph agent with four read-only tools (portfolio KPIs, data-quality
  exceptions, metric metadata, connected exposure from the knowledge graph). Every answer carries
  its evidence, filters, source tables and limitations. An input guard refuses actions, SQL,
  credential requests and prompt injection, and answers self-harm questions with help lines.

## Repository

| Folder | What it holds | Docs |
|---|---|---|
| `pipelines/` | Bronze, Silver and Gold SQL for the Lakeflow pipeline | [pipelines/README.md](pipelines/README.md) |
| `api/` | FastAPI service, assistant (`api/assistant/`), web dashboard (`api/static/`) | [docs/08_fastapi.md](docs/08_fastapi.md), [docs/07_langchain.md](docs/07_langchain.md) |
| `tests/` | pytest suites for the API and the assistant | |
| `eval/` | 41 assistant evaluation questions and the runner | [docs/07_langchain.md](docs/07_langchain.md#evaluation) |
| `postman/` | Postman collection, run by Newman in CI | [postman/README.md](postman/README.md) |
| `infra/` | Azure setup scripts, monitoring queries, pre-flight check | [Cloud and DevOps](#cloud-and-devops) |
| `.github/workflows/` | CI/CD pipeline | [CI/CD](#cicd) |
| `docs/` | contracts, API and assistant docs, project decisions | [docs/project/decisions.md](docs/project/decisions.md) |
| `powerbi/` | Power BI report | |
| `scripts/` | data upload and Gold inspection helpers | |

## Run locally

```bash
uv sync
cp .env.example .env          # fill in the values, see docs/08_fastapi.md
uv run --env-file .env uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/app` for the dashboard or `/docs` for Swagger. Locally the API takes
an `X-API-Key` from `API_KEYS`. The full steps, including the Databricks token, are in
[docs/08_fastapi.md](docs/08_fastapi.md#how-to-run-locally).

Before every push:

```bash
uv run ruff check . && uv run pytest -q
```

## CI/CD

`.github/workflows/ci-cd.yml` runs on every pull request and every merge to `main`.

| Step | On a PR | On a merge to `main` |
|---|---|---|
| lint (ruff), test (pytest), build (Docker) | ✓ | ✓ |
| push the image to `acrloant2gk`, tagged with the commit | | ✓ |
| deploy dev + smoke test (`/health` reports the new version) | | ✓ |
| `api-tests`: Newman runs the whole Postman collection against dev | | ✓ |
| approval: a GitHub issue that a team member answers with `approve` | | ✓ |
| deploy prod + smoke test | | ✓ |

- `main` is protected (ruleset PROTECT-MAIN): a pull request with one approving review and green
  lint, test and build checks; no force pushes or deletions.
- Merge to prod takes about 7 minutes plus the approval.
- **Rollback:** re-run the `deploy-prod` job of an earlier run, which redeploys that commit's image
  (rehearsed: 51 seconds), or `az containerapp update ... --image acrloant2gk.azurecr.io/loan-api:<commit>`.

## Cloud and DevOps

All resources are in resource group `rg_gtgh_data_09` (Germany West Central).

| Resource | Name | Purpose |
|---|---|---|
| Container Apps | `ca-loant2gk-dev`, `ca-loant2gk-prod` in `cae-loant2gk` | the API and dashboard; scale 0–1 |
| Container Registry | `acrloant2gk` | one image per commit |
| Key Vault (RBAC) | `kv-loant2gk` | Databricks token, Azure OpenAI key, dev API keys |
| Azure OpenAI | `oai-loant2gk` | `gpt-4.1-mini` for prod, `gpt-4.1-mini-eval` for dev and evaluation |
| Log Analytics | `workspace-rggtghdata09wtuo` | console and platform logs, saved queries |
| Entra ID app | `api-loant2gk` | sign-in on prod, role `Loan.Read` |

- **Secrets** live only in Key Vault. Each app reads them through its own managed identity, and
  team members get access per secret. Nothing secret is in the repo or in plain app settings.
- **Access:** prod accepts only a Microsoft Entra ID token for an assigned team member (the
  dashboard has "Sign in with Microsoft"); dev and local use an API key. Prod never accepts the key.
- **Monitoring:** every request writes one JSON log line with a correlation ID, path, status and
  duration. Six saved queries (`infra/monitoring/`) cover traffic, latency by route, errors,
  rejected calls, tracing one request, and platform warnings.
- **Scripts** (`infra/`):

  | Script | Does |
  |---|---|
  | `azure_setup.sh` | registry, Key Vault, Container Apps environment and the two apps |
  | `team_access.sh` | the team's role assignments |
  | `cicd_setup.sh` | the pipeline's identity: Entra app with GitHub OIDC federation, its role, registry access |
  | `auth_setup.sh` | the Entra app, its scope and role, team assignments, prod auth settings |
  | `openai_setup.sh` | Azure OpenAI deployments, key into Key Vault, app settings |
  | `monitoring_setup.sh` | saves the six queries in Log Analytics |
  | `preflight.sh` | 7 checks before a demo (`--keep-warm` keeps prod and the warehouse awake) |
  | `demo_mode.sh` | `on` / `off`: keep one replica of each app running |

Why each of these choices was made: [docs/project/decisions.md](docs/project/decisions.md).

## Team

| Member | Role | Area |
|---|---|---|
| George Korlos | DevOps & Cloud Engineer | Cloud, DevOps, Project Management |
| Ioannis Panagiotopoulos | Data + AI Engineer | Data generation, Databricks, tools and evaluation |
| Dimitra Motsakou | AI Engineer | LangGraph assistant, Azure OpenAI, assistant page |
| Petros Katsikarelis | Data Analytics & API Developer | FastAPI platform, web dashboard, Microsoft sign-in, PowerBI |
| Blerim Zhonga | Full-stack Developer | FastAPI routes, Databricks connection, prod token check, Postman |
