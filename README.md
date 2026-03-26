# Order-to-Cash Graph Explorer

A full-stack application for visualizing and querying SAP S/4HANA Order-to-Cash data using a graph database, natural language interface, and AI-powered analytics.

---

## demo
![image](https://github.com/amrita40/DodgeAI_home_assessment/blob/main/Screenshot%202026-03-26%20230932.png)


## Architecture

```
Frontend (Single HTML file)         Backend (Python FastAPI)
─────────────────────────           ────────────────────────
Cytoscape.js graph viz    ←────→    /api/graph/*    (NetworkX graph)
Chat interface (Grok)     ←────→    /api/query      (NL → Grok xAI)
Node metadata panels      ←────→    /api/analytics/ (computed stats)
Broken flow detection     ←────→    /api/sql        (SQLite queries)
```

---

## Quick Start (Frontend Only - Recommended)

1. Open `o2c_graph_explorer.html` in any modern browser (Chrome/Firefox/Edge)
2. Get a Grok API key from https://console.x.ai
3. Paste the key in the top-right input
4. Start querying!

**No installation required.** All data is embedded in the HTML file.

---

## Full Stack Setup (Frontend + Python Backend)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up data directory

```bash
mkdir data
# Copy all your JSONL files into ./data/
cp /path/to/your/*.jsonl ./data/
```

### 3. Set your Grok API key

```bash
cp .env.example .env
# Edit .env and add your XAI_API_KEY
```

### 4. Start the backend

```bash
python backend.py
# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 5. Open the frontend

Open `o2c_graph_explorer.html` in your browser. It works standalone or can call the backend API.

---

## Features

### Graph Visualization
- **7 node types**: Sales Orders, Deliveries, Billing Docs, Journal Entries, Customers, Products, Plants
- **Node inspection**: Click any node to see all metadata
- **Node expansion**: Double-click to expand connected nodes
- **3 view modes**: Full graph, Complete chains only, Broken flows only
- **Filters**: By customer, billing status, delivery status

### Natural Language Queries
- Ask questions in plain English about the O2C data
- AI generates data-backed answers with specific document IDs
- Referenced nodes are highlighted on the graph automatically
- **Guardrails**: Off-topic questions are rejected

### Pre-built Analytics
- Broken/incomplete O2C flows
- Revenue by customer
- Top products by billing count
- Cancelled billing documents
- Undelivered sales orders

### Python Backend (REST API)
- `GET /api/graph/full` — Full graph in Cytoscape.js format
- `GET /api/graph/node/{id}` — Node details + neighbors
- `GET /api/graph/trace/{doc_id}` — Trace full flow for any document
- `GET /api/analytics/broken-flows` — All broken O2C flows
- `GET /api/analytics/summary` — Dataset statistics
- `POST /api/query` — Natural language query (returns answer + highlight IDs)
- `GET /api/sql?q=...` — Direct SQL query against SQLite DB
- `GET /api/tables` — List all tables and columns

---

## Data Model

```
Customer ──orders──▶ Sales Order ──fulfilled_by──▶ Delivery ──billed_as──▶ Billing Doc ──posts_to──▶ Journal Entry
                          │                            │                        │
                          └──── items: Products ───────┘                       └─── accountingDocument
```

### Tables (SQLite)
| Table | Description |
|-------|-------------|
| `sales_order_headers` | SO header data (status, amounts, dates) |
| `sales_order_items` | Line items per SO (material, qty, amount) |
| `delivery_headers` | Delivery status and shipping data |
| `delivery_items` | Delivery line items linked to SO items |
| `billing_headers` | Invoice headers (amounts, cancel status) |
| `billing_items` | Invoice line items linked to deliveries |
| `journal_entries` | GL postings linked to billing |
| `ar_line_items` | AR clearing items |
| `business_partners` | Customer master data |
| `bp_addresses` | Customer addresses |
| `product_master` | Product master |
| `plant_master` | Plant master |

### Useful SQL Views
- `v_full_chain` — End-to-end SO→Delivery→Billing→Journal join
- `v_billing_by_material` — Products ranked by billing frequency

---

## Example Queries

### Natural Language (Chat Interface)
- *"Which products are associated with the highest number of billing documents?"*
- *"Trace the full flow of billing document 90504248"*
- *"Which sales orders have broken or incomplete flows?"*
- *"What is the total revenue by customer?"*
- *"Show me all cancelled billing documents and their total value"*

### SQL (Backend API)
```sql
-- Top materials by billing count
SELECT material, productDescription, billingDocCount, totalAmount 
FROM v_billing_by_material LIMIT 10;

-- Full O2C chain
SELECT * FROM v_full_chain WHERE billingDocumentIsCancelled = 'false';

-- Undelivered sales orders
SELECT salesOrder, soldToParty, totalNetAmount, overallDeliveryStatus 
FROM sales_order_headers WHERE overallDeliveryStatus = 'A';
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph Visualization | Cytoscape.js 3.28 |
| AI/NL Interface | xAI Grok (`grok-3-mini` or `grok-3`) |
| Markdown Rendering | Marked.js |
| Backend Framework | FastAPI |
| Graph Library | NetworkX |
| Database | SQLite (in-memory + file) |
| Frontend | Vanilla HTML/CSS/JS (single file) |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your key:

```
XAI_API_KEY=your_xai_api_key_here
```

Get your API key at: https://console.x.ai

---

## File Structure

```
o2c_graph_explorer.html   ← Standalone frontend (all data embedded)
backend.py                ← Python FastAPI backend
requirements.txt          ← Python dependencies
.env.example              ← API key template
README.md                 ← This file
data/                     ← Place JSONL files here (for backend)
```
