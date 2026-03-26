#!/usr/bin/env python3
"""
Order-to-Cash Graph Explorer - Python Backend
FastAPI + NetworkX + SQLite

Run:
    pip install fastapi uvicorn networkx pandas aiofiles python-dotenv google-generativeai
    python backend.py

The backend will:
1. Ingest all JSONL files from ./data/ directory
2. Build a NetworkX graph
3. Serve graph data and NL queries via REST API
4. Use Gemini to translate NL -> structured queries
"""

import json
import os
import glob
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict

import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. AI queries disabled.")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = Path("./data")
DB_PATH = Path("./o2c.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(
    title="O2C Graph Explorer API",
    description="Order-to-Cash graph and NL query backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
G = nx.DiGraph()
DB_CONN: sqlite3.Connection = None
DATA: Dict[str, List] = {}

# ─────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────

SCHEMA_MAP = {
    frozenset(["salesOrder","salesOrderType","soldToParty","totalNetAmount","overallDeliveryStatus"]): "sales_order_headers",
    frozenset(["salesOrder","salesOrderItem","material","requestedQuantity","netAmount","productionPlant"]): "sales_order_items",
    frozenset(["salesOrder","salesOrderItem","scheduleLine","confirmedDeliveryDate"]): "schedule_lines",
    frozenset(["deliveryDocument","shippingPoint","overallGoodsMovementStatus","overallPickingStatus"]): "delivery_headers",
    frozenset(["deliveryDocument","deliveryDocumentItem","plant","referenceSdDocument","actualDeliveryQuantity"]): "delivery_items",
    frozenset(["billingDocument","billingDocumentItem","material","billingQuantity","netAmount","referenceSdDocument"]): "billing_items",
    frozenset(["billingDocument","billingDocumentType","totalNetAmount","soldToParty","accountingDocument"]): "billing_headers",
    frozenset(["accountingDocument","glAccount","referenceDocument","amountInTransactionCurrency","accountingDocumentType"]): "journal_entries",
    frozenset(["accountingDocument","invoiceReference","salesDocument","customer","amountInTransactionCurrency"]): "ar_line_items",
    frozenset(["businessPartner","customer","businessPartnerFullName","businessPartnerIsBlocked"]): "business_partners",
    frozenset(["businessPartner","cityName","region","streetName","postalCode"]): "bp_addresses",
    frozenset(["customer","companyCode","paymentTerms","reconciliationAccount"]): "customer_company",
    frozenset(["customer","salesOrganization","distributionChannel","currency"]): "customer_sales",
    frozenset(["product","language","productDescription"]): "product_descriptions",
    frozenset(["product","productType","grossWeight","netWeight","productGroup"]): "product_master",
    frozenset(["product","plant","countryOfOrigin","mrpType","profitCenter"]): "product_plant",
    frozenset(["product","plant","storageLocation","physicalInventoryBlockInd"]): "product_storage",
    frozenset(["plant","plantName","valuationArea","salesOrganization"]): "plant_master",
}

def classify_file(filepath: str) -> Optional[str]:
    """Classify a JSONL file by its schema."""
    try:
        with open(filepath) as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            row = json.loads(first_line)
            keys = frozenset(row.keys())
            # Try exact match first
            if keys in SCHEMA_MAP:
                return SCHEMA_MAP[keys]
            # Try subset match
            for schema_keys, table in SCHEMA_MAP.items():
                if schema_keys.issubset(keys) or keys.issubset(schema_keys):
                    if len(schema_keys.intersection(keys)) >= 3:
                        return table
    except Exception as e:
        print(f"  Error classifying {filepath}: {e}")
    return None

def ingest_data(data_dir: str = "./data") -> Dict[str, List]:
    """Ingest all JSONL files from data directory."""
    print(f"\n📂 Ingesting data from {data_dir}...")
    tables = defaultdict(list)
    files = glob.glob(f"{data_dir}/*.jsonl")
    
    if not files:
        print(f"  No JSONL files found in {data_dir}")
        print(f"  Place your JSONL files in the ./data/ directory")
        return {}

    for filepath in sorted(files):
        table = classify_file(filepath)
        if table:
            count = 0
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            tables[table].append(json.loads(line))
                            count += 1
                        except:
                            pass
            print(f"  ✅ {Path(filepath).name} → {table} ({count} rows)")
        else:
            print(f"  ⚠  {Path(filepath).name} → unclassified, skipping")

    print(f"\n📊 Loaded tables:")
    for t, rows in tables.items():
        print(f"  {t}: {len(rows)} rows")
    
    return dict(tables)


def build_sqlite_db(data: Dict[str, List]) -> sqlite3.Connection:
    """Load all data into SQLite for SQL queries."""
    print("\n🗄  Building SQLite database...")
    
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    for table_name, rows in data.items():
        if not rows:
            continue
        # Get columns from first row
        cols = list(rows[0].keys())
        # Sanitize column names
        safe_cols = [c.replace("-","_") for c in cols]
        
        col_defs = ", ".join(f'"{c}" TEXT' for c in safe_cols)
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')
        
        placeholders = ", ".join("?" for _ in safe_cols)
        for row in rows:
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, (dict, list)):
                    v = json.dumps(v)
                elif v is None:
                    v = ""
                vals.append(str(v))
            try:
                conn.execute(
                    f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                    vals
                )
            except:
                pass
        
        conn.commit()
        print(f"  ✅ {table_name}: {len(rows)} rows")

    # Create useful views
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_full_chain AS
        SELECT 
            soh.salesOrder,
            soh.soldToParty,
            soh.totalNetAmount as soAmount,
            soh.overallDeliveryStatus,
            di.deliveryDocument,
            bh.billingDocument,
            bh.billingDocumentIsCancelled,
            bh.totalNetAmount as billingAmount,
            bh.accountingDocument,
            je.glAccount,
            je.amountInTransactionCurrency as journalAmount,
            je.clearingDate
        FROM sales_order_headers soh
        LEFT JOIN delivery_items di ON di.referenceSdDocument = soh.salesOrder
        LEFT JOIN billing_items bi ON bi.referenceSdDocument = di.deliveryDocument
        LEFT JOIN billing_headers bh ON bh.billingDocument = bi.billingDocument
        LEFT JOIN journal_entries je ON je.accountingDocument = bh.accountingDocument
    """)
    
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_billing_by_material AS
        SELECT 
            bi.material,
            pd.productDescription,
            COUNT(DISTINCT bi.billingDocument) as billingDocCount,
            SUM(CAST(bi.netAmount AS REAL)) as totalAmount
        FROM billing_items bi
        LEFT JOIN product_descriptions pd ON pd.product = bi.material AND pd.language = 'EN'
        GROUP BY bi.material
        ORDER BY billingDocCount DESC
    """)

    conn.commit()
    print("  ✅ Views created")
    return conn


def build_graph(data: Dict[str, List]) -> nx.DiGraph:
    """Build NetworkX graph from O2C data."""
    print("\n🔗 Building graph...")
    G = nx.DiGraph()

    # Product descriptions lookup
    prod_desc = {}
    for p in data.get("product_descriptions", []):
        if p.get("language") == "EN":
            prod_desc[p["product"]] = p["productDescription"]

    # Customer names lookup
    cust_names = {}
    for bp in data.get("business_partners", []):
        cust_names[bp["businessPartner"]] = bp.get("businessPartnerFullName") or bp.get("businessPartnerName", "")

    # Add customer nodes
    for bp in data.get("business_partners", []):
        G.add_node(
            f"CUST_{bp['businessPartner']}",
            type="Customer",
            label=bp.get("businessPartnerFullName", bp["businessPartner"]),
            data=bp
        )

    # Add sales order nodes
    so_items = defaultdict(list)
    for item in data.get("sales_order_items", []):
        so_items[item["salesOrder"]].append(item)

    for so in data.get("sales_order_headers", []):
        node_id = f"SO_{so['salesOrder']}"
        G.add_node(node_id, type="SalesOrder", label=so["salesOrder"], data={
            **so, "items": so_items.get(so["salesOrder"], [])
        })
        # Edge: Customer -> SO
        cust_node = f"CUST_{so['soldToParty']}"
        if G.has_node(cust_node):
            G.add_edge(cust_node, node_id, relation="PLACES_ORDER")

    # Delivery items: build delivery->SO mapping
    del_items = defaultdict(list)
    del_so_map = defaultdict(set)
    for di in data.get("delivery_items", []):
        del_items[di["deliveryDocument"]].append(di)
        if di.get("referenceSdDocument"):
            del_so_map[di["deliveryDocument"]].add(di["referenceSdDocument"])

    # Add delivery nodes
    for dh in data.get("delivery_headers", []):
        node_id = f"DEL_{dh['deliveryDocument']}"
        so_refs = list(del_so_map.get(dh["deliveryDocument"], []))
        plants = list(set(i.get("plant","") for i in del_items.get(dh["deliveryDocument"], []) if i.get("plant")))
        G.add_node(node_id, type="Delivery", label=dh["deliveryDocument"], data={
            **dh, "salesOrderRefs": so_refs, "plants": plants,
            "items": del_items.get(dh["deliveryDocument"], [])
        })
        for so_ref in so_refs:
            so_node = f"SO_{so_ref}"
            if G.has_node(so_node):
                G.add_edge(so_node, node_id, relation="FULFILLED_BY")

    # Billing items: build billing->delivery mapping
    bill_items = defaultdict(list)
    bill_del_map = defaultdict(set)
    for bi in data.get("billing_items", []):
        bill_items[bi["billingDocument"]].append({
            **bi, "productDescription": prod_desc.get(bi.get("material",""), "")
        })
        if bi.get("referenceSdDocument"):
            bill_del_map[bi["billingDocument"]].add(bi["referenceSdDocument"])

    # Journal entries by accounting doc
    je_map = defaultdict(list)
    for je in data.get("journal_entries", []):
        je_map[je["accountingDocument"]].append(je)

    # Add billing nodes
    for bh in data.get("billing_headers", []):
        node_id = f"BILL_{bh['billingDocument']}"
        del_refs = list(bill_del_map.get(bh["billingDocument"], []))
        acct = bh.get("accountingDocument", "")
        journals = je_map.get(acct, [])
        G.add_node(node_id, type="BillingDoc", label=bh["billingDocument"], data={
            **bh, "deliveryRefs": del_refs, "journalEntries": journals,
            "items": bill_items.get(bh["billingDocument"], [])
        })
        for del_ref in del_refs:
            del_node = f"DEL_{del_ref}"
            if G.has_node(del_node):
                G.add_edge(del_node, node_id, relation="BILLED_AS")

    # Add journal entry nodes
    for acct_doc, entries in je_map.items():
        for je in entries[:1]:  # One node per accounting doc
            node_id = f"JE_{acct_doc}"
            if not G.has_node(node_id):
                G.add_node(node_id, type="JournalEntry", label=acct_doc, data=je)
            # Link from billing
            bill_node = next(
                (f"BILL_{bh['billingDocument']}" for bh in data.get("billing_headers",[])
                 if bh.get("accountingDocument") == acct_doc), None
            )
            if bill_node and G.has_node(bill_node):
                G.add_edge(bill_node, node_id, relation="POSTS_TO")

    # Add plant nodes
    for pl in data.get("plant_master", []):
        node_id = f"PLANT_{pl['plant']}"
        G.add_node(node_id, type="Plant", label=pl.get("plantName", pl["plant"]), data=pl)

    # Add product nodes
    seen_products = set()
    for pm in data.get("product_master", []):
        pid = pm["product"]
        if pid not in seen_products:
            seen_products.add(pid)
            node_id = f"PROD_{pid}"
            G.add_node(node_id, type="Product", label=prod_desc.get(pid, pid), data={
                **pm, "description": prod_desc.get(pid, "")
            })

    print(f"  ✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Node types: {dict(defaultdict(int, [(G.nodes[n].get('type','?'), 1) for n in G.nodes()]))}")
    return G


# ─────────────────────────────────────────────
# API MODELS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    api_key: Optional[str] = None

class GraphFilter(BaseModel):
    node_types: Optional[List[str]] = None
    customer_id: Optional[str] = None
    billing_status: Optional[str] = None  # "active" | "cancelled" | None


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "tables": {k: len(v) for k, v in DATA.items()}
    }


@app.get("/api/graph/full")
def get_full_graph(
    node_types: Optional[str] = Query(None, description="Comma-separated node types"),
    customer_id: Optional[str] = Query(None),
    billing_status: Optional[str] = Query(None),
    limit: int = Query(500, le=2000)
):
    """Return graph data in Cytoscape.js format."""
    type_filter = set(node_types.split(",")) if node_types else None

    nodes = []
    edges = []
    included = set()

    for node_id, attrs in list(G.nodes(data=True))[:limit]:
        ntype = attrs.get("type", "")
        if type_filter and ntype not in type_filter:
            continue
        
        node_data = attrs.get("data", {})
        
        # Apply filters
        if customer_id:
            if ntype == "SalesOrder" and node_data.get("soldToParty") != customer_id:
                continue
            if ntype == "Customer" and node_id != f"CUST_{customer_id}":
                continue
        
        if billing_status and ntype == "BillingDoc":
            is_cancelled = str(node_data.get("billingDocumentIsCancelled","")).lower() in ("true","1")
            if billing_status == "active" and is_cancelled:
                continue
            if billing_status == "cancelled" and not is_cancelled:
                continue

        # Slim down data for wire transfer
        slim_data = {k: v for k, v in node_data.items()
                     if k not in ("items","journalEntries","salesOrderRefs","deliveryRefs","plants","raw")
                     and not isinstance(v, (list, dict))}
        
        nodes.append({
            "data": {
                "id": node_id,
                "type": ntype,
                "label": attrs.get("label", node_id),
                **slim_data
            }
        })
        included.add(node_id)

    for src, tgt, edata in G.edges(data=True):
        if src in included and tgt in included:
            edges.append({
                "data": {
                    "id": f"{src}->{tgt}",
                    "source": src,
                    "target": tgt,
                    "relation": edata.get("relation", "")
                }
            })

    return {"nodes": nodes, "edges": edges, "total_nodes": len(nodes), "total_edges": len(edges)}


@app.get("/api/graph/node/{node_id}")
def get_node_detail(node_id: str):
    """Get full details for a specific node including all connected data."""
    if not G.has_node(node_id):
        raise HTTPException(404, f"Node {node_id} not found")

    attrs = G.nodes[node_id]
    predecessors = [{"id": p, "type": G.nodes[p].get("type",""), "relation": G[p][node_id].get("relation","")}
                    for p in G.predecessors(node_id)]
    successors = [{"id": s, "type": G.nodes[s].get("type",""), "relation": G[node_id][s].get("relation","")}
                  for s in G.successors(node_id)]

    return {
        "id": node_id,
        "type": attrs.get("type"),
        "label": attrs.get("label"),
        "data": attrs.get("data", {}),
        "predecessors": predecessors,
        "successors": successors,
        "degree": G.degree(node_id)
    }


@app.get("/api/graph/trace/{doc_id}")
def trace_document(doc_id: str):
    """Trace full O2C flow for any document ID."""
    # Find starting node
    start_node = None
    for prefix in ["BILL_", "DEL_", "SO_", "JE_"]:
        candidate = f"{prefix}{doc_id}"
        if G.has_node(candidate):
            start_node = candidate
            break

    if not start_node:
        raise HTTPException(404, f"Document {doc_id} not found in graph")

    # BFS upstream and downstream
    chain_nodes = {start_node}

    def walk(node, direction="both", depth=0):
        if depth > 5:
            return
        if direction in ("both", "up"):
            for pred in G.predecessors(node):
                if pred not in chain_nodes:
                    chain_nodes.add(pred)
                    walk(pred, "up", depth+1)
        if direction in ("both", "down"):
            for succ in G.successors(node):
                if succ not in chain_nodes:
                    chain_nodes.add(succ)
                    walk(succ, "down", depth+1)

    walk(start_node)

    chain_data = []
    for nid in chain_nodes:
        attrs = G.nodes[nid]
        chain_data.append({"id": nid, "type": attrs.get("type"), "label": attrs.get("label"), "data": attrs.get("data",{})})

    edges = []
    for src, tgt, edata in G.edges(data=True):
        if src in chain_nodes and tgt in chain_nodes:
            edges.append({"source": src, "target": tgt, "relation": edata.get("relation","")})

    return {"chain": chain_data, "edges": edges, "doc_id": doc_id, "start_node": start_node}


@app.get("/api/analytics/broken-flows")
def get_broken_flows():
    """Find all O2C flows with gaps."""
    broken = []

    billed_deliveries = set()
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "BillingDoc":
            data = attrs.get("data", {})
            if not str(data.get("billingDocumentIsCancelled","")).lower() in ("true","1"):
                for del_ref in data.get("deliveryRefs", []):
                    billed_deliveries.add(del_ref)

    delivered_sos = set()
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Delivery":
            for so_ref in attrs.get("data", {}).get("salesOrderRefs", []):
                delivered_sos.add(so_ref)

    # Deliveries without billing
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "Delivery":
            del_id = node_id.replace("DEL_", "")
            if del_id not in billed_deliveries:
                broken.append({"type": "DELIVERED_NOT_BILLED", "nodeId": node_id,
                               "data": attrs.get("data", {})})

    # SOs without delivery
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "SalesOrder":
            data = attrs.get("data", {})
            so_id = data.get("salesOrder", "")
            if so_id not in delivered_sos:
                broken.append({"type": "ORDER_NOT_DELIVERED", "nodeId": node_id,
                               "data": data})

    return {"broken_flows": broken, "count": len(broken)}


@app.get("/api/analytics/summary")
def get_summary():
    """Get dataset summary statistics."""
    type_counts = defaultdict(int)
    for _, attrs in G.nodes(data=True):
        type_counts[attrs.get("type", "Unknown")] += 1

    billing_nodes = [(n, G.nodes[n]) for n in G.nodes() if G.nodes[n].get("type") == "BillingDoc"]
    active_billing = sum(1 for _, a in billing_nodes
                         if not str(a["data"].get("billingDocumentIsCancelled","")).lower() in ("true","1"))
    total_revenue = sum(
        float(a["data"].get("totalNetAmount", 0) or 0)
        for _, a in billing_nodes
        if not str(a["data"].get("billingDocumentIsCancelled","")).lower() in ("true","1")
    )

    # Material billing counts
    mat_counts = defaultdict(int)
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "BillingDoc":
            for item in attrs["data"].get("items", []):
                mat_counts[item.get("material", "")] += 1

    top_materials = sorted(mat_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "node_counts": dict(type_counts),
        "edge_count": G.number_of_edges(),
        "active_billing_docs": active_billing,
        "cancelled_billing_docs": len(billing_nodes) - active_billing,
        "total_active_revenue_inr": round(total_revenue, 2),
        "top_materials_by_billing": [{"material": m, "count": c} for m, c in top_materials],
    }


@app.post("/api/query")
async def natural_language_query(req: QueryRequest):
    """Answer natural language questions using Gemini."""
    question = req.question.strip()
    api_key = req.api_key or GEMINI_API_KEY

    # Guardrails: check if question is related to O2C domain
    off_topic_keywords = [
        "weather", "recipe", "poem", "story", "joke", "code", "programming",
        "python help", "javascript", "history of", "who invented", "capital of",
        "translate", "write me a", "explain quantum", "stock price", "sports",
        "movie", "music", "celebrity"
    ]
    q_lower = question.lower()
    if any(kw in q_lower for kw in off_topic_keywords):
        return {
            "answer": "This system is designed to answer questions related to the provided Order-to-Cash dataset only. Please ask about sales orders, deliveries, billing documents, customers, or products.",
            "highlight_ids": [],
            "data_used": []
        }

    if not api_key:
        return {"answer": "Please provide a Gemini API key.", "highlight_ids": [], "data_used": []}

    if not GEMINI_AVAILABLE:
        return {"answer": "google-generativeai package not installed. Run: pip install google-generativeai", "highlight_ids": [], "data_used": []}

    # Build data context
    context = build_query_context(question)

    system_prompt = """You are a data analyst for a SAP S/4HANA Order-to-Cash dataset.
ONLY answer questions about: sales orders, deliveries, billing documents, journal entries, customers, products, plants.
If asked anything else, say: "This system is designed to answer questions related to the provided Order-to-Cash dataset only."

When mentioning document IDs, include them as a JSON list at the end:
HIGHLIGHT_IDS:{"ids":["SO_740506","DEL_80738072","BILL_90504248"]}

Always provide specific, data-backed answers with actual IDs and numbers."""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"{system_prompt}\n\nDATA CONTEXT:\n{context}\n\nQUESTION: {question}",
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=1500)
        )
        text = response.text

        # Extract highlight IDs
        import re
        hl_match = re.search(r'HIGHLIGHT_IDS:\{"ids":\[([^\]]*)\]\}', text)
        highlight_ids = []
        if hl_match:
            try:
                highlight_ids = json.loads(f'[{hl_match.group(1)}]')
                text = text[:hl_match.start()].strip()
            except:
                pass

        return {"answer": text, "highlight_ids": highlight_ids, "data_used": list(context[:200])}

    except Exception as e:
        raise HTTPException(500, f"Gemini error: {str(e)}")


@app.get("/api/sql")
def run_sql(q: str = Query(..., description="SQL query to run against the O2C database")):
    """Execute a SQL query against the O2C SQLite database."""
    # Basic SQL injection guard
    forbidden = ["drop", "delete", "insert", "update", "create", "alter", "attach"]
    if any(kw in q.lower() for kw in forbidden):
        raise HTTPException(400, "Only SELECT queries are allowed")

    try:
        cursor = DB_CONN.execute(q)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchmany(500)]
        return {"columns": cols, "rows": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(400, f"SQL error: {str(e)}")


@app.get("/api/tables")
def list_tables():
    """List all available SQL tables and their schemas."""
    cursor = DB_CONN.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
    tables = {}
    for row in cursor.fetchall():
        name = row[0]
        cols = DB_CONN.execute(f'PRAGMA table_info("{name}")').fetchall()
        tables[name] = [c[1] for c in cols]
    return tables


def build_query_context(question: str) -> str:
    """Build relevant data context for LLM query."""
    ctx = ""
    q = question.lower()

    # Summary stats
    summary = get_summary()
    ctx += f"SUMMARY: {json.dumps(summary)}\n\n"

    # Customer data
    customers = [{"id": G.nodes[n]["data"].get("businessPartner"),
                  "name": G.nodes[n]["data"].get("businessPartnerFullName") or G.nodes[n]["data"].get("businessPartnerName")}
                 for n in G.nodes() if G.nodes[n].get("type") == "Customer"]
    ctx += f"CUSTOMERS: {json.dumps(customers)}\n\n"

    # Document-specific queries
    import re
    doc_ids = re.findall(r'\b\d{6,10}\b', question)
    for doc_id in doc_ids:
        try:
            trace = trace_document(doc_id)
            ctx += f"TRACE FOR {doc_id}: {json.dumps(trace, default=str)[:3000]}\n\n"
        except:
            pass

    # Broken flows
    if any(kw in q for kw in ["broken", "incomplete", "missing", "unbilled", "undelivered"]):
        broken = get_broken_flows()
        ctx += f"BROKEN FLOWS: {json.dumps(broken, default=str)[:2000]}\n\n"

    # Revenue queries
    if any(kw in q for kw in ["revenue", "amount", "total", "highest", "top"]):
        cust_rev = {}
        for n in G.nodes():
            if G.nodes[n].get("type") == "BillingDoc":
                data = G.nodes[n]["data"]
                if not str(data.get("billingDocumentIsCancelled","")).lower() in ("true","1"):
                    cid = data.get("soldToParty","")
                    cust_rev[cid] = cust_rev.get(cid, 0) + float(data.get("totalNetAmount", 0) or 0)
        ctx += f"REVENUE BY CUSTOMER: {json.dumps(cust_rev)}\n\n"

    return ctx


# Serve frontend
if Path("./frontend/build").exists():
    app.mount("/", StaticFiles(directory="./frontend/build", html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "O2C Graph API running. Open o2c_graph_explorer.html for the UI. API docs at /docs"}


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global G, DB_CONN, DATA
    print("\n🚀 O2C Graph Explorer Backend starting...")
    DATA = ingest_data("./data")
    if DATA:
        DB_CONN = build_sqlite_db(DATA)
        G = build_graph(DATA)
    else:
        print("\n⚠  No data found. Create ./data/ directory and place JSONL files there.")
        DB_CONN = sqlite3.connect(":memory:")
    print("\n✅ Ready! API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
