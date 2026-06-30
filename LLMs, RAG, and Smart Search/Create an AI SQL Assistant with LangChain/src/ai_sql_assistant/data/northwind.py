"""Northwind dataset acquisition, build, and scaling utilities."""

from __future__ import annotations

import hashlib
import io
import random
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ai_sql_assistant.constants import NORTHWIND_DB_PATH, NORTHWIND_SCALED_DB_PATH, RAW_DATA_DIR
from ai_sql_assistant.logging_utils import logger

NORTHWIND_SOURCES = [
    "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db",
    "https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db",
]


@dataclass(slots=True)
class BuildResult:
    """Result of data build operation."""

    raw_db_path: Path
    scaled_db_path: Path
    source: str
    raw_orders: int
    scaled_orders: int


def _init_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT NOT NULL,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id INTEGER PRIMARY KEY,
            company_name TEXT NOT NULL,
            contact_name TEXT,
            city TEXT,
            country TEXT
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            supplier_id INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            quantity_per_unit TEXT,
            unit_price REAL NOT NULL,
            units_in_stock INTEGER NOT NULL,
            discontinued INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(supplier_id) REFERENCES suppliers(supplier_id),
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        );

        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            contact_name TEXT,
            city TEXT,
            region TEXT,
            country TEXT,
            segment TEXT
        );

        CREATE TABLE IF NOT EXISTS employees (
            employee_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            title TEXT,
            hire_date TEXT,
            city TEXT,
            country TEXT
        );

        CREATE TABLE IF NOT EXISTS shippers (
            shipper_id INTEGER PRIMARY KEY,
            company_name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id TEXT NOT NULL,
            employee_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            required_date TEXT,
            shipped_date TEXT,
            ship_via INTEGER,
            freight REAL,
            ship_city TEXT,
            ship_country TEXT,
            market TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id),
            FOREIGN KEY(ship_via) REFERENCES shippers(shipper_id)
        );

        CREATE TABLE IF NOT EXISTS order_details (
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            discount REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY(order_id) REFERENCES orders(order_id),
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        );

        CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
        CREATE INDEX IF NOT EXISTS idx_orders_employee ON orders(employee_id);
        CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
        CREATE INDEX IF NOT EXISTS idx_orders_market ON orders(market);
        CREATE INDEX IF NOT EXISTS idx_order_details_product ON order_details(product_id);
        CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
        """
    )


def _build_synthetic_raw_northwind(db_path: Path, seed: int = 42) -> None:
    _init_seed(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    _create_schema(conn)

    countries = ["USA", "Germany", "UK", "France", "India", "Canada", "Brazil", "Japan", "Australia"]
    cities = {
        "USA": ["New York", "Boston", "Seattle", "Chicago", "Austin"],
        "Germany": ["Berlin", "Munich", "Hamburg"],
        "UK": ["London", "Manchester", "Bristol"],
        "France": ["Paris", "Lyon", "Nantes"],
        "India": ["Bengaluru", "Mumbai", "Delhi"],
        "Canada": ["Toronto", "Vancouver", "Montreal"],
        "Brazil": ["Sao Paulo", "Rio"],
        "Japan": ["Tokyo", "Osaka"],
        "Australia": ["Sydney", "Melbourne"],
    }
    regions = ["North", "South", "East", "West", "Central"]
    segments = ["Enterprise", "SMB", "Mid-Market", "Public Sector"]
    markets = ["North America", "Europe", "APAC", "LATAM"]

    categories = [
        (1, "Beverages", "Soft drinks, coffees, teas"),
        (2, "Condiments", "Sweet and savory sauces"),
        (3, "Confections", "Desserts and candies"),
        (4, "Dairy Products", "Cheeses"),
        (5, "Grains/Cereals", "Breads and crackers"),
        (6, "Meat/Poultry", "Prepared meats"),
        (7, "Produce", "Dried fruit and bean curd"),
        (8, "Seafood", "Seaweed and fish"),
    ]
    suppliers = [
        (idx + 1, f"Supplier {idx + 1}", f"Contact {idx + 1}", random.choice(cities[c]), c)
        for idx, c in enumerate(np.random.choice(countries, size=40, replace=True))
    ]

    products: list[tuple[int, str, int, int, str, float, int, int]] = []
    for product_id in range(1, 301):
        category_id = random.randint(1, 8)
        supplier_id = random.randint(1, 40)
        base_price = round(np.random.uniform(3, 300), 2)
        products.append(
            (
                product_id,
                f"Product {product_id}",
                supplier_id,
                category_id,
                f"{random.randint(1, 24)} units",
                base_price,
                random.randint(0, 600),
                1 if random.random() < 0.08 else 0,
            )
        )

    customers: list[tuple[str, str, str, str, str, str, str]] = []
    for idx in range(1, 801):
        country = random.choice(countries)
        customer_id = f"C{idx:05d}"
        customers.append(
            (
                customer_id,
                f"Company {idx}",
                f"Contact {idx}",
                random.choice(cities[country]),
                random.choice(regions),
                country,
                random.choice(segments),
            )
        )

    employees = [
        (
            idx,
            f"First{idx}",
            f"Last{idx}",
            random.choice(["Sales Rep", "Regional Manager", "Director", "Analyst"]),
            str(date(2010, 1, 1) + timedelta(days=random.randint(0, 3650))),
            random.choice(sum(cities.values(), [])),
            random.choice(countries),
        )
        for idx in range(1, 31)
    ]
    shippers = [(1, "Speedy Express"), (2, "United Package"), (3, "Federal Shipping")]

    conn.executemany("INSERT INTO categories VALUES (?, ?, ?)", categories)
    conn.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?)", suppliers)
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        products,
    )
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", customers)
    conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
    conn.executemany("INSERT INTO shippers VALUES (?, ?)", shippers)

    start_date = date(2020, 1, 1)
    order_rows: list[tuple[int, str, int, str, str, str, int, float, str, str, str]] = []
    detail_rows: list[tuple[int, int, float, int, float]] = []
    order_id = 1

    for _ in range(12_000):
        customer = random.choice(customers)
        employee_id = random.randint(1, 30)
        order_day = start_date + timedelta(days=random.randint(0, 1825))
        shipped_day = order_day + timedelta(days=random.randint(1, 8))
        country = customer[5]
        city = customer[3]
        market = (
            "Europe"
            if country in {"Germany", "UK", "France"}
            else "North America"
            if country in {"USA", "Canada"}
            else "APAC"
            if country in {"India", "Japan", "Australia"}
            else "LATAM"
        )
        order_rows.append(
            (
                order_id,
                customer[0],
                employee_id,
                str(order_day),
                str(order_day + timedelta(days=14)),
                str(shipped_day),
                random.randint(1, 3),
                round(np.random.uniform(5, 200), 2),
                city,
                country,
                market,
            )
        )

        item_count = random.randint(1, 7)
        picked = np.random.choice(np.arange(1, 301), size=item_count, replace=False)
        for product_id in picked:
            price = products[product_id - 1][5]
            detail_rows.append(
                (
                    order_id,
                    int(product_id),
                    round(price * np.random.uniform(0.9, 1.1), 2),
                    random.randint(1, 25),
                    round(float(np.random.choice([0.0, 0.05, 0.1, 0.15])), 2),
                )
            )
        order_id += 1

    conn.executemany(
        """
        INSERT INTO orders (
            order_id, customer_id, employee_id, order_date, required_date,
            shipped_date, ship_via, freight, ship_city, ship_country, market
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        order_rows,
    )
    conn.executemany(
        "INSERT INTO order_details(order_id, product_id, unit_price, quantity, discount) VALUES (?, ?, ?, ?, ?)",
        detail_rows,
    )

    conn.commit()
    conn.close()


def _try_download_sqlite(destination: Path) -> bool:
    """Try downloading public Northwind SQLite dump.

    Returns:
        bool: True if download succeeded.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source in NORTHWIND_SOURCES:
        try:
            request = Request(source, headers={"User-Agent": "ai-sql-assistant/0.1"})
            with urlopen(request, timeout=25) as response:
                payload = response.read()

            # Direct SQLite file path.
            if source.lower().endswith((".sqlite", ".db")) or payload.startswith(b"SQLite format 3"):
                destination.write_bytes(payload)
                logger.info("Downloaded Northwind from {}", source)
                return True

            archive = zipfile.ZipFile(io.BytesIO(payload))
            names = [name for name in archive.namelist() if name.lower().endswith((".sqlite", ".db"))]
            if not names:
                continue
            member = names[0]
            with archive.open(member) as src, destination.open("wb") as dst:
                dst.write(src.read())
            logger.info("Downloaded Northwind from {}", source)
            return True
        except Exception as exc:
            logger.warning("Northwind download failed for {}: {}", source, exc)
    return False


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _supports_internal_schema(db_path: Path) -> bool:
    """Check whether DB matches expected normalized schema for scaling pipeline."""
    try:
        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        required_tables = {"orders", "order_details", "customers", "products", "categories", "suppliers"}
        if not required_tables.issubset({name.lower() for name in tables}):
            conn.close()
            return False

        order_cols = {
            row[1].lower() for row in conn.execute("PRAGMA table_info('orders')").fetchall()
        }
        detail_cols = {
            row[1].lower() for row in conn.execute("PRAGMA table_info('order_details')").fetchall()
        }
        conn.close()
        return {"order_id", "order_date", "customer_id"}.issubset(order_cols) and {
            "order_id",
            "product_id",
            "unit_price",
            "quantity",
            "discount",
        }.issubset(detail_cols)
    except Exception:
        return False


def _clone_dimensions(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    dims = ["categories", "suppliers", "products", "customers", "employees", "shippers"]
    for table in dims:
        frame = pd.read_sql_query(f"SELECT * FROM {table}", src)
        frame.to_sql(table, dst, index=False, if_exists="append")


def _build_scaled_copy(raw_db_path: Path, scaled_db_path: Path, scale_factor: int = 8, seed: int = 42) -> None:
    _init_seed(seed)
    if scaled_db_path.exists():
        scaled_db_path.unlink()

    src = sqlite3.connect(raw_db_path)
    dst = sqlite3.connect(scaled_db_path)
    _create_schema(dst)

    _clone_dimensions(src, dst)

    orders = pd.read_sql_query("SELECT * FROM orders", src)
    details = pd.read_sql_query("SELECT * FROM order_details", src)
    max_order_id = int(orders["order_id"].max())

    scaled_orders: list[pd.DataFrame] = []
    scaled_details: list[pd.DataFrame] = []

    for replica in range(scale_factor):
        shift_days = replica * 90
        order_block = orders.copy()
        detail_block = details.copy()

        order_block["order_id"] = order_block["order_id"] + replica * max_order_id
        order_block["freight"] = (order_block["freight"] * np.random.uniform(0.95, 1.08, len(order_block))).round(2)
        order_block["order_date"] = pd.to_datetime(order_block["order_date"]) + pd.to_timedelta(shift_days, unit="D")
        order_block["required_date"] = pd.to_datetime(order_block["required_date"]) + pd.to_timedelta(shift_days, unit="D")
        order_block["shipped_date"] = pd.to_datetime(order_block["shipped_date"]) + pd.to_timedelta(shift_days, unit="D")
        order_block["order_date"] = order_block["order_date"].dt.date.astype(str)
        order_block["required_date"] = order_block["required_date"].dt.date.astype(str)
        order_block["shipped_date"] = order_block["shipped_date"].dt.date.astype(str)

        detail_block["order_id"] = detail_block["order_id"] + replica * max_order_id
        detail_block["quantity"] = np.maximum(1, (detail_block["quantity"] * np.random.uniform(0.95, 1.2, len(detail_block))).round()).astype(int)
        detail_block["unit_price"] = (detail_block["unit_price"] * np.random.uniform(0.96, 1.08, len(detail_block))).round(2)

        scaled_orders.append(order_block)
        scaled_details.append(detail_block)

    pd.concat(scaled_orders, ignore_index=True).to_sql("orders", dst, index=False, if_exists="append")
    pd.concat(scaled_details, ignore_index=True).to_sql("order_details", dst, index=False, if_exists="append")

    dst.commit()
    src.close()
    dst.close()


def sqlite_md5(db_path: Path) -> str:
    """Compute file checksum for reproducibility reports."""
    digest = hashlib.md5()  # nosec B324 - checksum, not cryptographic security
    with db_path.open("rb") as handle:
        while chunk := handle.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def build_northwind_databases(
    raw_db_path: Path = NORTHWIND_DB_PATH,
    scaled_db_path: Path = NORTHWIND_SCALED_DB_PATH,
    scale_factor: int = 8,
    seed: int = 42,
) -> BuildResult:
    """Build raw and scaled Northwind SQLite datasets.

    Download attempted first. Deterministic synthetic fallback used when network unavailable.
    """
    raw_db_path.parent.mkdir(parents=True, exist_ok=True)

    source = "downloaded_northwind"
    downloaded = _try_download_sqlite(raw_db_path)
    if not downloaded:
        source = "synthetic_northwind"
        _build_synthetic_raw_northwind(raw_db_path, seed=seed)
    elif not _supports_internal_schema(raw_db_path):
        source = "downloaded_incompatible_schema_fallback_synthetic"
        _build_synthetic_raw_northwind(raw_db_path, seed=seed)

    _build_scaled_copy(raw_db_path, scaled_db_path, scale_factor=scale_factor, seed=seed)

    raw_conn = sqlite3.connect(raw_db_path)
    scaled_conn = sqlite3.connect(scaled_db_path)
    raw_orders = _table_count(raw_conn, "orders")
    scaled_orders = _table_count(scaled_conn, "orders")
    raw_conn.close()
    scaled_conn.close()

    return BuildResult(
        raw_db_path=raw_db_path,
        scaled_db_path=scaled_db_path,
        source=source,
        raw_orders=raw_orders,
        scaled_orders=scaled_orders,
    )
