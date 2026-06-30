# Schema Report: `/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/LLMs, RAG, and Smart Search/Create an AI SQL Assistant with LangChain/data/sqlite/northwind_scaled.db`

## Relationships
- `order_details.product_id` -> `products.product_id`
- `order_details.order_id` -> `orders.order_id`
- `orders.ship_via` -> `shippers.shipper_id`
- `orders.employee_id` -> `employees.employee_id`
- `orders.customer_id` -> `customers.customer_id`
- `products.category_id` -> `categories.category_id`
- `products.supplier_id` -> `suppliers.supplier_id`

## Tables

### `categories`
- Rows: **8**
- Columns:
  - `category_id` `INTEGER` PK
  - `category_name` `TEXT` NOT NULL
  - `description` `TEXT`
- Indexes:
  - none
- Null Stats:
  - `category_id`: 0
  - `category_name`: 0
  - `description`: 0

### `customers`
- Rows: **800**
- Columns:
  - `customer_id` `TEXT` PK
  - `company_name` `TEXT` NOT NULL
  - `contact_name` `TEXT`
  - `city` `TEXT`
  - `region` `TEXT`
  - `country` `TEXT`
  - `segment` `TEXT`
- Indexes:
  - `sqlite_autoindex_customers_1` (customer_id)
- Null Stats:
  - `customer_id`: 0
  - `company_name`: 0
  - `contact_name`: 0
  - `city`: 0
  - `region`: 0
  - `country`: 0
  - `segment`: 0

### `employees`
- Rows: **30**
- Columns:
  - `employee_id` `INTEGER` PK
  - `first_name` `TEXT` NOT NULL
  - `last_name` `TEXT` NOT NULL
  - `title` `TEXT`
  - `hire_date` `TEXT`
  - `city` `TEXT`
  - `country` `TEXT`
- Indexes:
  - none
- Null Stats:
  - `employee_id`: 0
  - `first_name`: 0
  - `last_name`: 0
  - `title`: 0
  - `hire_date`: 0
  - `city`: 0
  - `country`: 0

### `order_details`
- Rows: **383208**
- Columns:
  - `order_id` `INTEGER` PK NOT NULL
  - `product_id` `INTEGER` PK NOT NULL
  - `unit_price` `REAL` NOT NULL
  - `quantity` `INTEGER` NOT NULL
  - `discount` `REAL` NOT NULL
- Indexes:
  - `idx_order_details_product` (product_id)
  - `sqlite_autoindex_order_details_1` (order_id, product_id)
- Null Stats:
  - `order_id`: 0
  - `product_id`: 0
  - `unit_price`: 0
  - `quantity`: 0
  - `discount`: 0

### `orders`
- Rows: **96000**
- Columns:
  - `order_id` `INTEGER` PK
  - `customer_id` `TEXT` NOT NULL
  - `employee_id` `INTEGER` NOT NULL
  - `order_date` `TEXT` NOT NULL
  - `required_date` `TEXT`
  - `shipped_date` `TEXT`
  - `ship_via` `INTEGER`
  - `freight` `REAL`
  - `ship_city` `TEXT`
  - `ship_country` `TEXT`
  - `market` `TEXT`
- Indexes:
  - `idx_orders_market` (market)
  - `idx_orders_date` (order_date)
  - `idx_orders_employee` (employee_id)
  - `idx_orders_customer` (customer_id)
- Null Stats:
  - `order_id`: 0
  - `customer_id`: 0
  - `employee_id`: 0
  - `order_date`: 0
  - `required_date`: 0
  - `shipped_date`: 0
  - `ship_via`: 0
  - `freight`: 0
  - `ship_city`: 0
  - `ship_country`: 0
  - `market`: 0

### `products`
- Rows: **300**
- Columns:
  - `product_id` `INTEGER` PK
  - `product_name` `TEXT` NOT NULL
  - `supplier_id` `INTEGER` NOT NULL
  - `category_id` `INTEGER` NOT NULL
  - `quantity_per_unit` `TEXT`
  - `unit_price` `REAL` NOT NULL
  - `units_in_stock` `INTEGER` NOT NULL
  - `discontinued` `INTEGER` NOT NULL
- Indexes:
  - `idx_products_category` (category_id)
- Null Stats:
  - `product_id`: 0
  - `product_name`: 0
  - `supplier_id`: 0
  - `category_id`: 0
  - `quantity_per_unit`: 0
  - `unit_price`: 0
  - `units_in_stock`: 0
  - `discontinued`: 0

### `shippers`
- Rows: **3**
- Columns:
  - `shipper_id` `INTEGER` PK
  - `company_name` `TEXT` NOT NULL
- Indexes:
  - none
- Null Stats:
  - `shipper_id`: 0
  - `company_name`: 0

### `suppliers`
- Rows: **40**
- Columns:
  - `supplier_id` `INTEGER` PK
  - `company_name` `TEXT` NOT NULL
  - `contact_name` `TEXT`
  - `city` `TEXT`
  - `country` `TEXT`
- Indexes:
  - none
- Null Stats:
  - `supplier_id`: 0
  - `company_name`: 0
  - `contact_name`: 0
  - `city`: 0
  - `country`: 0