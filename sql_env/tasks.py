"""Task definitions for the SQL Query Environment.

Contains 9 tasks across 3 difficulty tiers (easy, medium, hard), each with:
- Natural-language question
- Database schema (CREATE TABLE statements)
- Seed data (INSERT statements)
- Gold-standard SQL query
- Pre-computed gold result for grading

All tasks use a shared e-commerce/company database schema.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import sqlite3

from sql_env.db_utils import create_database, execute_query


# ---------------------------------------------------------------------------
# Shared Schema & Seed Data
# ---------------------------------------------------------------------------

SHARED_SCHEMA = """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    budget REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    department_id INTEGER NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(id),
    FOREIGN KEY (manager_id) REFERENCES employees(id)
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    city TEXT,
    registered_date TEXT NOT NULL
);

CREATE TABLE product_categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (category_id) REFERENCES product_categories(id)
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    total_amount REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

SHARED_SEED = """
-- Departments
INSERT INTO departments (id, name, budget) VALUES
    (1, 'Engineering', 500000.00),
    (2, 'Sales', 300000.00),
    (3, 'Marketing', 200000.00),
    (4, 'HR', 150000.00),
    (5, 'Finance', 250000.00);

-- Employees
INSERT INTO employees (id, first_name, last_name, email, department_id, salary, hire_date, manager_id) VALUES
    (1, 'Alice', 'Johnson', 'alice@company.com', 1, 120000, '2020-01-15', NULL),
    (2, 'Bob', 'Smith', 'bob@company.com', 1, 110000, '2020-03-20', 1),
    (3, 'Carol', 'Williams', 'carol@company.com', 1, 105000, '2021-06-01', 1),
    (4, 'David', 'Brown', 'david@company.com', 1, 95000, '2022-01-10', 1),
    (5, 'Eve', 'Davis', 'eve@company.com', 2, 85000, '2019-07-22', NULL),
    (6, 'Frank', 'Miller', 'frank@company.com', 2, 80000, '2020-11-05', 5),
    (7, 'Grace', 'Wilson', 'grace@company.com', 2, 78000, '2021-09-15', 5),
    (8, 'Hank', 'Moore', 'hank@company.com', 3, 75000, '2020-05-01', NULL),
    (9, 'Ivy', 'Taylor', 'ivy@company.com', 3, 72000, '2021-03-18', 8),
    (10, 'Jack', 'Anderson', 'jack@company.com', 4, 70000, '2019-01-05', NULL),
    (11, 'Karen', 'Thomas', 'karen@company.com', 4, 68000, '2022-08-20', 10),
    (12, 'Leo', 'Jackson', 'leo@company.com', 5, 90000, '2020-02-14', NULL),
    (13, 'Mia', 'White', 'mia@company.com', 5, 88000, '2021-11-30', 12),
    (14, 'Nathan', 'Harris', 'nathan@company.com', 1, 100000, '2023-04-01', 1),
    (15, 'Olivia', 'Martin', 'olivia@company.com', 2, 82000, '2023-06-15', 5);

-- Product Categories
INSERT INTO product_categories (id, name) VALUES
    (1, 'Electronics'),
    (2, 'Books'),
    (3, 'Clothing'),
    (4, 'Home & Garden'),
    (5, 'Sports');

-- Products
INSERT INTO products (id, name, category_id, price, stock_quantity) VALUES
    (1, 'Laptop Pro 15', 1, 1299.99, 50),
    (2, 'Wireless Mouse', 1, 29.99, 200),
    (3, 'USB-C Hub', 1, 49.99, 150),
    (4, 'Python Programming', 2, 39.99, 100),
    (5, 'Data Science Handbook', 2, 44.99, 80),
    (6, 'SQL Mastery', 2, 34.99, 120),
    (7, 'Running Shoes', 3, 89.99, 75),
    (8, 'Winter Jacket', 3, 149.99, 40),
    (9, 'Garden Tool Set', 4, 59.99, 60),
    (10, 'Indoor Plant Kit', 4, 24.99, 90),
    (11, 'Yoga Mat', 5, 19.99, 200),
    (12, 'Dumbbells Set', 5, 79.99, 45);

-- Customers
INSERT INTO customers (id, name, email, city, registered_date) VALUES
    (1, 'John Doe', 'john@email.com', 'New York', '2023-01-10'),
    (2, 'Jane Roe', 'jane@email.com', 'Los Angeles', '2023-02-15'),
    (3, 'Sam Lee', 'sam@email.com', 'Chicago', '2023-03-20'),
    (4, 'Pat Kim', 'pat@email.com', 'Houston', '2023-04-25'),
    (5, 'Alex Chen', 'alex@email.com', 'New York', '2023-06-01'),
    (6, 'Taylor Swift', 'taylor@email.com', 'Nashville', '2024-01-15'),
    (7, 'Morgan Yu', 'morgan@email.com', 'San Francisco', '2024-03-10');

-- Orders
INSERT INTO orders (id, customer_id, order_date, total_amount, status) VALUES
    (1, 1, '2024-01-15', 1329.98, 'completed'),
    (2, 2, '2024-01-20', 89.99, 'completed'),
    (3, 1, '2024-02-10', 74.98, 'completed'),
    (4, 3, '2024-02-28', 149.99, 'completed'),
    (5, 4, '2024-03-05', 59.99, 'shipped'),
    (6, 5, '2024-03-15', 1349.98, 'completed'),
    (7, 2, '2024-04-01', 34.99, 'completed'),
    (8, 1, '2024-04-10', 19.99, 'completed'),
    (9, 3, '2024-05-20', 109.98, 'pending'),
    (10, 6, '2024-06-01', 44.99, 'completed'),
    (11, 7, '2024-06-15', 79.99, 'shipped'),
    (12, 5, '2024-07-01', 89.99, 'completed');

-- Order Items
INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 1, 1299.99),
    (2, 1, 2, 1, 29.99),
    (3, 2, 7, 1, 89.99),
    (4, 3, 4, 1, 39.99),
    (5, 3, 6, 1, 34.99),
    (6, 4, 8, 1, 149.99),
    (7, 5, 9, 1, 59.99),
    (8, 6, 1, 1, 1299.99),
    (9, 6, 3, 1, 49.99),
    (10, 7, 6, 1, 34.99),
    (11, 8, 11, 1, 19.99),
    (12, 9, 7, 1, 89.99),
    (13, 9, 11, 1, 19.99),
    (14, 10, 5, 1, 44.99),
    (15, 11, 12, 1, 79.99),
    (16, 12, 7, 1, 89.99);
"""


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single SQL task for the agent to solve."""

    task_id: str
    difficulty: str  # "easy", "medium", "hard"
    question: str
    hint: str  # Short hint about SQL concepts needed
    gold_query: str
    schema_sql: str = field(default=SHARED_SCHEMA, repr=False)
    seed_sql: str = field(default=SHARED_SEED, repr=False)

    # Computed at load time
    _gold_rows: Optional[List[Tuple[Any, ...]]] = field(default=None, repr=False)
    _gold_columns: Optional[List[str]] = field(default=None, repr=False)

    def compute_gold_result(self) -> None:
        """Execute the gold query and cache the result."""
        conn = create_database(self.schema_sql, self.seed_sql)
        try:
            rows, cols, err = execute_query(conn, self.gold_query)
            if err:
                raise ValueError(
                    f"Gold query for task '{self.task_id}' failed: {err}"
                )
            self._gold_rows = rows
            self._gold_columns = cols
        finally:
            conn.close()

    @property
    def gold_rows(self) -> List[Tuple[Any, ...]]:
        if self._gold_rows is None:
            self.compute_gold_result()
        return self._gold_rows  # type: ignore

    @property
    def gold_columns(self) -> List[str]:
        if self._gold_columns is None:
            self.compute_gold_result()
        return self._gold_columns  # type: ignore


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {}


def _register(task: Task) -> None:
    TASKS[task.task_id] = task


# ============================= EASY ========================================

_register(Task(
    task_id="easy_01",
    difficulty="easy",
    question="List all employees in the Sales department. Show their first name, last name, and salary.",
    hint="Use SELECT with a WHERE clause. You'll need to join or filter by department.",
    gold_query="""
        SELECT e.first_name, e.last_name, e.salary
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        WHERE d.name = 'Sales'
        ORDER BY e.last_name;
    """,
))

_register(Task(
    task_id="easy_02",
    difficulty="easy",
    question="How many orders were placed in 2024? Return a single count.",
    hint="Use COUNT() with a WHERE clause filtering on order_date.",
    gold_query="""
        SELECT COUNT(*) AS order_count
        FROM orders
        WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';
    """,
))

_register(Task(
    task_id="easy_03",
    difficulty="easy",
    question="What are the distinct product categories? List just the category names, sorted alphabetically.",
    hint="Use SELECT DISTINCT or query the product_categories table directly.",
    gold_query="""
        SELECT name
        FROM product_categories
        ORDER BY name;
    """,
))


# ============================= MEDIUM ======================================

_register(Task(
    task_id="medium_01",
    difficulty="medium",
    question="Show the total revenue per product category. Display the category name and total revenue, sorted by revenue descending.",
    hint="Join order_items → products → product_categories, then GROUP BY category with SUM.",
    gold_query="""
        SELECT pc.name AS category, SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM order_items oi
        JOIN products p ON oi.product_id = p.id
        JOIN product_categories pc ON p.category_id = pc.id
        GROUP BY pc.name
        ORDER BY total_revenue DESC;
    """,
))

_register(Task(
    task_id="medium_02",
    difficulty="medium",
    question="Find employees who manage more than 2 other employees. Show the manager's first name, last name, and the number of direct reports.",
    hint="Use a self-join or subquery on manager_id, GROUP BY, and HAVING.",
    gold_query="""
        SELECT m.first_name, m.last_name, COUNT(e.id) AS direct_reports
        FROM employees e
        JOIN employees m ON e.manager_id = m.id
        GROUP BY m.id, m.first_name, m.last_name
        HAVING COUNT(e.id) > 2
        ORDER BY direct_reports DESC;
    """,
))

_register(Task(
    task_id="medium_03",
    difficulty="medium",
    question="List customers who have placed at least 2 orders. Show the customer name and their total number of orders, sorted by order count descending.",
    hint="Join customers and orders, then GROUP BY customer with HAVING COUNT >= 2.",
    gold_query="""
        SELECT c.name, COUNT(o.id) AS order_count
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        GROUP BY c.id, c.name
        HAVING COUNT(o.id) >= 2
        ORDER BY order_count DESC;
    """,
))


# ============================= HARD ========================================

_register(Task(
    task_id="hard_01",
    difficulty="hard",
    question="Rank departments by their average employee salary (descending). Show department name, average salary (rounded to 2 decimal places), and the rank. Include all departments.",
    hint="Use AVG with GROUP BY and either a window function (RANK/ROW_NUMBER) or a subquery for ranking.",
    gold_query="""
        SELECT
            d.name AS department,
            ROUND(AVG(e.salary), 2) AS avg_salary,
            RANK() OVER (ORDER BY AVG(e.salary) DESC) AS salary_rank
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        GROUP BY d.id, d.name
        ORDER BY avg_salary DESC;
    """,
))

_register(Task(
    task_id="hard_02",
    difficulty="hard",
    question="Find products that have never been ordered. Show the product name and its category name.",
    hint="Use a LEFT JOIN with IS NULL, or a NOT IN / NOT EXISTS subquery.",
    gold_query="""
        SELECT p.name AS product_name, pc.name AS category_name
        FROM products p
        JOIN product_categories pc ON p.category_id = pc.id
        LEFT JOIN order_items oi ON oi.product_id = p.id
        WHERE oi.id IS NULL
        ORDER BY p.name;
    """,
))

_register(Task(
    task_id="hard_03",
    difficulty="hard",
    question="Show the cumulative (running) total of order amounts by date. Display the order date and the running total up to and including that date, sorted by date.",
    hint="Use a window function: SUM() OVER (ORDER BY ...) for a running total.",
    gold_query="""
        SELECT
            order_date,
            total_amount,
            SUM(total_amount) OVER (ORDER BY order_date, id) AS running_total
        FROM orders
        ORDER BY order_date, id;
    """,
))


# ============================= EXPERT ======================================

_register(Task(
    task_id="expert_01",
    difficulty="expert",
    question="Using a CTE (WITH clause), find the full management chain for each employee. Show the employee's first name, last name, and the number of management levels above them (0 for top-level managers). Sort by chain length descending, then by last name.",
    hint="Use a recursive CTE with the manager_id column to walk up the management chain. Count the levels.",
    gold_query="""
        WITH RECURSIVE chain AS (
            SELECT id, first_name, last_name, manager_id, 0 AS depth
            FROM employees
            WHERE manager_id IS NULL
            UNION ALL
            SELECT e.id, e.first_name, e.last_name, e.manager_id, c.depth + 1
            FROM employees e
            JOIN chain c ON e.manager_id = c.id
        )
        SELECT first_name, last_name, depth
        FROM chain
        ORDER BY depth DESC, last_name;
    """,
))

_register(Task(
    task_id="expert_02",
    difficulty="expert",
    question="For each department, calculate: the number of employees, average salary, and classify the department as 'Over Budget' if total salaries exceed the department budget, or 'Within Budget' otherwise. Show department name, employee count, average salary (rounded to 2 decimals), total salary, budget, and budget status. Sort by department name.",
    hint="Use CASE expression with GROUP BY. Join employees with departments and compare SUM(salary) to budget.",
    gold_query="""
        SELECT
            d.name AS department,
            COUNT(e.id) AS emp_count,
            ROUND(AVG(e.salary), 2) AS avg_salary,
            SUM(e.salary) AS total_salary,
            d.budget,
            CASE
                WHEN SUM(e.salary) > d.budget THEN 'Over Budget'
                ELSE 'Within Budget'
            END AS budget_status
        FROM departments d
        JOIN employees e ON e.department_id = d.id
        GROUP BY d.id, d.name, d.budget
        ORDER BY d.name;
    """,
))

_register(Task(
    task_id="expert_03",
    difficulty="expert",
    question="Find employees whose salary is above the average salary of their own department. Show the employee's first name, last name, their salary, their department name, and the department average salary (rounded to 2 decimals). Sort by department name, then by salary descending.",
    hint="Use a correlated subquery or a CTE/subquery that computes per-department average, then filter employees above it.",
    gold_query="""
        SELECT
            e.first_name,
            e.last_name,
            e.salary,
            d.name AS department,
            ROUND(dept_avg.avg_sal, 2) AS dept_avg_salary
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        JOIN (
            SELECT department_id, AVG(salary) AS avg_sal
            FROM employees
            GROUP BY department_id
        ) dept_avg ON e.department_id = dept_avg.department_id
        WHERE e.salary > dept_avg.avg_sal
        ORDER BY d.name, e.salary DESC;
    """,
))

_register(Task(
    task_id="expert_04",
    difficulty="expert",
    question="Analyze monthly order trends: for each month that has orders, show the month (as YYYY-MM), the number of orders, total revenue, and the percentage change in revenue compared to the previous month (NULL for the first month). Round the percentage to 2 decimal places. Sort by month.",
    hint="Use strftime to extract month, window functions LAG() to get previous month's revenue, then calculate percentage change.",
    gold_query="""
        WITH monthly AS (
            SELECT
                strftime('%Y-%m', order_date) AS month,
                COUNT(*) AS order_count,
                SUM(total_amount) AS revenue
            FROM orders
            GROUP BY strftime('%Y-%m', order_date)
        )
        SELECT
            month,
            order_count,
            revenue,
            ROUND(
                (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0
                / LAG(revenue) OVER (ORDER BY month),
                2
            ) AS revenue_growth_pct
        FROM monthly
        ORDER BY month;
    """,
))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_task(task_id: str) -> Task:
    """Get a task by its ID. Raises KeyError if not found."""
    if task_id not in TASKS:
        raise KeyError(
            f"Task '{task_id}' not found. Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def get_tasks_by_difficulty(difficulty: str) -> List[Task]:
    """Get all tasks for a given difficulty tier."""
    return [t for t in TASKS.values() if t.difficulty == difficulty]


def get_all_task_ids() -> List[str]:
    """Return all registered task IDs."""
    return list(TASKS.keys())


def get_random_task(difficulty: Optional[str] = None, seed: Optional[int] = None) -> Task:
    """Pick a random task, optionally filtered by difficulty."""
    import random

    rng = random.Random(seed)
    candidates = list(TASKS.values())
    if difficulty:
        candidates = [t for t in candidates if t.difficulty == difficulty]
    if not candidates:
        raise ValueError(f"No tasks found for difficulty='{difficulty}'")
    return rng.choice(candidates)
