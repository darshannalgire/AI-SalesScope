# main.py
import os
import re
import traceback
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For Flask (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Static folder for saved charts
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Canonical required columns (logical order). Keep these names in lower-case.
REQUIRED_COLUMNS_ORDER = [
    "order date",
    "region",
    "category",
    "sales",
    "profit",
    "discount",
    "order id",
    "customer name"
]

# Acceptable variant names for each canonical column (normalized forms).
COLUMN_VARIANTS = {
    "order date": ["order date", "orderdate", "order_date", "date", "order date (order date)", "order date (date)"],
    "region": ["region", "state", "territory", "region/state"],
    "category": ["category", "product category", "category name", "category_type"],
    "sales": ["sales", "sale", "amount", "total sales", "sales amount"],
    "profit": ["profit", "profit amount", "net profit"],
    "discount": ["discount", "discounts", "discount (%)", "discount %", "discount_rate", "discount rate", "discount percent"],
    "order id": ["order id", "order_id", "orderid", "order number", "order no", "order #"],
    "customer name": ["customer name", "customer", "customer_name", "customer id/name", "customer id"]
}

# If True, required columns must appear in sequence (but variants allowed).
# If False, only presence is checked (order ignored). Set False for robustness.
ENFORCE_SEQUENCE = False

# Helpers --------------------------------------------------------------------

def normalize_column_name(name):
    """Normalize a raw column name:
       - lower-case
       - strip whitespace
       - replace punctuation/underscores/slashes with spaces
       - convert % to 'percent'
       - collapse multiple spaces
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = s.replace('%', ' percent ')
    for ch in ['_', '/', '\\', '-', '—', '–', '.']:
        s = s.replace(ch, ' ')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = ' '.join(s.split())
    return s

def read_file_header_and_df(filepath):
    """Read file into dataframe. Returns (df, None) or (None, error_message).
       Supports CSV, XLS, XLSX.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".csv":
            # try utf-8 first, fallback to latin1
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except Exception:
                df = pd.read_csv(filepath, encoding='latin1')
        else:
            # read_excel will auto-detect engine
            df = pd.read_excel(filepath)
    except Exception as e:
        return None, f"Failed to read file: {e}"
    return df, None

def validate_required_columns(df_columns):
    """
    Validate the file columns against REQUIRED_COLUMNS_ORDER using COLUMN_VARIANTS.
    Returns (True, "OK") if OK, or (False, message).
    Behavior:
      - normalize df_columns using normalize_column_name
      - try to match each required canonical column by searching for any of its variants
      - if ENFORCE_SEQUENCE True: variants must appear in the same order (but may be separated by other columns)
      - if ENFORCE_SEQUENCE False: only presence is checked (order ignored)
    """
    cols_normalized = [normalize_column_name(c) for c in df_columns]

    # Build reverse lookup of variant -> canonical
    variant_to_canonical = {}
    for canon, variants in COLUMN_VARIANTS.items():
        vs = set([normalize_column_name(v) for v in variants] + [normalize_column_name(canon)])
        for v in vs:
            variant_to_canonical[v] = canon

    if ENFORCE_SEQUENCE:
        idx = 0
        for required in REQUIRED_COLUMNS_ORDER:
            found = False
            j = idx
            while j < len(cols_normalized):
                colnorm = cols_normalized[j]
                if colnorm in variant_to_canonical and variant_to_canonical[colnorm] == required:
                    idx = j + 1
                    found = True
                    break
                j += 1
            if not found:
                sample = ", ".join(cols_normalized[:50])
                return False, f"Required column '{required}' not found in required order. Found headers (normalized): {sample}"
        return True, "OK"
    else:
        missing = []
        for required in REQUIRED_COLUMNS_ORDER:
            ok = False
            for colnorm in cols_normalized:
                if colnorm in variant_to_canonical and variant_to_canonical[colnorm] == required:
                    ok = True
                    break
            if not ok:
                missing.append(required)
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
        return True, "OK"

# Utility: choose a reasonable tick list for long x-axis
def reduced_ticks_labels(labels, max_ticks=12):
    n = len(labels)
    if n == 0:
        return [], []
    step = max(1, n // max_ticks)
    ticks = list(range(0, n, step))
    ticklabels = [labels[i] for i in ticks]
    return ticks, ticklabels

# Core processing -----------------------------------------------------------

def process_data(filepath, top_k_products=10, product_forecast_k=5):
    """
    Robust version of process_data with defensive initialization and detailed error reporting.
    Returns a dict:
      - error: bool
      - message: str (if error)
      - trace: str (traceback, on error)
    On success returns keys:
      error=False, total_sales, total_profit, year_table, best_year,
      top_products, bottom_products, product_forecasts, graphs, forecast_values
    """
    # Defensive inits
    df = None
    year_agg = pd.DataFrame()
    monthly_sales = pd.DataFrame()
    top_products_df = pd.DataFrame(columns=['product', 'sales'])
    bottom_products_df = pd.DataFrame(columns=['product', 'sales'])
    graphs = {}
    product_forecasts = []
    product_col = None

    try:
        # --- Read file ---
        df, err = read_file_header_and_df(filepath)
        if df is None:
            return {'error': True, 'message': err or "Unable to read file."}

        # --- Validate required columns ---
        valid, msg = validate_required_columns(list(df.columns))
        if not valid:
            normalized = [normalize_column_name(c) for c in df.columns]
            return {'error': True, 'message': f"{msg}\nFound headers (normalized): {', '.join(normalized)}"}

        # Normalize column names to safe internal names
        col_map = {c: normalize_column_name(c) for c in df.columns}
        df = df.rename(columns=col_map)

        # Ensure numeric columns exist and coerce
        for col in ['sales', 'profit', 'discount']:
            if col not in df.columns:
                return {'error': True, 'message': f"Column '{col}' missing after normalization."}
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Parse order date
        if 'order date' not in df.columns:
            return {'error': True, 'message': "Column 'Order Date' not found after normalization."}
        df['order date'] = pd.to_datetime(df['order date'], errors='coerce')
        df.dropna(subset=['order date'], inplace=True)
        if df.empty:
            return {'error': True, 'message': "No valid order date rows found after parsing 'order date'."}

        # Basic cleaning
        df.drop_duplicates(inplace=True)

        # --- Aggregations ---
        df['year'] = df['order date'].dt.year
        year_agg = df.groupby('year').agg(sales=('sales', 'sum'), profit=('profit', 'sum')).reset_index()
        best_year = int(year_agg.loc[year_agg['sales'].idxmax()]['year']) if (not year_agg.empty) else None

        # detect product column (pick first column containing 'product')
        product_col_candidates = [c for c in df.columns if 'product' in c]
        if product_col_candidates:
            product_col = product_col_candidates[0]
        else:
            product_col = 'product name' if 'product name' in df.columns else None

        if product_col and product_col in df.columns:
            prod_sales = df.groupby(product_col)['sales'].sum().reset_index().rename(columns={product_col: 'product'})
            top_products_df = prod_sales.sort_values('sales', ascending=False).head(top_k_products)
            bottom_products_df = prod_sales.sort_values('sales', ascending=True).head(top_k_products)
        else:
            top_products_df = pd.DataFrame(columns=['product', 'sales'])
            bottom_products_df = pd.DataFrame(columns=['product', 'sales'])

        # Monthly sales for trend (use period M string index)
        df['month'] = df['order date'].dt.to_period('M').astype(str)
        monthly_sales = df.groupby('month')['sales'].sum().reset_index().sort_values('month')

        # --- Create charts (each chart wrapped in try so one failure won't kill everything) ---
        # Region-wise sales
        try:
            if 'region' in df.columns:
                region_agg = df.groupby('region')['sales'].sum().reset_index()
                plt.figure(figsize=(8, 4), dpi=150)
                sns.barplot(x='region', y='sales', data=region_agg)
                plt.title('Region-wise Sales')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.tight_layout()
                region_path = os.path.join(STATIC_FOLDER, 'region_sales.png')
                plt.savefig(region_path, bbox_inches='tight', dpi=150); plt.close()
                graphs['region'] = os.path.basename(region_path)
        except Exception:
            pass

        # Category-wise pie
        try:
            if 'category' in df.columns:
                plt.figure(figsize=(5, 5), dpi=150)
                df.groupby('category')['sales'].sum().plot.pie(autopct='%1.1f%%')
                plt.title('Category-wise Sales'); plt.ylabel('')
                plt.tight_layout()
                category_path = os.path.join(STATIC_FOLDER, 'category_sales.png')
                plt.savefig(category_path, bbox_inches='tight', dpi=150); plt.close()
                graphs['category'] = os.path.basename(category_path)
        except Exception:
            pass

        # Monthly sales trend (improved)
        try:
            if not monthly_sales.empty:
                ms = monthly_sales.sort_values('month').reset_index(drop=True)
                labels = ms['month'].astype(str).tolist()
                y_vals = ms['sales'].tolist()

                plt.figure(figsize=(12, 4), dpi=150)
                plt.plot(range(len(labels)), y_vals, marker='o', linewidth=1.25, markersize=4)
                plt.title('Monthly Sales Trend')
                ticks, ticklabels = reduced_ticks_labels(labels, max_ticks=12)
                plt.xticks(ticks, ticklabels, rotation=45, ha='right', fontsize=8)
                plt.ylabel('sales')
                plt.tight_layout()
                monthly_path = os.path.join(STATIC_FOLDER, 'monthly_sales.png')
                plt.savefig(monthly_path, bbox_inches='tight', dpi=150); plt.close()
                graphs['monthly'] = os.path.basename(monthly_path)
        except Exception:
            pass

        # Discount vs Profit scatter
        try:
            plt.figure(figsize=(6, 4), dpi=150)
            sns.scatterplot(x='discount', y='profit', data=df, alpha=0.6)
            plt.title('Discount vs Profit')
            plt.tight_layout()
            discount_profit_path = os.path.join(STATIC_FOLDER, 'discount_profit.png')
            plt.savefig(discount_profit_path, bbox_inches='tight', dpi=150); plt.close()
            graphs['discount_profit'] = os.path.basename(discount_profit_path)
        except Exception:
            pass

        # Year vs Sales & Profit
        try:
            if not year_agg.empty:
                plt.figure(figsize=(8, 4), dpi=150)
                x = np.arange(len(year_agg))
                w = 0.35
                plt.bar(x - w/2, year_agg['sales'], width=w, label='Sales')
                plt.bar(x + w/2, year_agg['profit'], width=w, label='Profit')
                plt.xticks(x, year_agg['year'].astype(str))
                plt.title('Year wise Sales and Profit')
                plt.legend()
                plt.tight_layout()
                year_path = os.path.join(STATIC_FOLDER, 'year_vs_profit.png')
                plt.savefig(year_path, bbox_inches='tight', dpi=150); plt.close()
                graphs['year_vs_profit'] = os.path.basename(year_path)
        except Exception:
            pass

        # Top products chart
        try:
            if not top_products_df.empty:
                plt.figure(figsize=(8, 4), dpi=150)
                sns.barplot(x='sales', y='product', data=top_products_df.sort_values('sales', ascending=False))
                plt.title(f'Top {top_k_products} Products by Sales')
                plt.tight_layout()
                top_prod_path = os.path.join(STATIC_FOLDER, 'top_products.png')
                plt.savefig(top_prod_path, bbox_inches='tight', dpi=150); plt.close()
                graphs['top_products'] = os.path.basename(top_prod_path)
        except Exception:
            pass

        # --- Forecasting: overall monthly forecast (simple linear) ---
        forecast = []
        forecast_path = None
        try:
            ms = monthly_sales.reset_index(drop=True)
            if not ms.empty:
                ms['month_num'] = np.arange(len(ms))
                if len(ms) >= 2:
                    model = LinearRegression()
                    model.fit(ms[['month_num']], ms['sales'])
                    future_idx = np.arange(len(ms), len(ms) + 3).reshape(-1, 1)
                    forecast = model.predict(future_idx).tolist()

                    # overlay chart (use index numbers for x)
                    plt.figure(figsize=(10, 4), dpi=150)
                    plt.plot(ms['month_num'], ms['sales'], label='Actual', marker='o', markersize=3)
                    plt.plot(future_idx, forecast, '--', label='Forecast', color='red', linewidth=1)
                    plt.legend()
                    plt.title('Sales Forecast (Linear Regression)')
                    plt.tight_layout()
                    forecast_path = os.path.join(STATIC_FOLDER, 'forecast_sales.png')
                    plt.savefig(forecast_path, bbox_inches='tight', dpi=150); plt.close()
                    graphs['forecast'] = os.path.basename(forecast_path)
        except Exception:
            forecast = []
            forecast_path = None

        # --- Product-level monthly trends and short linear forecasts ---
        product_forecasts = []
        try:
            if product_col and product_col in df.columns and not top_products_df.empty:
                # Build pivot: index = month string (full range), columns = product, values = sales
                all_months = sorted(df['month'].unique())
                pivot = df.groupby(['month', product_col])['sales'].sum().unstack(fill_value=0)
                # ensure all months present
                pivot = pivot.reindex(index=all_months, fill_value=0)

                # choose top products list
                top_products_list = list(top_products_df['product'].head(product_forecast_k))
                pivot_top = pivot.loc[:, [p for p in top_products_list if p in pivot.columns]] if not pivot.empty else pd.DataFrame()

                # per-product forecasts
                for prod in top_products_list:
                    if prod not in pivot.columns:
                        continue
                    series = pivot[prod].reset_index(drop=True)
                    history = [float(x) for x in series.values.tolist()]
                    pred = []
                    if len(series) >= 2:
                        X = np.arange(len(series)).reshape(-1, 1)
                        y = series.values
                        m = LinearRegression()
                        m.fit(X, y)
                        future = np.arange(len(series), len(series) + 3).reshape(-1, 1)
                        pred = m.predict(future).tolist()
                    product_forecasts.append({
                        'product': prod,
                        'history': [round(float(x), 2) for x in history],
                        'forecast': [round(float(x), 2) for x in pred]
                    })

                # product trends combined plot (pivot_top)
                if not pivot_top.empty:
                    plt.figure(figsize=(12, 5), dpi=150)
                    labels = list(pivot_top.index)
                    for col in pivot_top.columns:
                        vals = pivot_top[col].values
                        plt.plot(range(len(labels)), vals, marker='.', markersize=4, linewidth=1, label=str(col))
                    ticks, ticklabels = reduced_ticks_labels(labels, max_ticks=12)
                    plt.xticks(ticks, ticklabels, rotation=45, ha='right', fontsize=8)
                    plt.title(f'Top {product_forecast_k} Products - Monthly Sales')
                    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
                    plt.tight_layout()
                    product_trend_path = os.path.join(STATIC_FOLDER, 'product_trends.png')
                    plt.savefig(product_trend_path, bbox_inches='tight', dpi=150); plt.close()
                    graphs['product_trends'] = os.path.basename(product_trend_path)
        except Exception:
            product_forecasts = []

        # Totals
        total_sales = float(df['sales'].sum() if 'sales' in df.columns else 0)
        total_profit = float(df['profit'].sum() if 'profit' in df.columns else 0)

        # --- Determine product-level trend direction (compare recent vs forecast) ---
        last_k = 3
        increase_threshold_pct = 10.0   # classify as "increasing" if forecast_mean ≥ recent_mean * (1 + 10%)
        decrease_threshold_pct = -5.0   # classify as "decreasing" if forecast_mean ≤ recent_mean * (1 - 5%)

        increasing_products = []
        decreasing_products = []
        flat_products = []

        for pf in product_forecasts:
            hist = pf.get('history', []) or []
            fc = pf.get('forecast', []) or []
            if len(fc) == 0 or len(hist) == 0:
                flat_products.append({'product': pf.get('product'), 'reason': 'insufficient data'})
                continue

            recent_vals = hist[-last_k:] if len(hist) >= 1 else hist
            recent_mean = float(np.mean(recent_vals)) if len(recent_vals) > 0 else 0.0
            forecast_mean = float(np.mean(fc))

            if recent_mean == 0:
                if forecast_mean > 0:
                    pct_change = 999.0
                    increasing_products.append({'product': pf['product'], 'recent_mean': round(recent_mean,2), 'forecast_mean': round(forecast_mean,2), 'pct_change': pct_change})
                else:
                    flat_products.append({'product': pf['product'], 'recent_mean': round(recent_mean,2), 'forecast_mean': round(forecast_mean,2), 'pct_change': 0.0})
                continue

            pct_change = (forecast_mean - recent_mean) / recent_mean * 100.0

            if pct_change >= increase_threshold_pct:
                increasing_products.append({'product': pf['product'], 'recent_mean': round(recent_mean,2), 'forecast_mean': round(forecast_mean,2), 'pct_change': round(pct_change,2)})
            elif pct_change <= decrease_threshold_pct:
                decreasing_products.append({'product': pf['product'], 'recent_mean': round(recent_mean,2), 'forecast_mean': round(forecast_mean,2), 'pct_change': round(pct_change,2)})
            else:
                flat_products.append({'product': pf['product'], 'recent_mean': round(recent_mean,2), 'forecast_mean': round(forecast_mean,2), 'pct_change': round(pct_change,2)})
       
        # --- Tidy outputs (convert dataframes to dicts) ---
        year_table = (
            year_agg.sort_values('year', ascending=False).to_dict(orient='records')
            if not year_agg.empty else []
        )
        top_products = (
            top_products_df.sort_values('sales', ascending=False).to_dict(orient='records')
            if not top_products_df.empty else []
        )
        bottom_products = (
            bottom_products_df.sort_values('sales', ascending=True).to_dict(orient='records')
            if not bottom_products_df.empty else []
        )
  
        return {
            'error': False,
            'total_sales': round(total_sales, 2),
            'total_profit': round(total_profit, 2),
            'year_table': year_table,
            'best_year': best_year,
            'top_products': top_products,
            'bottom_products': bottom_products,
            'product_forecasts': product_forecasts,
            'graphs': graphs,
            'forecast_values': [round(float(val), 2) for val in forecast],
            # 🟢 ADD THESE THREE LINES:
            'increasing_products': increasing_products,
            'decreasing_products': decreasing_products,
            'flat_products': flat_products
        }


    except Exception as e:
        tb = traceback.format_exc()
        # print to server logs for developer debugging
        print("process_data error:", str(e))
        print(tb)
        return {'error': True, 'message': f"Processing error: {e}", 'trace': tb}
