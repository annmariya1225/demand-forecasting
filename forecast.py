"""
Demand forecasting with a Random Forest.

Run it on the bundled Walmart dataset:
    python forecast.py

Or point it at your own CSV:
    python forecast.py --csv your_data.csv --date-col Date --demand-col Sales
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ['Month', 'WeekOfYear', 'Quarter',
            'Lag_1', 'Lag_2', 'Lag_4',
            'Rolling_4', 'Rolling_12', 'Holiday']


def load_data(csv_path, date_col, demand_col, product_col, holiday_col, selected_product):
    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not in CSV. Available: {list(df.columns)}")
    if demand_col not in df.columns:
        raise ValueError(f"Demand column '{demand_col}' not in CSV. Available: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
    df = df.dropna(subset=[date_col, demand_col])

    rename_map = {date_col: 'Date', demand_col: 'Demand'}
    if product_col and product_col in df.columns:
        rename_map[product_col] = 'Product'
    if holiday_col and holiday_col in df.columns:
        rename_map[holiday_col] = 'Holiday'
    df = df.rename(columns=rename_map)

    if selected_product is not None and 'Product' in df.columns:
        df = df[df['Product'].astype(str) == str(selected_product)].copy()
        if df.empty:
            raise ValueError(f"No rows found for product '{selected_product}'")
    elif 'Product' in df.columns:
        agg = {'Demand': 'sum'}
        if 'Holiday' in df.columns:
            agg['Holiday'] = 'max'
        df = df.groupby('Date', as_index=False).agg(agg)

    return df.sort_values('Date').reset_index(drop=True)


def add_features(df):
    df = df.copy()
    df['Month']      = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter']    = df['Date'].dt.quarter

    df['Lag_1'] = df['Demand'].shift(1)
    df['Lag_2'] = df['Demand'].shift(2)
    df['Lag_4'] = df['Demand'].shift(4)

    df['Rolling_4']  = df['Demand'].shift(1).rolling(4).mean()
    df['Rolling_12'] = df['Demand'].shift(1).rolling(12).mean()

    if 'Holiday' not in df.columns:
        df['Holiday'] = 0
    else:
        df['Holiday'] = df['Holiday'].fillna(0).astype(int)

    return df.dropna().reset_index(drop=True)


def evaluate(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def forecast_future(df, n_steps):
    """Retrain on the full history and recursively forecast n_steps ahead."""
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(df[FEATURES], df['Demand'])

    median_gap = df['Date'].diff().median()
    if pd.isna(median_gap):
        median_gap = pd.Timedelta(days=7)

    history = df[['Date', 'Demand']].copy()
    predictions = []

    for _ in range(n_steps):
        next_date = history['Date'].iloc[-1] + median_gap
        recent = history['Demand']

        row = {
            'Month'     : next_date.month,
            'WeekOfYear': int(next_date.isocalendar()[1]),
            'Quarter'   : next_date.quarter,
            'Lag_1'     : recent.iloc[-1],
            'Lag_2'     : recent.iloc[-2] if len(recent) >= 2 else recent.iloc[-1],
            'Lag_4'     : recent.iloc[-4] if len(recent) >= 4 else recent.mean(),
            'Rolling_4' : recent.tail(4).mean(),
            'Rolling_12': recent.tail(12).mean(),
            'Holiday'   : 0,
        }
        pred = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        predictions.append({'Date': next_date, 'Forecast': round(pred, 2)})
        history = pd.concat(
            [history, pd.DataFrame([{'Date': next_date, 'Demand': pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


def plot_results(train, test, y_pred, forecast_df, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    ax = axes[0]
    ax.plot(train['Date'].tail(40), train['Demand'].tail(40),
            color='steelblue', linewidth=1.5, label='History')
    ax.plot(test['Date'], test['Demand'].values,
            color='steelblue', linewidth=2, marker='o', label='Actual')
    ax.plot(test['Date'], y_pred,
            color='crimson', linewidth=2, linestyle='--', marker='s', label='Predicted')
    ax.set_title('Model evaluation on test period')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    full = pd.concat([train, test])
    ax.plot(full['Date'].tail(40), full['Demand'].tail(40),
            color='steelblue', linewidth=1.5, label='Recent history')
    ax.plot(forecast_df['Date'], forecast_df['Forecast'],
            color='crimson', linewidth=2, linestyle='--', marker='o', label='Forecast')
    ax.set_title('Future demand forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Saved plot to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Demand forecasting with Random Forest.')
    parser.add_argument('--csv',         default='data/Walmart_Sales.csv', help='Path to input CSV')
    parser.add_argument('--date-col',    default='Date',         help='Name of the date column')
    parser.add_argument('--demand-col',  default='Weekly_Sales', help='Name of the demand/sales column')
    parser.add_argument('--product-col', default='Store',        help='Name of the product/store column (optional)')
    parser.add_argument('--holiday-col', default='Holiday_Flag', help='Name of the holiday flag column (optional)')
    parser.add_argument('--product',     default='1',            help='Specific product/store to model (or "all")')
    parser.add_argument('--test-size',   type=int, default=10,   help='How many recent rows to hold out')
    parser.add_argument('--forecast',    type=int, default=10,   help='How many future periods to predict')
    parser.add_argument('--output-dir',  default='outputs',      help='Where to save predictions and plots')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = None if args.product.lower() == 'all' else args.product

    print(f'Loading {args.csv}')
    raw = load_data(
        args.csv,
        date_col=args.date_col, demand_col=args.demand_col,
        product_col=args.product_col, holiday_col=args.holiday_col,
        selected_product=selected,
    )
    print(f'Loaded {len(raw)} rows, {raw["Date"].min().date()} to {raw["Date"].max().date()}')

    df = add_features(raw)
    if len(df) <= args.test_size + 5:
        raise ValueError(
            f'Not enough rows after feature engineering ({len(df)}). '
            f'Need more than {args.test_size + 5}.'
        )

    train = df.iloc[:-args.test_size]
    test  = df.iloc[-args.test_size:]
    print(f'Training on {len(train)} rows, testing on {len(test)}')

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURES], train['Demand'])

    y_pred = model.predict(test[FEATURES])
    metrics = evaluate(test['Demand'].values, y_pred)
    print('\nTest metrics:')
    for name, value in metrics.items():
        if name == 'MAPE':
            print(f'  {name:5s}: {value:.2f}%')
        elif name == 'R2':
            print(f'  {name:5s}: {value:.3f}')
        else:
            print(f'  {name:5s}: {value:,.2f}')

    predictions_df = test[['Date', 'Demand']].copy()
    predictions_df['Predicted'] = y_pred.round(2)
    predictions_df['Error'] = (predictions_df['Demand'] - predictions_df['Predicted']).round(2)
    predictions_df.to_csv(out_dir / 'test_predictions.csv', index=False)

    print(f'\nForecasting {args.forecast} periods ahead')
    forecast_df = forecast_future(df, args.forecast)
    forecast_df.to_csv(out_dir / 'forecast.csv', index=False)

    plot_results(train, test, y_pred, forecast_df, out_dir / 'results.png')

    print(f'\nResults written to {out_dir}/')


if __name__ == '__main__':
    main()