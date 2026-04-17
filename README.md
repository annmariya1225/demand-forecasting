# Demand Forecasting in Supply Chain Management

**Course:** Industrial Management (IM41081)  
**Author:** Ann Mariya P Alappat (24ME10054)  
**Demo video:** [YouTube link](https://youtu.be/bKwC9GOyfqU)

---

A small machine learning project that forecasts product demand from historical sales data. Built as a practical example of how ML fits into supply chain decisions like inventory planning and replenishment.

The model is a Random Forest regressor trained on lag features, rolling averages, and calendar signals (month, week of year, holidays). It produces both a backtested evaluation on a held-out test set and a forward-looking forecast.

## What's in the repo

```
demand-forecasting/
├── data/
│   └── Walmart_Sales.csv          # sample weekly sales dataset
├── demand_forecasting.ipynb       # notebook walkthrough
├── forecast.py                    # same pipeline as a CLI script
├── requirements.txt
└── README.md
```

## Installation

Clone the repo and install the dependencies. Using a virtual environment is recommended.

```bash
git clone https://github.com/annmariya1225/demand-forecasting.git
cd demand-forecasting

python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Notebook

Open `demand_forecasting.ipynb` in Jupyter and run the cells top to bottom.

```bash
jupyter notebook demand_forecasting.ipynb
```

The second cell (`Configuration`) is where you change the input file, pick a product, or adjust the test size and forecast horizon.

### Script

Run it from the command line. With no arguments it uses the bundled Walmart dataset and forecasts Store 1.

```bash
python forecast.py
```

To aggregate across every store instead of picking one:

```bash
python forecast.py --product all
```

All outputs (predictions and plot) are saved to `outputs/` by default.

## Using your own data


To use your own CSV:

```bash
python forecast.py \
    --csv path/to/your.csv \
    --date-col Date \
    --demand-col Sales \
    --product-col Product \
    --product all \
    --test-size 10 \
    --forecast 10
```

Your CSV needs at least two columns:

- **A date column** : most common formats work; the loader tries day-first parsing by default.
- **A numeric demand column** : units sold, revenue, orders, anything measurable.

Optional:

- **A product/category column** (pick one via `--product`, or aggregate with `all`).
- **A holiday flag** (0/1).

Pass the column names through the `--date-col`, `--demand-col`, `--product-col`, and `--holiday-col` arguments, or edit the `Configuration` cell in the notebook.

## The sample dataset

`data/Walmart_Sales.csv` is the public Walmart weekly sales dataset covering 45 stores from February 2010 to October 2012. It's a good fit for this kind of demo because it's weekly (so the data is clean and seasonal patterns are visible), it has a holiday flag, and there's enough variety across stores to show how the model behaves on different demand profiles.

## How it works

1. **Load and clean** : parse dates, drop bad rows, filter or aggregate by product.
2. **Feature engineering** : derive calendar features, lag values, and rolling averages from the demand column.
3. **Train/test split** : hold out the most recent `TEST_SIZE` rows (chronological, never random - random splits leak future information into training).
4. **Train a Random Forest** on everything before the split.
5. **Evaluate** on the held-out period using MAE, RMSE, MAPE, and R².
6. **Forecast** : retrain on the full history and recursively predict the next `FORECAST_STEPS` periods, feeding each prediction back as the lag input for the next step.

### Why Random Forest?

It handles non-linear relationships, doesn't need feature scaling, is reasonably robust to outliers, and gives feature importances out of the box. For a teaching-oriented project with modest data sizes, it strikes a good balance between accuracy and being easy to reason about.

## Sample results

Running the default configuration (Walmart Store 1, 10-week test, 10-week forecast):

```
MAE  : 50,885.87
RMSE : 62,392.09
MAPE : 3.28%
R^2  : 0.253
```

A MAPE of roughly 3% is reasonable for this dataset given the holiday effects in the test window. Aggregating across all 45 stores tightens it further to around 2.4%.
