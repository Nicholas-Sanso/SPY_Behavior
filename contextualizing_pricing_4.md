```python
import os
import pandas as pd
import requests
import time
import json
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
import seaborn as sns 

from sklearn.cross_decomposition import PLSRegression
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from scipy.stats import gaussian_kde 
```


```python
data_folder = os.path.join(os.path.expanduser("~/Desktop/Trading"), "Data")
os.makedirs(data_folder, exist_ok=True)
tickers_csv_file = os.path.join(data_folder, "sp500_tickers.csv")


# THIS pulls sector and subsector info either localy if it's cached or from the api

API_KEY = "YwnbHRjcJvf6Md2OPoKbSRGHlzZ7hjR6"

# --- LOAD FROM CACHE OR FETCH ---
if os.path.exists(tickers_csv_file):
    print("Loading tickers from CSV cache...")
    df_sp500 = pd.read_csv(tickers_csv_file)
else:
    print("Fetching tickers from API...")
    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={API_KEY}"
    df_sp500 = pd.DataFrame(requests.get(url).json())

    # Save to CSV
    df_sp500.to_csv(tickers_csv_file, index=False)
    print(f"Saved {len(df_sp500)} tickers to CSV cache.")

    
# --- PREVIEW ---
print(df_sp500.shape)
print(df_sp500.columns)
```

    Loading tickers from CSV cache...
    (503, 8)
    Index(['symbol', 'name', 'sector', 'subSector', 'headQuarter',
           'dateFirstAdded', 'cik', 'founded'],
          dtype='object')



```python
tickers = df_sp500["symbol"].dropna().unique().tolist()
```


```python
API_KEY = "YwnbHRjcJvf6Md2OPoKbSRGHlzZ7hjR6"
data_folder = os.path.join(os.path.expanduser("~/Desktop/Trading"), "Data")
os.makedirs(data_folder, exist_ok=True)
tickers_csv_file = os.path.join(data_folder, "sp500_tickers.csv")

# Load tickers
df_sp500 = pd.read_csv(tickers_csv_file)
tickers = df_sp500["symbol"].dropna().unique().tolist()

# Output file
output_file = os.path.join(data_folder, "price_and_earnings.json")

def fetch_price_and_earnings(tickers, output_file):
    if os.path.exists(output_file):
        print(f"Loading from cache: {output_file}")
        return pd.DataFrame(json.load(open(output_file)))


def fetch_price_and_earnings(tickers, output_file):
    if os.path.exists(output_file):
        print(f"Loading from cache: {output_file}")
        return pd.DataFrame(json.load(open(output_file)))

    records = []
    for ticker in tickers:
        try:
            # Get current price
            quote_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={API_KEY}"
            price_data = requests.get(quote_url).json()
            if not price_data:
                continue
            price = price_data[0]["price"]
            price_date = price_data[0].get("date")  # trading date

            # Get latest annual income statement
            income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=1&apikey={API_KEY}"
            income_data = requests.get(income_url).json()
            if not income_data:
                continue
            eps = income_data[0].get("eps")
            net_income = income_data[0].get("netIncome")
            report_date = income_data[0].get("date")  # fiscal period end date

            records.append({
                "symbol": ticker,
                "price": price,
                "price_date": price_date,
                "eps": eps,
                "netIncome": net_income,
                "date": report_date
            })

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        time.sleep(0.2)  # polite rate limit

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {output_file}")

    return pd.DataFrame(records)


price_earnings_df = fetch_price_and_earnings(tickers, output_file)
print(price_earnings_df.columns)
print(price_earnings_df.shape)
```

    Loading from cache: /Users/nicholassanso/Desktop/Trading/Data/price_and_earnings.json
    Index(['symbol', 'price', 'price_date', 'eps', 'netIncome', 'date'], dtype='object')
    (503, 6)



```python
#CLEANS AND TAKES LOG OF THE PE DATA

# Only keep rows with positive EPS
pe_data = price_earnings_df[price_earnings_df["eps"] > 0].copy()

# Compute log(PE)
pe_data["log_PE"] = np.log(pe_data["price"] / pe_data["eps"])

# Print row count for reference
print(pe_data.shape)
```

    (480, 7)


# break


```python
# FETCHES INCOME STATEMENT AND BS STATEMENT AND CF STATEMENT
def fetch_statement(endpoint, tickers, period, limit, data_folder):
    """Fetch statements with unique JSON filename based on endpoint, period, limit."""
    output_file = os.path.join(
        data_folder,
        f"{endpoint}_{period}_limit{limit}.json"
    )

    if os.path.exists(output_file):
        print(f"Loading from cache: {output_file}")
        with open(output_file, "r") as f:
            return json.load(f)

    records = []
    for ticker in tickers:
        url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period={period}&limit={limit}&apikey={API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data:
                for row in data:
                    row["symbol"] = ticker
                records.extend(data)
        except Exception as e:
            print(f"Error fetching {ticker} ({endpoint}): {e}")
        time.sleep(.2)  # API polite rate limit

    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {output_file}")
    # "RECORDS HERE IS A LIST OF STRINGS NOT A DF"
    return records

# AT THIS POINT INCOME_DATA_2_YEARS IS STILL A LIST OF STRINGS STORED AS A VAR
income_data_2_years   = fetch_statement("income-statement", tickers, "annual", 2, data_folder)
balance_data_2_years  = fetch_statement("balance-sheet-statement", tickers, "annual", 2, data_folder)
cashflow_data_2_years = fetch_statement("cash-flow-statement", tickers, "annual", 2, data_folder)

# INCOME_DATA_2_YEARS IS CONVERTED TO A DF
income_data_2_years   = pd.DataFrame(income_data_2_years)
balance_data_2_years  = pd.DataFrame(balance_data_2_years)
cashflow_data_2_years = pd.DataFrame(cashflow_data_2_years)

print("Income shape:", income_data_2_years.shape)
print("Balance shape:", balance_data_2_years.shape)
print("Cash flow shape:", cashflow_data_2_years.shape)
```

    Loading from cache: /Users/nicholassanso/Desktop/Trading/Data/income-statement_annual_limit2.json
    Loading from cache: /Users/nicholassanso/Desktop/Trading/Data/balance-sheet-statement_annual_limit2.json
    Loading from cache: /Users/nicholassanso/Desktop/Trading/Data/cash-flow-statement_annual_limit2.json
    Income shape: (1006, 38)
    Balance shape: (1006, 54)
    Cash flow shape: (1006, 40)



```python
def count_problematic_entries(df, label):
    print(f"\n🔍 Problematic value counts for {label} statement:")
    def count_issues(col):
        numeric_col = pd.to_numeric(col, errors='coerce')
        return ((numeric_col.isna()) | (numeric_col == 0)).sum()

    issue_counts = df.apply(count_issues)
    sorted_counts = issue_counts.sort_values()
    print(sorted_counts)  
```


```python
count_problematic_entries(income_data_2_years, "Income")
count_problematic_entries(balance_data_2_years, "Balance Sheet")
count_problematic_entries(cashflow_data_2_years, "Cash Flow")
```

    
    🔍 Problematic value counts for Income statement:
    costAndExpenses                               0
    ebitda                                        0
    ebitdaratio                                   0
    incomeBeforeTaxRatio                          0
    operatingIncome                               0
    operatingIncomeRatio                          0
    grossProfitRatio                              0
    grossProfit                                   0
    netIncome                                     0
    incomeBeforeTax                               0
    netIncomeRatio                                0
    calendarYear                                  0
    eps                                           0
    epsdiluted                                    0
    cik                                           0
    weightedAverageShsOut                         0
    weightedAverageShsOutDil                      0
    revenue                                       0
    operatingExpenses                             8
    depreciationAndAmortization                   9
    incomeTaxExpense                             12
    costOfRevenue                                20
    interestExpense                              63
    totalOtherIncomeExpensesNet                  65
    sellingGeneralAndAdministrativeExpenses      79
    otherExpenses                               330
    interestIncome                              352
    generalAndAdministrativeExpenses            548
    researchAndDevelopmentExpenses              605
    sellingAndMarketingExpenses                 727
    date                                       1006
    period                                     1006
    acceptedDate                               1006
    fillingDate                                1006
    reportedCurrency                           1006
    symbol                                     1006
    link                                       1006
    finalLink                                  1006
    dtype: int64
    
    🔍 Problematic value counts for Balance Sheet statement:
    totalEquity                                   0
    netDebt                                       0
    totalLiabilities                              0
    cik                                           0
    totalLiabilitiesAndStockholdersEquity         0
    totalLiabilitiesAndTotalEquity                0
    calendarYear                                  0
    totalAssets                                   0
    cashAndCashEquivalents                        0
    totalStockholdersEquity                       0
    cashAndShortTermInvestments                   0
    totalNonCurrentAssets                         2
    totalNonCurrentLiabilities                    4
    totalCurrentAssets                            5
    totalDebt                                     5
    otherNonCurrentAssets                         5
    longTermDebt                                  8
    totalCurrentLiabilities                      12
    retainedEarnings                             15
    otherNonCurrentLiabilities                   15
    netReceivables                               20
    propertyPlantEquipmentNet                    25
    otherCurrentLiabilities                      28
    accumulatedOtherComprehensiveIncomeLoss      32
    accountPayables                              62
    commonStock                                  70
    otherCurrentAssets                           74
    goodwillAndIntangibleAssets                  79
    shortTermDebt                                96
    othertotalStockholdersEquity                 97
    goodwill                                    106
    intangibleAssets                            184
    capitalLeaseObligations                     217
    totalInvestments                            232
    inventory                                   327
    longTermInvestments                         340
    deferredTaxLiabilitiesNonCurrent            349
    taxPayables                                 423
    minorityInterest                            440
    deferredRevenue                             479
    taxAssets                                   526
    shortTermInvestments                        581
    deferredRevenueNonCurrent                   754
    preferredStock                              898
    otherAssets                                 954
    otherLiabilities                            982
    date                                       1006
    period                                     1006
    acceptedDate                               1006
    fillingDate                                1006
    reportedCurrency                           1006
    symbol                                     1006
    link                                       1006
    finalLink                                  1006
    dtype: int64
    
    🔍 Problematic value counts for Cash Flow statement:
    netIncome                                      0
    freeCashFlow                                   0
    netCashUsedProvidedByFinancingActivities       0
    cik                                            0
    operatingCashFlow                              0
    calendarYear                                   0
    cashAtEndOfPeriod                              0
    netCashProvidedByOperatingActivities           0
    netCashUsedForInvestingActivites               0
    cashAtBeginningOfPeriod                        1
    netChangeInCash                                1
    changeInWorkingCapital                         4
    otherNonCashItems                             12
    depreciationAndAmortization                   14
    otherWorkingCapital                           17
    otherFinancingActivites                       45
    capitalExpenditure                            48
    investmentsInPropertyPlantAndEquipment        71
    otherInvestingActivites                       90
    debtRepayment                                 98
    accountsReceivables                          134
    deferredIncomeTax                            158
    accountsPayables                             173
    stockBasedCompensation                       182
    commonStockRepurchased                       197
    dividendsPaid                                202
    effectOfForexChangesOnCash                   276
    acquisitionsNet                              357
    inventory                                    364
    purchasesOfInvestments                       401
    salesMaturitiesOfInvestments                 401
    commonStockIssued                            625
    date                                        1006
    period                                      1006
    acceptedDate                                1006
    fillingDate                                 1006
    reportedCurrency                            1006
    symbol                                      1006
    link                                        1006
    finalLink                                   1006
    dtype: int64



```python
def sort_by_symbol_then_date(df):
    # Sort ascending by symbol, then by date (oldest first)
    return df.sort_values(["symbol", "date"], ascending=[True, True]).reset_index(drop=True)


income_sorted = sort_by_symbol_then_date(income_data_2_years)
balance_sorted = sort_by_symbol_then_date(balance_data_2_years)
cashflow_sorted = sort_by_symbol_then_date(cashflow_data_2_years)
pe_data_sorted = sort_by_symbol_then_date(pe_data)


income_sorted.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>reportedCurrency</th>
      <th>cik</th>
      <th>fillingDate</th>
      <th>acceptedDate</th>
      <th>calendarYear</th>
      <th>period</th>
      <th>revenue</th>
      <th>costOfRevenue</th>
      <th>...</th>
      <th>incomeBeforeTaxRatio</th>
      <th>incomeTaxExpense</th>
      <th>netIncome</th>
      <th>netIncomeRatio</th>
      <th>eps</th>
      <th>epsdiluted</th>
      <th>weightedAverageShsOut</th>
      <th>weightedAverageShsOutDil</th>
      <th>link</th>
      <th>finalLink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-31</td>
      <td>A</td>
      <td>USD</td>
      <td>0001090872</td>
      <td>2023-12-20</td>
      <td>2023-12-19 18:52:16</td>
      <td>2023</td>
      <td>FY</td>
      <td>6833000000</td>
      <td>3368000000</td>
      <td>...</td>
      <td>0.195961</td>
      <td>99000000</td>
      <td>1240000000</td>
      <td>0.181472</td>
      <td>4.22</td>
      <td>4.19</td>
      <td>294000000</td>
      <td>296000000</td>
      <td>https://www.sec.gov/Archives/edgar/data/109087...</td>
      <td>https://www.sec.gov/Archives/edgar/data/109087...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-10-31</td>
      <td>A</td>
      <td>USD</td>
      <td>0001090872</td>
      <td>2024-12-20</td>
      <td>2024-12-19 18:51:56</td>
      <td>2024</td>
      <td>FY</td>
      <td>6510000000</td>
      <td>2975000000</td>
      <td>...</td>
      <td>0.233641</td>
      <td>232000000</td>
      <td>1289000000</td>
      <td>0.198003</td>
      <td>4.44</td>
      <td>4.43</td>
      <td>290000000</td>
      <td>291000000</td>
      <td>https://www.sec.gov/Archives/edgar/data/109087...</td>
      <td>https://www.sec.gov/Archives/edgar/data/109087...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-30</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>0000320193</td>
      <td>2023-11-03</td>
      <td>2023-11-02 18:08:27</td>
      <td>2023</td>
      <td>FY</td>
      <td>383285000000</td>
      <td>214137000000</td>
      <td>...</td>
      <td>0.296740</td>
      <td>16741000000</td>
      <td>96995000000</td>
      <td>0.253062</td>
      <td>6.16</td>
      <td>6.13</td>
      <td>15744231000</td>
      <td>15812547000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-09-28</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>0000320193</td>
      <td>2024-11-01</td>
      <td>2024-11-01 06:01:36</td>
      <td>2024</td>
      <td>FY</td>
      <td>391035000000</td>
      <td>210352000000</td>
      <td>...</td>
      <td>0.315790</td>
      <td>29749000000</td>
      <td>93736000000</td>
      <td>0.239713</td>
      <td>6.11</td>
      <td>6.08</td>
      <td>15343783000</td>
      <td>15408095000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-12-31</td>
      <td>ABBV</td>
      <td>USD</td>
      <td>0001551152</td>
      <td>2024-02-20</td>
      <td>2024-02-20 12:45:17</td>
      <td>2023</td>
      <td>FY</td>
      <td>54318000000</td>
      <td>20415000000</td>
      <td>...</td>
      <td>0.115063</td>
      <td>1377000000</td>
      <td>4863000000</td>
      <td>0.089528</td>
      <td>2.73</td>
      <td>2.72</td>
      <td>1768000000</td>
      <td>1773000000</td>
      <td>https://www.sec.gov/Archives/edgar/data/155115...</td>
      <td>https://www.sec.gov/Archives/edgar/data/155115...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-12-31</td>
      <td>ABBV</td>
      <td>USD</td>
      <td>0001551152</td>
      <td>2025-02-14</td>
      <td>2025-02-14 13:17:23</td>
      <td>2024</td>
      <td>FY</td>
      <td>56334000000</td>
      <td>16904000000</td>
      <td>...</td>
      <td>0.065964</td>
      <td>-570000000</td>
      <td>4278000000</td>
      <td>0.075940</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>1769000000</td>
      <td>1773000000</td>
      <td>https://www.sec.gov/Archives/edgar/data/155115...</td>
      <td>https://www.sec.gov/Archives/edgar/data/155115...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-12-31</td>
      <td>ABNB</td>
      <td>USD</td>
      <td>0001559720</td>
      <td>2024-02-16</td>
      <td>2024-02-16 16:02:16</td>
      <td>2023</td>
      <td>FY</td>
      <td>9917000000</td>
      <td>1703000000</td>
      <td>...</td>
      <td>0.211959</td>
      <td>-2690000000</td>
      <td>4792000000</td>
      <td>0.483211</td>
      <td>7.52</td>
      <td>7.24</td>
      <td>637000000</td>
      <td>662000000</td>
      <td>https://www.sec.gov/Archives/edgar/data/155972...</td>
      <td>https://www.sec.gov/Archives/edgar/data/155972...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-12-31</td>
      <td>ABNB</td>
      <td>USD</td>
      <td>0001559720</td>
      <td>2025-02-13</td>
      <td>2025-02-13 16:04:28</td>
      <td>2024</td>
      <td>FY</td>
      <td>11102000000</td>
      <td>1878000000</td>
      <td>...</td>
      <td>0.300036</td>
      <td>683000000</td>
      <td>2648000000</td>
      <td>0.238516</td>
      <td>4.19</td>
      <td>4.11</td>
      <td>632000000</td>
      <td>645000000</td>
      <td>https://www.sec.gov/Archives/edgar/data/155972...</td>
      <td>https://www.sec.gov/Archives/edgar/data/155972...</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>




```python
#DOUBLE CHECK THE ECISION TO ADD A CONSTANT... ELONGATING THE TAIL OF THE DISTRIBUTION

def compute_log_change(df, constant=1e-3, drop_first=True):
    """
    Compute log-differences for year-over-year growth of financial statement items.
    Keeps 'symbol' and 'date' columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must include 'symbol' and 'date' columns, sorted by both.
    constant : float
        Small stabilizing constant for log transform.
    drop_first : bool
        Whether to drop the first row per symbol (NaN after diff).
    """
    df = df.copy()
    # Numeric part
    num_df = df.select_dtypes(include=[np.number])
    log_df = np.log(np.abs(num_df) + constant)
    log_diff = log_df.diff()

    # Rename to show log-change
    log_diff.columns = [f"{col}_logchg" for col in log_diff.columns]

    # Combine with non-numeric columns
    result = pd.concat([df[["symbol", "date"]], log_diff], axis=1)

    if drop_first:
        # Drop the first row per symbol (which has NaN diffs)
        result = result.groupby("symbol", group_keys=False).apply(lambda g: g.iloc[1:])

    return result.reset_index(drop=True)


# --- Apply grouped by symbol ---
income_log_change = (
    income_sorted.groupby("symbol", group_keys=False)
    .apply(lambda g: compute_log_change(g))
)

balance_log_change = (
    balance_sorted.groupby("symbol", group_keys=False)
    .apply(lambda g: compute_log_change(g))
)

cashflow_log_change = (
    cashflow_sorted.groupby("symbol", group_keys=False)
    .apply(lambda g: compute_log_change(g))
)

```


```python
income_log_change.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>date</th>
      <th>revenue_logchg</th>
      <th>costOfRevenue_logchg</th>
      <th>grossProfit_logchg</th>
      <th>grossProfitRatio_logchg</th>
      <th>researchAndDevelopmentExpenses_logchg</th>
      <th>generalAndAdministrativeExpenses_logchg</th>
      <th>sellingAndMarketingExpenses_logchg</th>
      <th>sellingGeneralAndAdministrativeExpenses_logchg</th>
      <th>...</th>
      <th>totalOtherIncomeExpensesNet_logchg</th>
      <th>incomeBeforeTax_logchg</th>
      <th>incomeBeforeTaxRatio_logchg</th>
      <th>incomeTaxExpense_logchg</th>
      <th>netIncome_logchg</th>
      <th>netIncomeRatio_logchg</th>
      <th>eps_logchg</th>
      <th>epsdiluted_logchg</th>
      <th>weightedAverageShsOut_logchg</th>
      <th>weightedAverageShsOutDil_logchg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2024-10-31</td>
      <td>-0.048424</td>
      <td>-0.124075</td>
      <td>0.020001</td>
      <td>0.068295</td>
      <td>-0.004167</td>
      <td>28.015603</td>
      <td>24.615086</td>
      <td>-0.073637</td>
      <td>...</td>
      <td>1.098612</td>
      <td>0.127445</td>
      <td>0.175050</td>
      <td>0.851618</td>
      <td>0.038755</td>
      <td>0.086722</td>
      <td>0.050808</td>
      <td>0.055686</td>
      <td>-0.013699</td>
      <td>-0.017036</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AAPL</td>
      <td>2024-09-28</td>
      <td>0.020018</td>
      <td>-0.017834</td>
      <td>0.065970</td>
      <td>0.045850</td>
      <td>0.047492</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045668</td>
      <td>...</td>
      <td>-0.742114</td>
      <td>0.082240</td>
      <td>0.062019</td>
      <td>0.574935</td>
      <td>-0.034177</td>
      <td>-0.053976</td>
      <td>-0.008149</td>
      <td>-0.008189</td>
      <td>-0.025764</td>
      <td>-0.025911</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ABBV</td>
      <td>2024-12-31</td>
      <td>0.036443</td>
      <td>-0.188720</td>
      <td>0.151023</td>
      <td>0.114408</td>
      <td>0.510774</td>
      <td>0.035336</td>
      <td>-0.046520</td>
      <td>0.024329</td>
      <td>...</td>
      <td>-0.182598</td>
      <td>-0.519934</td>
      <td>-0.549983</td>
      <td>-0.882026</td>
      <td>-0.128170</td>
      <td>-0.162638</td>
      <td>-0.128783</td>
      <td>-0.129288</td>
      <td>0.000565</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ABNB</td>
      <td>2024-12-31</td>
      <td>0.112875</td>
      <td>0.097816</td>
      <td>0.115969</td>
      <td>0.003090</td>
      <td>0.177276</td>
      <td>-0.535827</td>
      <td>0.197520</td>
      <td>-0.127965</td>
      <td>...</td>
      <td>0.286826</td>
      <td>0.460383</td>
      <td>0.346129</td>
      <td>-1.370802</td>
      <td>-0.593143</td>
      <td>-0.703902</td>
      <td>-0.584760</td>
      <td>-0.566093</td>
      <td>-0.007880</td>
      <td>-0.026015</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ABT</td>
      <td>2024-12-31</td>
      <td>0.044878</td>
      <td>0.027178</td>
      <td>0.062300</td>
      <td>0.017387</td>
      <td>0.037536</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.068190</td>
      <td>...</td>
      <td>-0.197280</td>
      <td>0.051046</td>
      <td>0.006131</td>
      <td>1.915390</td>
      <td>0.850911</td>
      <td>0.802174</td>
      <td>0.843222</td>
      <td>0.848433</td>
      <td>-0.001372</td>
      <td>-0.000572</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ACGL</td>
      <td>2024-12-31</td>
      <td>0.241774</td>
      <td>0.250392</td>
      <td>0.226011</td>
      <td>-0.015719</td>
      <td>0.000000</td>
      <td>0.154151</td>
      <td>0.000000</td>
      <td>0.154151</td>
      <td>...</td>
      <td>0.943262</td>
      <td>0.322661</td>
      <td>0.080583</td>
      <td>-0.880291</td>
      <td>-0.029928</td>
      <td>-0.270771</td>
      <td>-0.040156</td>
      <td>-0.037704</td>
      <td>0.010254</td>
      <td>0.007889</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ACN</td>
      <td>2025-08-31</td>
      <td>0.071019</td>
      <td>0.081285</td>
      <td>0.049464</td>
      <td>-0.021489</td>
      <td>0.000000</td>
      <td>0.016138</td>
      <td>0.028329</td>
      <td>0.023656</td>
      <td>...</td>
      <td>-0.838718</td>
      <td>0.057209</td>
      <td>-0.013718</td>
      <td>0.066944</td>
      <td>0.055377</td>
      <td>-0.015503</td>
      <td>0.060365</td>
      <td>0.060208</td>
      <td>-0.004728</td>
      <td>-0.005527</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ADBE</td>
      <td>2024-11-29</td>
      <td>0.102549</td>
      <td>0.001698</td>
      <td>0.115703</td>
      <td>0.013139</td>
      <td>0.127177</td>
      <td>0.078899</td>
      <td>0.074348</td>
      <td>0.075301</td>
      <td>...</td>
      <td>0.243078</td>
      <td>0.019229</td>
      <td>-0.083073</td>
      <td>0.000000</td>
      <td>0.024027</td>
      <td>-0.078230</td>
      <td>0.046057</td>
      <td>0.043823</td>
      <td>-0.021901</td>
      <td>-0.019803</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ADI</td>
      <td>2024-11-02</td>
      <td>-0.266455</td>
      <td>-0.090338</td>
      <td>-0.381037</td>
      <td>-0.114393</td>
      <td>-0.109594</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.173051</td>
      <td>...</td>
      <td>0.171913</td>
      <td>-0.708037</td>
      <td>-0.439696</td>
      <td>-0.725320</td>
      <td>-0.706521</td>
      <td>-0.438023</td>
      <td>-0.692996</td>
      <td>-0.691469</td>
      <td>-0.012152</td>
      <td>-0.014457</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ADM</td>
      <td>2024-12-31</td>
      <td>-0.093736</td>
      <td>-0.080320</td>
      <td>-0.262577</td>
      <td>-0.166573</td>
      <td>26.317977</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.069841</td>
      <td>...</td>
      <td>-0.258574</td>
      <td>-0.644069</td>
      <td>-0.534745</td>
      <td>-0.553595</td>
      <td>-0.660107</td>
      <td>-0.546561</td>
      <td>-0.564947</td>
      <td>-0.566129</td>
      <td>-0.123809</td>
      <td>-0.094757</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ADP</td>
      <td>2025-06-30</td>
      <td>0.068346</td>
      <td>0.050384</td>
      <td>0.086029</td>
      <td>0.017648</td>
      <td>0.033846</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.092995</td>
      <td>...</td>
      <td>0.364565</td>
      <td>0.086045</td>
      <td>0.017630</td>
      <td>0.093743</td>
      <td>0.083734</td>
      <td>0.015311</td>
      <td>0.092154</td>
      <td>0.092035</td>
      <td>-0.008561</td>
      <td>-0.008527</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ADSK</td>
      <td>2025-01-31</td>
      <td>0.119579</td>
      <td>0.044216</td>
      <td>0.127760</td>
      <td>0.008172</td>
      <td>0.079145</td>
      <td>27.200238</td>
      <td>28.324168</td>
      <td>0.098258</td>
      <td>...</td>
      <td>0.265703</td>
      <td>0.197465</td>
      <td>0.077529</td>
      <td>0.167723</td>
      <td>0.204876</td>
      <td>0.084809</td>
      <td>0.210127</td>
      <td>0.200410</td>
      <td>-0.004640</td>
      <td>0.004619</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AEE</td>
      <td>2024-12-31</td>
      <td>0.016267</td>
      <td>-0.015744</td>
      <td>0.052259</td>
      <td>0.035916</td>
      <td>0.000000</td>
      <td>-26.410241</td>
      <td>0.000000</td>
      <td>-26.410241</td>
      <td>...</td>
      <td>0.120836</td>
      <td>-0.053653</td>
      <td>-0.069517</td>
      <td>-0.790646</td>
      <td>0.025708</td>
      <td>0.009381</td>
      <td>0.011348</td>
      <td>0.009089</td>
      <td>0.015106</td>
      <td>0.015072</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AEP</td>
      <td>2024-12-31</td>
      <td>0.027230</td>
      <td>-0.014460</td>
      <td>0.122425</td>
      <td>0.094883</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.022174</td>
      <td>0.258707</td>
      <td>0.229724</td>
      <td>-0.331357</td>
      <td>0.295453</td>
      <td>0.266174</td>
      <td>0.273441</td>
      <td>0.274569</td>
      <td>0.021334</td>
      <td>0.021172</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AES</td>
      <td>2024-12-31</td>
      <td>-0.031901</td>
      <td>-0.019873</td>
      <td>-0.082431</td>
      <td>-0.050270</td>
      <td>0.000000</td>
      <td>0.121697</td>
      <td>0.000000</td>
      <td>0.121697</td>
      <td>...</td>
      <td>-0.634568</td>
      <td>2.489526</td>
      <td>2.373307</td>
      <td>-1.486983</td>
      <td>1.941176</td>
      <td>1.929279</td>
      <td>1.859074</td>
      <td>1.906054</td>
      <td>0.053069</td>
      <td>0.001404</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 30 columns</p>
</div>




```python
def count_zeros_nans_logchg(df):
    # Keep only numeric columns that end with "_logchg"
    numeric_cols = [c for c in df.select_dtypes(include=[float, int]).columns if c.endswith("_logchg")]
    
    # Count zeros and NaNs
    zero_counts = (df[numeric_cols] == 0).sum()
    nan_counts = df[numeric_cols].isna().sum()
    
    # Combine into a single DataFrame
    summary = pd.DataFrame({
        "zeros": zero_counts,
        "nans": nan_counts
    }).sort_values(by=["zeros", "nans"], ascending=True)
    
    # Force full display
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(summary)
    
    return summary

# Example usage
print("Income 0/NaN counts per YoY column:")
income_summary = count_zeros_nans_logchg(income_log_change)

print("\nBalance 0/NaN counts per YoY column:")
balance_summary = count_zeros_nans_logchg(balance_log_change)

print("\nCashflow 0/NaN counts per YoY column:")
cashflow_summary = count_zeros_nans_logchg(cashflow_log_change)

# the purpose here is to make it easy to identify which line items (columns) are fully filled out from our sample so that we are only grabbing columns (features)
# that are likely to be filled out by the stock under consideration, cause ultimately after we find a regression that has explanatory power... we can still only apply it 
# to the stock under consideration if it has the same line items filled out 
# the 503 nans is a result of the .pct change method that we used which creates a nan on every other row
```

    Income 0/NaN counts per YoY column:
                                                    zeros  nans
    revenue_logchg                                      0     0
    grossProfit_logchg                                  0     0
    costAndExpenses_logchg                              0     0
    ebitda_logchg                                       0     0
    ebitdaratio_logchg                                  0     0
    operatingIncome_logchg                              0     0
    operatingIncomeRatio_logchg                         0     0
    incomeBeforeTax_logchg                              0     0
    incomeBeforeTaxRatio_logchg                         0     0
    netIncomeRatio_logchg                               0     0
    netIncome_logchg                                    1     0
    eps_logchg                                          1     0
    epsdiluted_logchg                                   1     0
    operatingExpenses_logchg                            4     0
    depreciationAndAmortization_logchg                  5     0
    costOfRevenue_logchg                                7     0
    incomeTaxExpense_logchg                             7     0
    weightedAverageShsOut_logchg                        7     0
    grossProfitRatio_logchg                            10     0
    weightedAverageShsOutDil_logchg                    10     0
    totalOtherIncomeExpensesNet_logchg                 32     0
    sellingGeneralAndAdministrativeExpenses_logchg     34     0
    interestExpense_logchg                             35     0
    otherExpenses_logchg                              137     0
    interestIncome_logchg                             169     0
    generalAndAdministrativeExpenses_logchg           248     0
    researchAndDevelopmentExpenses_logchg             295     0
    sellingAndMarketingExpenses_logchg                349     0
    
    Balance 0/NaN counts per YoY column:
                                                    zeros  nans
    totalAssets_logchg                                  0     0
    totalLiabilities_logchg                             0     0
    totalStockholdersEquity_logchg                      0     0
    totalEquity_logchg                                  0     0
    totalLiabilitiesAndStockholdersEquity_logchg        0     0
    totalLiabilitiesAndTotalEquity_logchg               0     0
    cashAndCashEquivalents_logchg                       1     0
    cashAndShortTermInvestments_logchg                  1     0
    totalNonCurrentAssets_logchg                        1     0
    totalCurrentAssets_logchg                           2     0
    otherNonCurrentAssets_logchg                        2     0
    longTermDebt_logchg                                 2     0
    totalNonCurrentLiabilities_logchg                   2     0
    totalDebt_logchg                                    2     0
    netDebt_logchg                                      2     0
    retainedEarnings_logchg                             3     0
    totalCurrentLiabilities_logchg                      6     0
    otherNonCurrentLiabilities_logchg                   6     0
    netReceivables_logchg                               8     0
    propertyPlantEquipmentNet_logchg                   10     0
    otherCurrentLiabilities_logchg                     10     0
    accumulatedOtherComprehensiveIncomeLoss_logchg     16     0
    accountPayables_logchg                             28     0
    otherCurrentAssets_logchg                          33     0
    shortTermDebt_logchg                               41     0
    othertotalStockholdersEquity_logchg                47     0
    goodwillAndIntangibleAssets_logchg                 60     0
    intangibleAssets_logchg                            91     0
    totalInvestments_logchg                            95     0
    capitalLeaseObligations_logchg                    100     0
    goodwill_logchg                                   125     0
    longTermInvestments_logchg                        150     0
    deferredTaxLiabilitiesNonCurrent_logchg           162     0
    inventory_logchg                                  166     0
    taxPayables_logchg                                172     0
    deferredRevenue_logchg                            229     0
    minorityInterest_logchg                           229     0
    taxAssets_logchg                                  249     0
    shortTermInvestments_logchg                       274     0
    commonStock_logchg                                280     0
    deferredRevenueNonCurrent_logchg                  372     0
    otherAssets_logchg                                469     0
    preferredStock_logchg                             471     0
    otherLiabilities_logchg                           486     0
    
    Cashflow 0/NaN counts per YoY column:
                                                     zeros  nans
    netCashProvidedByOperatingActivities_logchg          0     0
    netCashUsedForInvestingActivites_logchg              0     0
    netCashUsedProvidedByFinancingActivities_logchg      0     0
    cashAtBeginningOfPeriod_logchg                       0     0
    operatingCashFlow_logchg                             0     0
    freeCashFlow_logchg                                  0     0
    netIncome_logchg                                     1     0
    cashAtEndOfPeriod_logchg                             1     0
    netChangeInCash_logchg                               2     0
    changeInWorkingCapital_logchg                        3     0
    otherNonCashItems_logchg                             4     0
    depreciationAndAmortization_logchg                   6     0
    otherWorkingCapital_logchg                           6     0
    otherFinancingActivites_logchg                      12     0
    capitalExpenditure_logchg                           24     0
    investmentsInPropertyPlantAndEquipment_logchg       26     0
    debtRepayment_logchg                                35     0
    otherInvestingActivites_logchg                      36     0
    accountsReceivables_logchg                          55     0
    deferredIncomeTax_logchg                            65     0
    accountsPayables_logchg                             77     0
    commonStockRepurchased_logchg                       77     0
    stockBasedCompensation_logchg                       87     0
    dividendsPaid_logchg                                96     0
    acquisitionsNet_logchg                             116     0
    effectOfForexChangesOnCash_logchg                  137     0
    salesMaturitiesOfInvestments_logchg                162     0
    inventory_logchg                                   164     0
    purchasesOfInvestments_logchg                      171     0
    commonStockIssued_logchg                           282     0



```python
def outlier_check_1(df, title):
    # Flatten all numeric columns into one long vector
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    values = df[numeric_cols].values.flatten()
    values = values[~np.isnan(values)]  # drop NaNs

    # Scatter vs. index, colored by density
    plt.figure(figsize=(10, 6))
    sns.kdeplot(values, fill=True, color="lightblue", alpha=0.3, linewidth=0)  # background density
    plt.scatter(range(len(values)), values, 
                c=values, cmap="viridis", s=5, alpha=0.6)

    plt.title(f"Density Scatterplot: {title}", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Call for each of your cleaned DataFrames
outlier_check_1(income_post_nans, "Income")
outlier_check_1(balance_post_nans, "Balance")
outlier_check_1(cashflow_post_nans, "Cashflow")

#demonstrates the necessity for addressing outlier concerns 
# ONLY PURPOSE OF. THIS IS TO CONFIRM WE HAVE SERIOUS OUTLIERS THAT NEED TO BE ADDRESSED
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[13], line 20
         17     plt.show()
         19 # Call for each of your cleaned DataFrames
    ---> 20 outlier_check_1(income_post_nans, "Income")
         21 outlier_check_1(balance_post_nans, "Balance")
         22 outlier_check_1(cashflow_post_nans, "Cashflow")


    NameError: name 'income_post_nans' is not defined



```python
def OUTLIER_CHECK_2(df, title):
    # Flatten numeric columns into one vector
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    values = df[numeric_cols].values.flatten()
    values = values[~np.isnan(values)]  # remove NaNs

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=200, density=True, alpha=0.6, color="steelblue")
    
    # KDE overlay
    kde = gaussian_kde(values)
    xs = np.linspace(-8, 8, 400)
    plt.plot(xs, kde(xs), color="darkred", lw=2)

    plt.title(f"Distribution: {title}", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(-8, 8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Call for each dataset
OUTLIER_CHECK_2(income_post_nans, "Income")
OUTLIER_CHECK_2(balance_post_nans, "Balance")
OUTLIER_CHECK_2(cashflow_post_nans, "Cashflow")

# ONLY PURPOSE OF. THIS IS TO CONFIRM WE HAVE SERIOUS OUTLIERS THAT NEED TO BE ADDRESSED
```
# Step 1: Keep only common (symbol, date) pairs
for df in [income_post_nans, balance_post_nans, cashflow_post_nans,pe_data]:
    df["symbol_date"] = list(zip(df["symbol"], df["date"]))

```python
def add_symbol_date(df_dict):
    """
    Adds a 'symbol_date' column to each DataFrame in df_dict.
    Prints the columns and shape of each updated DataFrame.
    
    Parameters:
    df_dict : dict
        Dictionary of DataFrames keyed by variable names (strings)
    
    Returns:
    None (updates DataFrames in place)
    """
    for name, df in df_dict.items():
        df["symbol_date"] = list(zip(df["symbol"], df["date"]))
        print(f"{name}: columns =  shape = {df.shape}")


# Usage
dfs = {
    "income_post_nans": income_post_nans,
    "balance_post_nans": balance_post_nans,
    "cashflow_post_nans": cashflow_post_nans,
    "pe_data": pe_data
}

add_symbol_date(dfs)

```


```python
common_pairs = (
    set(income_post_nans["symbol_date"])
    & set(balance_post_nans["symbol_date"])
    & set(cashflow_post_nans["symbol_date"])
    & set(pe_data["symbol_date"])
)

def filter_by_common_pairs(df, common_pairs):
    """
    Keep only rows where the 'symbol_date' is in common_pairs.
    Returns a new DataFrame.
    """
    return df[df["symbol_date"].isin(common_pairs)].copy()

income_post_nans_overlapped = filter_by_common_pairs(income_post_nans, common_pairs)
balance_post_nans_overlapped = filter_by_common_pairs(balance_post_nans, common_pairs)
cashflow_post_nans_overlapped = filter_by_common_pairs(cashflow_post_nans, common_pairs)
pe_post_nans_overlapped = filter_by_common_pairs(pe_data, common_pairs)

print(income_post_nans_overlapped.shape)
print(balance_post_nans_overlapped.shape)
print(cashflow_post_nans_overlapped.shape)
print(pe_post_nans_overlapped.shape)

# we only want to include a ticker if we have all the financial statement items for all dataframes
```


```python
# Step 2: Sort consistently
income_post_nans_overlapped = income_post_nans_overlapped.sort_values(["symbol", "date"]).reset_index(drop=True)
balance_post_nans_overlapped = balance_post_nans_overlapped.sort_values(["symbol", "date"]).reset_index(drop=True)
cashflow_post_nans_overlapped = cashflow_post_nans_overlapped.sort_values(["symbol", "date"]).reset_index(drop=True)
pe_post_nans_overlapped = pe_post_nans_overlapped.sort_values(["symbol", "date"]).reset_index(drop=True)


print(
    "Cleaned and aligned shapes:", 
    income_post_nans_overlapped.shape,   
    balance_post_nans_overlapped.shape, 
    cashflow_post_nans_overlapped.shape,
    pe_post_nans_overlapped.shape,
)
```


```python
def check_alignment(dfs: dict):
    """
    Check alignment of multiple DataFrames on 'symbol' and 'date'.
    Expects each DataFrame to have the same row order and columns: 'symbol', 'date'.
    
    Parameters
    
    None (prints summary of mismatches)
    """
    # Ensure equal lengths
    lengths = {name: len(df) for name, df in dfs.items()}
    if len(set(lengths.values())) > 1:
        print("⚠️ Row counts differ between DataFrames:")
        for name, length in lengths.items():
            print(f"  {name}: {length} rows")
    else:
        print(f"✅ All DataFrames have {list(lengths.values())[0]} rows")

    # Concatenate for comparison
    merged = pd.concat(
        {name: df[["symbol", "date"]].reset_index(drop=True) for name, df in dfs.items()},
        axis=1
    )

    # Compare across DataFrames
    base = list(dfs.keys())[0]  # pick first as reference
    symbol_mismatches = 0
    date_mismatches = 0

    for i in range(len(dfs[base])):
        symbols = [merged[(name, "symbol")][i] for name in dfs.keys()]
        dates   = [merged[(name, "date")][i]   for name in dfs.keys()]
        if len(set(symbols)) > 1:
            symbol_mismatches += 1
        if len(set(dates)) > 1:
            date_mismatches += 1

    print(f"Symbol mismatches: {symbol_mismatches}")
    print(f"Date mismatches:   {date_mismatches}")

```


```python
check_alignment({
    "income": income_post_nans_overlapped,
    "balance": balance_post_nans_overlapped,
    "cashflow": cashflow_post_nans_overlapped,
    "pe": pe_post_nans_overlapped
})
```


```python
def run_univariate_regressions(X, y):

    # --- Drop duplicate columns by name ---
    X = X.loc[:, ~X.columns.duplicated()].copy()

    results = []

    for col in X.columns:
        X_const = sm.add_constant(X[[col]])
        model = sm.OLS(y, X_const).fit()
        coef = model.params[col]
        t_value = model.tvalues[col]
        pval = model.pvalues[col]
        r2 = model.rsquared
        results.append((col, coef, t_value, pval, r2))

    results_df = pd.DataFrame(
        results, columns=["feature", "coef", "t_value", "pval", "r2"]
    )

    # Apply Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        results_df["pval"], alpha=0.1, method="fdr_bh"
    )
    results_df["pval_fdr"] = pvals_corrected
    results_df["reject_null"] = reject

    return results_df

```


```python
def run_statement_univariate(statement_df, log_pe_df, label=""):

    merged = statement_df.merge(log_pe_df, on="symbol", how="inner")
    X = merged.select_dtypes(include=["number"]).drop(columns=["log_PE"])
    y = merged["log_PE"]

    uni_results = run_univariate_regressions(X, y)

    return merged, uni_results

```


```python
# 1. Prepare the dependent variable
log_pe_df = pe_post_nans_overlapped[["symbol", "log_PE"]].copy()

income_merged, income_univariate = run_statement_univariate(income_post_nans_overlapped, log_pe_df, label="income")
balance_merged, balance_univariate = run_statement_univariate(balance_post_nans_overlapped, log_pe_df, label="balance")
cashflow_merged, cashflow_univariate = run_statement_univariate(cashflow_post_nans_overlapped, log_pe_df, label="cashflow")

print(income_merged.shape, balance_merged.shape, cashflow_merged.shape)
print(income_univariate.shape, balance_univariate.shape, cashflow_univariate.shape)
print(income_univariate.columns)
print(income_merged.columns)
```


```python
# Combine for convenience, but we’ll split by statement type
plot_df = pd.concat([
    income_univariate.assign(statement_type='Income'),
    balance_univariate.assign(statement_type='Balance'),
    cashflow_univariate.assign(statement_type='Cashflow')
], ignore_index=True)

statement_order = ['Income', 'Balance', 'Cashflow']
n_statements = len(statement_order)

# Create subplots (one per statement type)
fig, axes = plt.subplots(n_statements, 1, figsize=(12, 5*n_statements), sharex=True)

for i, statement in enumerate(statement_order):
    ax = axes[i]
    
    # Subset for this statement
    df = plot_df[plot_df['statement_type'] == statement].copy()
    
    # Sort by absolute t-value descending
    df = df.sort_values('t_value', ascending=False).reset_index(drop=True)
    
    # Normalize r² for color mapping
    norm = mpl.colors.Normalize(vmin=df['r2'].min(), vmax=df['r2'].max())
    cmap = mpl.cm.viridis
    colors = [cmap(norm(val)) for val in df['r2']]
    
    # Y positions
    y_pos = range(len(df))
    
    # Horizontal bar chart
    ax.barh(y=y_pos, width=df['t_value'], color=colors, edgecolor='black')
    
    # Highlight significant features
    for idx, row in enumerate(df.itertuples()):
        if row.reject_null:
            ax.text(
                x=row.t_value + (0.05 if row.t_value>0 else -0.05),
                y=idx,
                s='*',
                va='center',
                ha='left' if row.t_value>0 else 'right',
                color='red',
                fontsize=12
            )
    
    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'], fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Feature')
    ax.set_title(f'{statement} Statement: t-values by Feature (Color = R²)')

# Shared x-label
axes[-1].set_xlabel('t-value')

# Add a single colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('R²')

plt.tight_layout()
plt.show()

```


```python
def select_significant_features(univariate_results, merged_df, 
                                t_threshold=1.2, r2_threshold=0.005, p_threshold=0.2, 
                                label=""):
    """
    Select significant features from univariate regression results 
    and return the corresponding data from the merged dataframe.
    """
    total_features = len(univariate_results)

    # Filter by thresholds
    signif = univariate_results[
        (univariate_results["t_value"].abs() > t_threshold) &
        (univariate_results["r2"] > r2_threshold) &
        (univariate_results["pval"] < p_threshold)
    ]

    features = signif["feature"].tolist()
    selected = merged_df[["symbol","date","symbol_date", "log_PE"] + features]

    if label:
        n_selected = len(features)
        pct = (n_selected / total_features * 100) if total_features > 0 else 0
        print(f"{label.title()}: {n_selected}/{total_features} features selected ({pct:.1f}%)")

    return signif, selected

```


```python
signif_income, income_selected = select_significant_features(income_univariate, income_merged, label="income")

signif_balance, balance_selected = select_significant_features(balance_univariate, balance_merged, label="balance")

signif_cashflow, cashflow_selected = select_significant_features(cashflow_univariate, cashflow_merged, label="cashflow")


print(income_selected.shape)
print(balance_selected.shape)
print(cashflow_selected.shape)
print(income_selected.columns)
print(balance_selected.columns)
print(cashflow_selected.columns)
```


```python
def run_pca(df, n_components=None, prefix="", columns=None):
    """
    Run PCA on a DataFrame with optional column selection.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    n_components : int or None
        Number of PCA components. If None, use all available features.
    prefix : str
        Prefix for the PCA component column names.
    columns : list of str or None
        Subset of columns to run PCA on. If None, use all numeric columns
        except common ID columns like symbol/date.

    Returns
    -------
    pd.DataFrame
        DataFrame with PCA component columns added.
    PCA
        The fitted PCA object.
    """

    # Drop common ID columns if present
    exclude_cols = ["symbol", "date", "symbol_date", "log_PE"]

    if columns is None:
        feature_df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    else:
        feature_df = df[columns]

    # Make sure it's numeric
    feature_df = feature_df.select_dtypes(include="number")

    # Handle n_components safely
    max_components = min(feature_df.shape[0], feature_df.shape[1])
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)

    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    comp_df = pd.DataFrame(
        components,
        columns=[f"{prefix}{i+1}" for i in range(components.shape[1])],
        index=df.index
    )

    result_df = pd.concat([df, comp_df], axis=1)

    # Print shape
    print(f"PCA Result Shape ({prefix}): {result_df.shape}")

    return result_df, pca

```


```python
# Income statement PCA
income_df_with_pca, income_pca_model = run_pca(
    income_selected,        # Pass full DataFrame; ID columns are preserved
    n_components=3,
    prefix="income_"
)

# Balance sheet PCA
balance_df_with_pca, balance_pca_model = run_pca(
    balance_selected,
    n_components=3,
    prefix="balance_"
)

# Cash flow PCA
cashflow_df_with_pca, cashflow_pca_model = run_pca(
    cashflow_selected,
    n_components=3,
    prefix="cashflow_"
)

```


```python
print(income_df_with_pca.columns)
print(balance_df_with_pca.columns)
print(cashflow_df_with_pca.columns)
```


```python
import pandas as pd
import statsmodels.api as sm

def regress_log_pe_on_pca(pca_df, prefix, label=""):
    """
    Run regression of log(PE) on PCA components and return both model + summary.

    Parameters
    ----------
    pca_df : pd.DataFrame
        DataFrame containing PCA components and 'log_PE' column.
    prefix : str
        Prefix for the PCA component names (e.g., 'income_', 'balance_', 'cashflow_').
    label : str, optional
        Label to display in printed output and summary.

    Returns
    -------
    model : statsmodels RegressionResults
        The fitted regression model.
    summary_df : pd.DataFrame
        Compact summary of coefficients, t-values, and p-values.
    """

    # Ensure log_PE exists
    if "log_PE" not in pca_df.columns:
        raise KeyError("'log_PE' column not found in provided DataFrame.")

    # Select PCA columns
    pca_cols = [c for c in pca_df.columns if c.startswith(prefix)]
    X = pca_df[pca_cols]
    y = pca_df["log_PE"]

    # Add intercept
    X = sm.add_constant(X)

    # Run regression
    model = sm.OLS(y, X).fit()

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        "coef": model.params,
        "t_value": model.tvalues,
        "p_value": model.pvalues
    }).reset_index().rename(columns={"index": "variable"})

    if label:
        summary_df["label"] = label
        print(f"{label.title()} PCA Regression:")
        print(f"  Components: {len(pca_cols)}")
        print(f"  Observations: {len(pca_df)}")
        print(f"  R-squared: {model.rsquared:.4f}\n")

    return model, summary_df

```


```python
income_model, income_summary = regress_log_pe_on_pca(income_df_with_pca, prefix="income_", label="income")
balance_model, balance_summary = regress_log_pe_on_pca(balance_df_with_pca, prefix="balance_", label="balance")
cashflow_model, cashflow_summary = regress_log_pe_on_pca(cashflow_df_with_pca, prefix="cashflow_", label="cashflow")
```


```python
all_pca_summaries = pd.concat([income_summary, balance_summary, cashflow_summary], ignore_index=True)
display(all_pca_summaries)
```


```python

```
