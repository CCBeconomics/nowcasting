# [GDP Growth Nowcasting Workshop: Trinidad and Tobago](https://CCBeconomics.github.io/nowcasting/tto/)

**Facilitators**: Jose L. Saboin, Diego Guerrero, Angelo Mazzocca

**Course overview:** A training programme for public-sector professionals working on economic policy and national statistics in Trinidad and Tobago. It gives participants the theoretical foundations and the practical skills to build and run a GDP-growth *nowcasting* tool in Python and R, using structured and unstructured real-time data: Central Bank of Trinidad and Tobago (CBTT) statistics, international trade and energy prices, Google Trends and satellite nighttime lights, to estimate economic activity before official GDP is published.

---

**Friday: Installation clinic (in person)**
- [Installation guide (start here)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/install/install_guide.txt)
- Environment files: [environment.yml](https://github.com/CCBeconomics/nowcasting/blob/main/tto/install/environment.yml) | [environment-geo.yml](https://github.com/CCBeconomics/nowcasting/blob/main/tto/install/environment-geo.yml) | [Manual installation](https://github.com/CCBeconomics/nowcasting/blob/main/tto/install/manual_install.txt)
- [Activity 0: Python check](https://github.com/CCBeconomics/nowcasting/blob/main/tto/install/act0_env_check.ipynb)
- [Activity 0: R check](https://github.com/CCBeconomics/nowcasting/blob/main/tto/install/act0_env_check_R.Rmd)

**Monday (morning): Welcome, Python and R fundamentals**
- Python fundamentals: [Activity 1: Python basics](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act1_python_basics.ipynb)
- [Slides: R fundamentals](https://CCBeconomics.github.io/nowcasting/tto/slides/s1_r.html) | [Activity 1: R basics](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act1_r_basics.R)
- Data: [gdp.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/gdp.csv) | [ntl_shape.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/ntl_shape.csv) | [eia_brent.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/eia_brent.csv)

**Monday (afternoon): Data: scraping, Google Trends, nighttime lights and the build**
- [Slides: Web scraping](https://CCBeconomics.github.io/nowcasting/tto/slides/s2_scraping.html) | [Activity 2: Web scraping (FRED, CBTT, Excel)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act2_scraping.ipynb)
- [Slides: Google Trends](https://CCBeconomics.github.io/nowcasting/tto/slides/s4_gtrends.html) | [Activity 3: Google Trends](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act3_gtrends.ipynb)
- [Slides: Nighttime lights](https://CCBeconomics.github.io/nowcasting/tto/slides/s3_ntl.html) | [Activity 4: Nighttime lights](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act4_ntl.ipynb) | [config.py](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/config.py) | [ntl_helpers.py](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/ntl_helpers.py) | [Coordinates.xlsx](https://github.com/CCBeconomics/nowcasting/blob/main/tto/raw/Coordinates.xlsx)
- [Activity 5: Building the dataset](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act5_build.ipynb)
- Data: [TT_GTrends_Keywords.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/raw/TT_GTrends_Keywords.csv) | [data.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/data.csv) | [meta.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/meta.csv)

**Tuesday (morning): Block 1 (regularization), Block 2 (trees) and Block 3 (LSTM)**
- Block 1 | [Theory: Regularization](https://CCBeconomics.github.io/nowcasting/tto/slides/theory_1_regularization.pdf) | [Activity 6: Regularization (Python)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act6_regularization.ipynb) | [Activity 6 (R)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act6_regularization.R)
- Block 2 | [Theory: Tree-based models](https://CCBeconomics.github.io/nowcasting/tto/slides/theory_2_trees.pdf) | [Activity 7: Trees, random forest, boosting (Python)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act7_trees.ipynb) | [Activity 7 (R)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act7_trees.R)
- Block 3 | [Theory: LSTM](https://CCBeconomics.github.io/nowcasting/tto/slides/theory_3_lstm.pdf) | [Activity 8: LSTM (Python)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act8_lstm.ipynb)
- Data: [blocks_quarterly.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/blocks_quarterly.csv) | [blocks_meta.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/blocks_meta.csv) | [blocks_monthly.csv](https://github.com/CCBeconomics/nowcasting/blob/main/tto/data/blocks_monthly.csv)

**Tuesday (afternoon): Methodology, variable selection and a first nowcast**
- [Slides: Nowcasting methodology](https://CCBeconomics.github.io/nowcasting/tto/slides/s5_methodology.html) | [Nowcasting in the Caribbean (PDF)](https://CCBeconomics.github.io/nowcasting/tto/slides/s5_methodology_caribbean.pdf)
- [Activity 9: Variable selection and a first nowcast (OLS)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act9_selection.ipynb)

**Wednesday (morning): The nowcast tool and Block 4 (time-series models)**
- [Activity 10: The nowcast tool](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act10_nwcst.ipynb) | [nwcst_helpers.py](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/nwcst_helpers.py)
- Block 4 | [Theory: DFM, BVAR and MIDAS](https://CCBeconomics.github.io/nowcasting/tto/slides/theory_4_timeseries.pdf) | [Activity 11: Time-series models (R)](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act11_timeseries.R)

**Wednesday (afternoon): Results**
- [Activity 12: Results](https://github.com/CCBeconomics/nowcasting/blob/main/tto/activity/act12_results.ipynb)

---

## Folder layout

```
install/    environment files and the two act0 checks
activity/   the notebooks (.ipynb) and R scripts (.R) you work in, plus config.py (the one file you edit)
data/       inputs shipped with the course (Trinidad and Tobago panel, GDP, lights, Trends, the block datasets)
raw/        Coordinates.xlsx; downloads land here when you run the activities
slides/     the presentations (HTML) and theory notes (PDF)
output/     written by the activities: the model-block scoreboard (workshop_results.csv) and the nowcast (Activity 10)
```

Download the repository (green **Code** button → **Download ZIP**), then open the `tto/` folder in VS Code or JupyterLab and run each notebook from `activity/`, so that `../data/` resolves. R scripts: set the working directory to `activity/` first. Kernel `nwcst` for every notebook except Activity 4, which needs `geo`.

**Credentials:** Activity 2 needs a free FRED API key and Activity 4 a free NASA LAADS token. Paste them into your own copy only: never into a file you share.
