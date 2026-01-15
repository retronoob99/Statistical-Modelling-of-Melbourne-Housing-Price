# Statistical-Modelling-of-Melbourne-Housing-Price

This project analyzes and models housing prices in Melbourne using multiple linear regression, interaction effects, feature selection, and model diagnostics, with results summarized in the included presentation.
​

# Project goals
Key questions explored in this project include: which regression model best predicts house prices, how rooms and distance relate to price, whether property types differ in pricing patterns, how bathrooms influence price, and whether room effects interact with region (Southern Metropolitan).
​

# Repository contents
final_project.R: Full R workflow for sampling data, feature engineering, fitting regression models (additive/interaction), model selection, diagnostics, transformations, and prediction intervals.
​

Final-Project-Presentation-Melbource-housing.pptx: Slide deck summarizing research questions, EDA highlights, model comparison/selection, diagnostic checks, transformations, and conclusions.
​

# Data
The R script reads a cleaned Melbourne housing dataset from a local CSV path (MelbournehousingFULLcleaned.csv) and samples 200 rows for analysis.
​

Note: The CSV file is not included in the repository as provided; update the read.csv(...) path in final_project.R to match your local/project structure.
​

# Methods and modeling
Feature engineering
The script constructs:

Centered variables for Rooms and Distance (subtracting the mean).
​

Binary indicators for property type (e.g., Unit, Townhouse) and region (Southern Metropolitan).
​

Additional additive terms and a large set of pairwise interaction terms among predictors for the interaction model.
​

Models compared
Full model with a wide set of predictors plus interactions.
​

Additive-only model (no interactions).
​

Interaction model (includes many interaction terms).
​

The presentation discusses the conceptual difference between additive vs interaction structure (e.g., room effect varying by location in the interaction model).
​

Model selection
Multiple selection approaches are applied, including:

Best subset selection via leaps::regsubsets.
​

Backward/forward selection and stepwise regression via step(...).
​

The slide deck reports AIC/R²-based comparisons across backward elimination, forward selection, and stepwise regression.
​

# Diagnostics and transformations
The workflow includes standard regression checks and influence analysis:

Normality checks (QQ plot + Shapiro-Wilk test).
​

Linearity and homoscedasticity checks using model diagnostic plots.
​

Breusch–Pagan test (lmtest::bptest) for heteroscedasticity.
​

Cook’s distance to identify influential observations.
​

Transformations explored include Box-Cox analysis (MASS::boxcox) and refitting variants such as log-price and sqrt-price models, with AIC comparisons.
​

# Visualizations and outputs
The script generates:

Side-by-side plots comparing predicted values from interaction vs additive models using ggplot2 and gridExtra.
​

Diagnostic plots (residual plots, QQ plot) and Cook’s distance plot.
​

The presentation highlights findings such as strong relationships between home features and price, region-based differences, and the presence of influential outliers and heteroscedasticity in residuals.
​

# How to run
Open final_project.R in RStudio.
​

Update the dataset path in read.csv(...) to point to your local copy of the cleaned Melbourne housing CSV.
​

Install required packages (see below).
​

Run the script top-to-bottom to reproduce model fits, plots, selection steps, diagnostics, and prediction interval example.
​

# Dependencies
The script uses (at minimum) the following R packages:

ggplot2, plotly, gridExtra, scales (visualization)
​

leaps (subset selection)
​

lmtest (Breusch–Pagan test)
​

MASS (Box-Cox transformation)
​

# Notes / limitations
The code currently samples 200 rows from the full dataset, so results may vary depending on the random seed and sample.
​

The dataset file is referenced via an absolute local path and must be changed for portability.
​

Some slides report summary metrics (e.g., correlations, R², AIC comparisons) that may depend on the exact dataset and sampling used.
​

Presentation
See Melbourne_Housing.pptx for a structured explanation of the motivation, research questions, model comparison, diagnostics, transformation results, and conclusions.
​

