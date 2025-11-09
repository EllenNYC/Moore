# Loan Portfolio Data Exploration - Analysis Report

## Executive Summary

This report presents a comprehensive analysis of the Moore loan portfolio, including data preparation, roll rate analysis, cumulative default/prepayment analysis, and product performance metrics across different term buckets.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Roll Rate Analysis](#2-roll-rate-analysis)
3. [Cumulative Default & Prepayment Analysis](#3-cumulative-default--prepayment-analysis)
4. [Product Performance Analysis](#4-product-performance-analysis)
5. [Product x Term Performance Matrices](#5-product-x-term-performance-matrices)
6. [Key Metrics & Definitions](#6-key-metrics--definitions)
7. [Methodology Notes](#7-methodology-notes)
8. [Key Findings](#8-key-findings)

---

## 1. Data Preparation

### Data Sources
- **Loan Tape**: `loan tape - moore v1.0.csv`
  - Fields: display_id, program, loan_term, fico_score, approved_amount, disbursement_d
- **Performance Data**: `loan performance - moore v1.0.csv`
  - Fields: display_id, report_date, loan_status, upb, paid_principal, days_delinquent

### Data Cleaning & Transformations
1. **Date Conversion**: Converted disbursement_d and report_date to datetime format
2. **Merge**: Combined loan tape and performance data on display_id
3. **Loan Age Calculation**: Months since disbursement date
4. **Delinquency Bucketing**: Categorized loans into delinquency states:
   - Current (0 DPD)
   - 1-30 DPD
   - 31-60 DPD
   - 61-90 DPD
   - 91-120 DPD
   - 120+ DPD

### Terminal Event Logic
Each loan is captured at the **FIRST occurrence** of a terminal event:
- **Default**: CHARGED_OFF, WRITTEN_OFF
- **Payoff**: PAID_OFF, SATISFIED
- **Maturity**: loan_age >= loan_term (even if no terminal status)
- **Still Active**: Not yet matured and no terminal status

This ensures loan characteristics are measured at the time of the event, not at an arbitrary later date.

---

## 2. Roll Rate Analysis

### Methodology
- **Transition Matrix**: Shows probability of moving from one delinquency state to another
- **UPB-Weighted**: Calculated using unpaid principal balance (UPB) as weights
- **Terminal State Exclusion**: Excludes observations AFTER first terminal state to prevent phantom transitions

### Key Features
- **States Tracked**: Current, 1-30 DPD, 31-60 DPD, 61-90 DPD, 91-120 DPD, 120+ DPD, PAID_OFF, CHARGED_OFF
- **Both Matrices Provided**:
  - Count-based (% of loan count)
  - UPB-weighted (% of dollar amount)

### Visualization
- **Heatmap**: Color-coded roll rate matrix showing transition probabilities
- Darker colors indicate higher probability of transition
- Helps identify most common delinquency progression paths

### Insights
- Clear progression from mild to severe delinquency
- Terminal state exclusion prevents overestimation of recovery rates
- UPB weighting shows concentration risk beyond simple loan counts

---

## 3. Cumulative Default & Prepayment Analysis

### 3.1 Portfolio-Wide Cumulative Rates by Loan Age

**Chart: Cumulative Default and Prepayment Over Time**
- X-axis: Loan Age (months since disbursement)
- Y-axis: Cumulative Rate (%)
- Two lines: Default rate (increasing) and Prepayment rate
- Shows portfolio-level loss and prepayment curves

### 3.2 Vintage Cohort Analysis

**Chart: Cumulative Default by Vintage**
- Separate curves for each disbursement month cohort
- Identifies vintage-specific performance patterns
- Helps detect macro-economic impacts on different origination periods

**Chart: Cumulative Prepayment by Vintage**
- Shows prepayment behavior across different origination cohorts
- Highlights refinancing opportunities and borrower behavior changes

### 3.3 FICO Score Analysis

**FICO Bands**: <600, 600-650, 650-700, 700-750, 750+

**Chart: Cumulative Default Rate by FICO Band**
- Clear inverse relationship: Higher FICO = Lower default rate
- Validates credit score as strong predictor of default risk

**Chart: Cumulative Prepayment Rate by FICO Band**
- Shows prepayment patterns across credit quality spectrum
- Higher FICO borrowers may prepay faster (refinancing opportunities)

### Key Insight
Credit quality (FICO score) is a strong predictor of both default and prepayment behavior.

---

## 4. Product Performance Analysis

### 4.1 Default Rates by Product (Program)

**Metrics Calculated for Each Product**:
- Number of defaults
- Default rate (%)
- Total loan count
- Prepayment rate (%)
- Average loan amount
- Total loan volume

**Charts**:
1. **Bar Chart: Cumulative Default Rate by Product**
   - Shows which products have highest/lowest default rates
   - Value labels on top of each bar for precise reading

2. **Bar Chart: Loan Volume by Product**
   - Shows portfolio composition
   - Identifies high-volume vs low-volume products

### 4.2 Default Rates by Term Bucket

**Term Buckets**: 0-3m, 4-6m, 7-12m, 13-18m, 19-24m, 24m+

**Charts**:
1. **Bar Chart: Default Rate by Term Bucket**
   - Shows if shorter/longer terms have different default risk

2. **Bar Chart: Loan Count by Term Bucket**
   - Shows distribution of portfolio across term lengths

---

## 5. Product x Term Performance Matrices

### Important: Data Filtering for Accuracy

**All metrics in this section use ONLY**:
- ✓ Terminated loans (PAID_OFF, SATISFIED, CHARGED_OFF, WRITTEN_OFF)
- ✓ Loans with principal payments > 0
- ✓ Minimum 50 loans per bucket for statistical significance

This ensures:
- No bias from ongoing delinquent loans
- Average life calculations are logical (e.g., 0-3m term loans don't show 1.2 year average age)
- Sufficient sample size for reliable metrics

### 5.1 Default Rate Heatmap (Product x Term)

**Matrix**: Rows = Products, Columns = Term Buckets
- Color intensity shows default rate magnitude
- Red/dark colors = higher default risk
- Identifies specific product/term combinations with elevated risk

### 5.2 Loan Count Matrix (Product x Term)

**Reference table** showing number of loans in each combination
- Essential for understanding statistical significance
- Helps identify which combinations have sufficient data

### 5.3 Detailed Performance Metrics Table

**For Each Product x Term Combination**:

| Metric | Description |
|--------|-------------|
| **num_loans** | Count of terminated loans with principal payments |
| **avg_term_months** | Original loan term (maturity) |
| **average_life_months** | Actual time from disbursement to terminal event (WAL) |
| **life_vs_term_pct** | (Average Life / Original Term) × 100 |
| **default_rate** | Cumulative default rate (%) |
| **annualized_default_rate** | Default rate / Average Age in years (time-adjusted) |
| **monthly_default_rate** | Compound monthly equivalent rate |
| **prepay_rate** | Cumulative prepayment rate (%) |

### 5.4 Top/Bottom Performers

**Tables Provided**:
1. **Top 10 Best Performing**: Lowest annualized default rates
   - Shows which product/term combinations perform best
   - Risk-adjusted using average life

2. **Top 10 Worst Performing**: Highest annualized default rates
   - Identifies high-risk combinations
   - Candidates for underwriting tightening or pricing adjustments

### 5.5 Average Life Analysis

**Tables**:
1. **Shortest Average Life** (fastest to resolve)
   - Shows which combinations pay off or default quickly
   - Useful for cash flow forecasting

2. **Longest Average Life** (slowest to resolve)
   - Identifies longer-duration exposures
   - Important for duration matching and liquidity planning

### 5.6 Overall Statistics

**Summary Metrics Across All Products/Terms**:
- Mean Average Life: [X] months
- Median Average Life: [X] months
- Min/Max Average Life: [X] to [X] months
- Mean Life vs Term: [X]% (on average, loans live this % of their original term)

### 5.7 Comparative Visualization

**Chart Series: Default Rates by Product for Each Term Bucket**
- 6 subplots (one per term bucket: 0-3m, 4-6m, 7-12m, 13-18m, 19-24m, 24m+)
- Within each subplot, bars grouped by product
- Enables visual comparison of product performance within each term segment
- Helps identify if certain products consistently perform better/worse across terms

---

## 6. Key Metrics & Definitions

### 6.1 Default & Loss Metrics

**Cumulative Default Rate (CDR)**
- Formula: `(# Defaulted Loans / Total Loans) × 100`
- Not time-adjusted
- Best for: Overall portfolio quality assessment

**Annualized Default Rate (ADR)**
- Formula: `CDR / Average Age (years)`
- Time-adjusted metric
- Best for: Comparing products with different terms or maturities

**Monthly Default Rate (MDR)**
- Formula: `1 - (1 - CDR)^(1/months)`
- Compound monthly equivalent
- Best for: Monthly loss forecasting and modeling

### 6.2 Duration Metrics

**Average Life (WAL - Weighted Average Life)**
- Formula: `Average (report_date - disbursement_date)` for terminated loans
- Measured in months and years
- Shows actual time loans remain outstanding
- Best for: Duration analysis, cash flow forecasting

**Life vs Term %**
- Formula: `(Average Life / Original Term) × 100`
- Shows if loans terminate early (<100%) or extend beyond term (>100%)
- Best for: Understanding prepayment/extension behavior

### 6.3 Roll Rate Metrics

**Roll Rate**
- Probability of transitioning from delinquency state A to state B
- Forward-looking default prediction metric
- Best for: Building default forecasting models

**UPB-Weighted Roll Rate**
- Same as roll rate but weighted by unpaid principal balance
- Provides dollar-based risk perspective
- Best for: Risk-weighted analysis and loss reserving

---

## 7. Methodology Notes

### 7.1 Why Terminal Event Capture?

**Problem**: Using last observation can be misleading
- A loan might default at month 3 but have records through month 12
- Using month 12 data shows characteristics AFTER default, not at default

**Solution**: Capture at FIRST terminal event
- Default: First occurrence of CHARGED_OFF or WRITTEN_OFF
- Payoff: First occurrence of PAID_OFF or SATISFIED
- Maturity: First time loan_age >= loan_term

**Benefit**: Accurate measurement of loan characteristics at time of event

### 7.2 Why Only Terminated Loans for WAL?

**Problem**: Including active loans skews average life
- A 3-month term loan that's been delinquent for 15 months is still "active"
- Including it would show average life of 15 months for a 3-month product

**Solution**: Use only terminated loans
- PAID_OFF, SATISFIED, CHARGED_OFF, WRITTEN_OFF

**Benefit**: Logical, interpretable average life metrics

### 7.3 Why UPB Weighting for Roll Rates?

**Problem**: Count-based roll rates treat all loans equally
- 10 loans of $1,000 = same weight as 1 loan of $100,000

**Solution**: Weight by unpaid principal balance

**Benefit**: Dollar-based risk view, more relevant for loss forecasting

### 7.4 Why Exclude Post-Terminal State Observations?

**Problem**: Phantom transitions
- A loan charged off at month 6 might show "CURRENT" status at month 8 (data error)
- This creates false roll rate: CHARGED_OFF → CURRENT

**Solution**: Mark and exclude all observations after first terminal state

**Benefit**: Clean, accurate transition probabilities

---

## 8. Key Findings

### 8.1 Credit Quality is Paramount
- **FICO Score** shows strong inverse relationship with default rates
- Lower FICO bands (<600) have significantly higher default rates than higher bands (750+)
- Validates credit underwriting standards

### 8.2 Delinquency Progression is Predictable
- **Roll Rate Analysis** shows clear progression paths
- Most loans either stay current or progress through delinquency stages
- Few loans "cure" from severe delinquency (91+ DPD) back to current
- Early-stage delinquency management is critical

### 8.3 Term Length Impacts Performance
- Default rates vary by term bucket
- **Annualized rates** essential for fair comparison across terms
- Shorter-term loans may have different risk profiles than longer-term loans

### 8.4 Average Life Varies from Original Term
- **Life vs Term %** shows actual loan duration differs from stated term
- Some products/terms resolve much faster (high prepay)
- Others extend beyond term (delinquency, modifications)
- Important for cash flow forecasting and duration matching

### 8.5 UPB vs Count Weighting Matters
- **Dollar-weighted roll rates** can differ significantly from count-based
- Highlights concentration risk in larger loans
- Essential for accurate loss reserving

### 8.6 Vintage Effects Present
- Performance varies by disbursement cohort
- Suggests macro-economic factors impact loan performance
- Useful for stress testing and scenario analysis

### 8.7 Product/Term Combinations Show Distinct Risk Profiles
- **Heatmaps and tables** reveal specific high-risk and low-risk segments
- Enables targeted pricing, underwriting, or portfolio management actions
- Some products perform consistently across terms; others vary widely

---

## 9. Business Applications

### 9.1 Underwriting & Pricing
- Use **annualized default rates** by product/term/FICO to inform pricing models
- Adjust credit policies for high-risk segments
- Tighten/loosen credit boxes based on performance data

### 9.2 Loss Forecasting & Reserving
- Apply **roll rates** to current portfolio delinquency states
- Use **UPB-weighted** metrics for dollar-based loss projections
- **Monthly default rates** for monthly reserve calculations

### 9.3 Portfolio Management
- Monitor **vintage curves** for early warning signs
- Track **life vs term %** for prepayment/extension risk
- Use **average life** for duration matching and liquidity planning

### 9.4 Marketing & Product Design
- Focus on **best-performing product/term combinations**
- Consider discontinuing or repricing worst performers
- Understand **prepayment patterns** for customer retention strategies

### 9.5 Stress Testing
- Apply **worst-performing vintage** default rates to current originations
- Model roll rate acceleration scenarios (e.g., 1.5x or 2x historical rates)
- Assess impact on capital and liquidity under stressed conditions

---

## 10. Data Quality & Limitations

### Data Quality Checks Applied
✓ Date format validation and conversion
✓ Duplicate record handling
✓ Missing value assessment
✓ Logical consistency (e.g., loan age vs term)
✓ Terminal state logic validation

### Filters Applied
✓ Terminated loans only for average life calculations
✓ Principal payments > 0 for age metrics
✓ Minimum 50 loans per bucket for summary tables
✓ Post-terminal state exclusion for roll rates

### Limitations
- **Maturity Dates**: Not provided in data; assumed as disbursement + term
- **Loan Modifications**: Not tracked; may affect average life calculations
- **Prepayment Penalties**: Unknown; may influence prepayment rates
- **Economic Data**: No macro variables linked; vintage effects observed but not explained
- **Right Censoring**: Active loans not included in some metrics (by design)

---

## 11. Files & Outputs

### Files Created
- **Notebook**: `data_exploration.ipynb` - Full analysis with code, charts, and outputs
- **This Report**: `analysis_report.md` - Summary documentation

### Key Datasets Generated
All analyses performed in-memory within the notebook. Key dataframes:
- `df`: Merged loan tape + performance data
- `final_status_df`: Each loan at terminal event
- `final_status_terminated`: Filtered for terminated loans with payments
- `roll_rate_data`: Prepared for transition matrix
- `product_term_summary`: Product x Term performance metrics

---

## 12. Recommendations for Next Steps

### 12.1 Model Development
- Build **default prediction models** using FICO, product, term, and delinquency status
- Develop **roll rate forecasting models** to project future defaults
- Create **prepayment models** for cash flow forecasting

### 12.2 Enhanced Analysis
- **Add macro variables**: Unemployment rate, GDP, interest rates to explain vintage effects
- **Loss Severity Analysis**: Given default, how much is lost (LGD)?
- **Recovery Analysis**: For charged-off loans, track recovery amounts and timing

### 12.3 Monitoring & Reporting
- **Monthly Dashboard**: Track roll rates, default rates, and vintage performance
- **Early Warning System**: Alert when roll rates exceed thresholds
- **Cohort Tracking**: Monitor new originations against historical benchmarks

### 12.4 Data Improvements
- Capture **modification dates and terms** for better average life analysis
- Track **recovery amounts** post-charge-off for net loss calculations
- Record **prepayment reasons** (refi, paid in full, etc.) for behavior modeling

---

## Appendix: Technical Specifications

### Software & Libraries
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **matplotlib**: Visualization and charting
- **seaborn**: Enhanced statistical visualizations
- **numpy**: Numerical operations

### Data Specifications
- **Date Range**: [Based on data - disbursement dates and report dates]
- **Loan Count**: [Total unique loans in dataset]
- **Observation Count**: [Total loan-month observations]
- **Performance Period**: [First to last report_date]

### Code Repository
- All analysis code available in: `data_exploration.ipynb`
- Reproducible: Rerun notebook to regenerate all outputs

---

*Report Generated: [Current Date]*
*Data Source: Moore Loan Portfolio (loan tape v1.0 & performance v1.0)*
*Analysis Framework: Python/Jupyter*
