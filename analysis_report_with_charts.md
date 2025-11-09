# Moore Loan Portfolio - Data Exploration Report

**Generated:** 2025-11-08 20:44:39

**Data Source:** loan tape - moore v1.0.csv, loan performance - moore v1.0.csv

---

# Comprehensive Loan Data Exploration & Analysis

This notebook provides comprehensive data exploration including:
- Data loading and initial exploration from model.ipynb
- **Loan age calculation** from disbursement date
- **Delinquency status tracking** over time
- **Historical roll rate analysis** (transition matrices)
- **Cumulative default rates** by vintage and characteristics
- **Cumulative prepayment rates** by vintage and characteristics

---

## 1. Setup and Data Loading

```
Libraries loaded successfully!

```

```
Loan Tape Shape: (83235, 11)

Columns: ['display_id', 'program', 'loan_term', 'mdr', 'int_rate', 'fico_score', ' approved_amount ', 'disbursement_d', ' co_amt_est ', 'vertical', 'issuing_bank']
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 83235 entries, 0 to 83234
Data columns (total 11 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   display_id       83235 non-null  object
 1   program          83235 non-null  object
 2   loan_term        83235 non-null  int64 
 3   mdr              83235 non-null  object
 4   int_rate         83235 non-null  object
 5   fico_score       83235 non-null  int64 
 6   approved_amount  83235 non-null  object
 7   disbursement_d   83235 non-null  object
 8   co_amt_est       79241 non-null  object
 9   vertical         83235 non-null  object
 10  issuing_bank     83235 non-null  object
dtypes: int64(2), object(9)
memory usage: 7.0+ MB

```

```
Loan Performance Shape: (1045858, 9)

Columns: ['display_id', 'report_date', 'co_amt', 'charge_off_date', 'loan_status', 'upb', 'paid_principal', 'paid_interest', 'days_delinquent']
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1045858 entries, 0 to 1045857
Data columns (total 9 columns):
 #   Column           Non-Null Count    Dtype  
---  ------           --------------    -----  
 0   display_id       1045858 non-null  object 
 1   report_date      1045858 non-null  object 
 2   co_amt           1045858 non-null  float64
 3   charge_off_date  57468 non-null    object 
 4   loan_status      1045858 non-null  object 
 5   upb              1045858 non-null  float64
 6   paid_principal   1045858 non-null  float64
 7   paid_interest    1045858 non-null  float64
 8   days_delinquent  1045858 non-null  int64  
dtypes: float64(4), int64(1), object(4)
memory usage: 71.8+ MB

```

## 2. Data Cleaning and Preprocessing

```
Loan tape cleaned!
================================================================================
PORTFOLIO STATISTICS
================================================================================

Numeric Columns Summary:

```

```
       approved_amount     loan_term      int_rate           mdr  \
count     83235.000000  83235.000000  83235.000000  83235.000000   
mean       4483.484973     20.385331      0.158837      0.056037   
std        3601.481484     13.670825      0.097828      0.052709   
min         500.000000      3.000000      0.000000      0.029000   
25%        1891.100000     12.000000      0.079000      0.039000   
50%        3463.400000     24.000000      0.149000      0.039000   
75%        6000.000000     24.000000      0.249000      0.039000   
max       25000.000000     60.000000      0.299000      0.300000   

         fico_score    co_amt_est  
count  83235.000000  79241.000000  
mean     704.518820    275.216569  
std       78.594371    408.861870  
min      540.000000      1.010000  
25%      644.000000     34.890000  
50%      706.000000    116.770000  
75%      769.000000    341.810000  
max      850.000000   5776.560000  
```

```
Loan performance cleaned!

```

## 3. Origination Analysis

###  The distribution of key loan characteristics:

* Loan terms are concentrated in 12 and 24 months, with smaller segments at 3, 6, 36, and 60 months.
* Approved loan amounts are right-skewed, with most loans under $5,000.
* FICO scores are broadly distributed, but the majority fall between 600 and 750.
* Interest rates vary widely, with higher rates more common among lower FICO bands.
* The portfolio includes a mix of programs, but recent vintages show a shift toward higher quality (P1) and away from higher risk (P3).
* The top verticals by loan count are Home Services, Legal Services, and Automotive.

![Chart 1](/home/ycnin/moore/charts/chart_1.png)

*Chart 1: [Description based on analysis]*

```

Key Portfolio Metrics:
Total Loans: 83,235
Total Portfolio Value: $373,182,872
Average Loan Size: $4,483
Average FICO Score: 705
Average Interest Rate: 15.88%
Average Loan Term: 20.4 months

```

### Generate summary statistics by program
* Credit quality: P3<P2<P1
* Younger population: P3 program targets more risk population with lower FICO
* Higher interest rate risk: APR for P3 is much higher (25% vs 18%/8%)
* Lower Balance: approved balance for P3 is lower
* Shorter term: majority P3 are short term loans (12M or less) compared to P1/P2 (24M)

```
        Loan_Count Avg_Loan_Term Avg_MDR Avg_Int_Rate Avg_FICO  \
P1        31957.00         21.49    4.67         8.05   770.54   
P2        32785.00         21.71    5.37        18.20   695.97   
P3        18493.00         16.12    7.62        25.32   605.58   
Overall   83235.00         20.39    5.60        15.88   704.52   

        Avg_Approved_Amount Avg_co_amt_est_pct_of_Approved  
P1                  5437.32                           1.56  
P2                  4277.80                           7.47  
P3                  3199.85                          18.46  
Overall             4483.48                           7.39  
```

### Key Portfolio Trends- credit quality evolved over time:

1. Program Mix Evolution:
   - P1 (highest quality) has been increasing since 2023 Q2, from ~30% to ~56% of originations
   - P3 (highest risk) has been decreasing since 2022 Q4 peak of ~29% to ~12% in latest quarter
   - P2 has remained relatively stable around 40% until recent decline to ~31%

2. Credit Quality by Program:
   - Stay relatively unchanged

This shift toward higher quality originations (more P1, less P3) since 2023 suggests a more conservative credit strategy being implemented.

![Chart 2](/home/ycnin/moore/charts/chart_2.png)

*Chart 2: [Description based on analysis]*

![Chart 3](/home/ycnin/moore/charts/chart_3.png)

*Chart 3: [Description based on analysis]*

![Chart 4](/home/ycnin/moore/charts/chart_4.png)

*Chart 4: [Description based on analysis]*

## 4. Merge hist and static table, calculate additional columns

```
Merged dataset shape: (1039585, 16)

Vintage Distribution:
vintage
2019-10     196
2019-11      96
2019-12     141
2020-01     460
2020-02     584
2020-03    1625
2020-04    1505
2020-05    2563
2020-06    3320
2020-07    5546
Freq: M, Name: count, dtype: int64

```

```

Delinquency Status Distribution:
delinquency_bucket
CURRENT       571352
Prepaid       313268
Default        57023
1-30 DPD       37809
31-60 DPD      14156
61-90 DPD      10276
91-120 DPD      8050
120+ DPD         955
Name: count, dtype: int64

Delinquency Severity Rates:
Any Delinquency: 7.03%
30+ DPD: 3.30%
60+ DPD: 1.90%
90+ DPD: 0.89%

```

## 4. Create Delinquency, CumDft, CumLoss Charts

![Chart 5](/home/ycnin/moore/charts/chart_5.png)

*Chart 5: [Description based on analysis]*

```
array([0, True], dtype=object)
```

![Chart 6](/home/ycnin/moore/charts/chart_6.png)

*Chart 6: [Description based on analysis]*

![Chart 7](/home/ycnin/moore/charts/chart_7.png)

*Chart 7: [Description based on analysis]*

![Chart 8](/home/ycnin/moore/charts/chart_8.png)

*Chart 8: [Description based on analysis]*

```
program
P1    0.997186
P2    0.997919
P3    0.999819
dtype: float64
```

![Chart 9](/home/ycnin/moore/charts/chart_9.png)

*Chart 9: [Description based on analysis]*

## 4. Historical Roll Rate Analysis

Roll rates show the probability of loans transitioning from one delinquency state to another (or to default/prepay) in the next period.

```
loan_status
PAID_OFF       27172
CHARGED_OFF     7253
CURRENT         3184
DELINQUENT        16
Name: count, dtype: int64
```

```
                  display_id report_date  co_amt charge_off_date  loan_status  \
1039438  00052567-0-aa23e02d  2023-10-31     0.0             NaT  WRITTEN_OFF   
883959   00221721-0-21a2f5ee  2023-02-28     0.0             NaT  WRITTEN_OFF   
1019987  0040b1e5-0-55d4bbe9  2023-09-30     0.0             NaT  WRITTEN_OFF   
436325              004c09d2  2022-05-31     0.0             NaT  WRITTEN_OFF   
700877              005b6ea0  2022-09-30     0.0             NaT  WRITTEN_OFF   

         upb  paid_principal  paid_interest  days_delinquent disbursement_d  \
1039438  0.0             0.0            0.0                0     2023-09-29   
883959   0.0             0.0            0.0                0     2023-02-03   
1019987  0.0             0.0            0.0                0     2023-07-19   
436325   0.0             0.0            0.0                0     2022-05-03   
700877   0.0             0.0            0.0                0     2022-09-14   

         loan_term  fico_score  approved_amount program        vertical  \
1039438          3         850          14000.0      P1   Home Services   
883959          24         595           5000.0      P3  Legal Services   
1019987         12         787           1922.0      P2   Home Services   
436325          24         684          14500.0      P2  Legal Services   
700877          24         728           4500.0      P2  Legal Services   

         loan_age_months  vintage disbursement_year disbursement_quarter  \
1039438                1  2023-09              2023               2023Q3   
883959                 1  2023-02              2023               2023Q1   
1019987                2  2023-07              2023               2023Q3   
436325                 1  2022-05              2022               2022Q2   
700877                 1  2022-09              2022               2022Q3   

        delinquency_bucket  is_delinquent  is_30plus  is_60plus  is_90plus  \
1039438                NaN          False      False      False      False   
883959                 NaN          False      False      False      False   
1019987                NaN          False      False      False      False   
436325                 NaN          False      False      False      False   
700877                 NaN          False      False      False      False   

         prev_upb  cum_co_loss  upb_at_dft  loss_at_dft  cum_charge_off  
1039438   14000.0          0.0         0.0          0.0             0.0  
883959        NaN          0.0         0.0          0.0             0.0  
1019987    1922.0          0.0         0.0          0.0             0.0  
436325        NaN          0.0         0.0          0.0             0.0  
700877        NaN          0.0         0.0          0.0             0.0  
```

```
Roll rate analysis dataset: (936220, 32)

Next state distribution:
next_delinquency_bucket
CURRENT       495768
Prepaid       312596
Default        57023
1-30 DPD       37396
31-60 DPD      14156
61-90 DPD      10276
91-120 DPD      8050
120+ DPD         955
Name: count, dtype: int64

```

```
next_delinquency_bucket
CURRENT       495768
Prepaid       312596
Default        57023
1-30 DPD       37396
31-60 DPD      14156
61-90 DPD      10276
91-120 DPD      8050
120+ DPD         955
Name: count, dtype: int64
```

```

====================================================================================================
UPB-WEIGHTED Roll Rate Matrix (% of balance/UPB transitioning from row state to column state):
====================================================================================================
next_delinquency_bucket  CURRENT  1-30 DPD  31-60 DPD  61-90 DPD  91-120 DPD  \
delinquency_bucket                                                             
1-30 DPD                   28.48     36.69      31.92       0.62        0.01   
120+ DPD                    0.98      0.13        NaN       0.18        0.82   
31-60 DPD                   6.69      8.27      13.10      69.01        1.48   
61-90 DPD                   2.75      1.84       2.48       9.88       78.54   
91-120 DPD                  0.95      0.52       0.35       1.38        4.51   
CURRENT                    93.47      4.06       0.13       0.01        0.00   

next_delinquency_bucket  120+ DPD  Prepaid  Default  
delinquency_bucket                                   
1-30 DPD                     0.01     2.22     0.06  
120+ DPD                     9.80     0.60    87.50  
31-60 DPD                    0.06     1.34     0.06  
61-90 DPD                    0.55     0.73     3.23  
91-120 DPD                  10.32     0.48    81.48  
CURRENT                      0.00     2.32     0.01  
====================================================================================================
COUNT-BASED Roll Rate Matrix (% of loan count transitioning from row state to column state):
====================================================================================================
next_delinquency_bucket  CURRENT  1-30 DPD  31-60 DPD  61-90 DPD  91-120 DPD  \
delinquency_bucket                                                             
1-30 DPD                   26.83     35.22      32.72       0.61        0.02   
120+ DPD                    0.75      0.32       0.00       0.11        0.86   
31-60 DPD                   6.81      8.40      12.06      68.25        1.64   
61-90 DPD                   2.68      1.88       2.54       9.49       78.24   
91-120 DPD                  0.95      0.52       0.33       1.48        4.38   
CURRENT                    90.34      4.41       0.14       0.01        0.00   

next_delinquency_bucket  120+ DPD  Prepaid  Default  
delinquency_bucket                                   
1-30 DPD                     0.00     4.55     0.05  
120+ DPD                    10.18     1.50    86.28  
31-60 DPD                    0.02     2.74     0.08  
61-90 DPD                    0.47     1.62     3.09  
91-120 DPD                  10.54     1.00    80.80  
CURRENT                      0.00     5.09     0.01  

```

![Chart 10](/home/ycnin/moore/charts/chart_10.png)

*Chart 10: [Description based on analysis]*

## 5. Cumulative Default by term and program

```
Total unique loans: 76,669

================================================================================
LOAN TERMINAL EVENT SUMMARY
================================================================================
Defaulted (CHARGED_OFF):  3,054 (11.6%)
Prepaid (PAID_OFF):         21,764 (82.5%)
At Maturity (no terminal state):      1,549 (5.9%)
Still Active (not yet matured):       Deleted from analysis

Loan Status Distribution at Terminal Event:
loan_status
PAID_OFF        20681
CHARGED_OFF      3054
CURRENT          2286
DELINQUENT        281
GRACE_PERIOD       62
SATISFIED           2
TRANSFERRED         1

================================================================================
CUMULATIVE DEFAULT RATES BY PRODUCT (PROGRAM)
================================================================================
program  num_defaults  default_rate  num_loans  prepay_rate  avg_amount  total_amount
     P1           119      1.163587      10227    93.947394 4273.229827   43702321.44
     P2           655      6.711066       9760    86.690574 3220.149433   31428658.47
     P3          2280     35.736677       6380    57.915361 2485.876135   15859889.74

```

![Chart 11](/home/ycnin/moore/charts/chart_11.png)

*Chart 11: [Description based on analysis]*

```

================================================================================
CUMULATIVE DEFAULT RATES BY TERM BUCKET
================================================================================
 term program  num_defaults  default_rate  num_loans  prepay_rate  avg_approved_amount  avg_loan_age_years  avg_loan_age_months  annualized_default_rate
    3      P1             6      0.143266       4188    98.424069          4494.682354            0.267991             3.215892                 0.534594
    3      P2            37      2.315394       1598    92.553191          2794.847196            0.272231             3.266773                 8.505252
    3      P3           111     15.226337        729    78.737997          2003.450123            0.312134             3.745605                48.781449
    6      P1            15      0.860585       1743    95.983936          4017.758084            0.481824             5.781888                 1.786099
    6      P2           104      4.657412       2233    91.312136          2835.034326            0.485479             5.825744                 9.593442
    6      P3           344     25.350037       1357    69.933677          2066.715122            0.508044             6.096526                49.897343
   12      P1            47      1.806997       2601    93.925413          4260.959823            0.854538            10.254454                 2.114590
   12      P2           246      7.495430       3282    87.812310          3126.186246            0.845817            10.149798                 8.861768
   12      P3          1478     43.611685       3389    52.375332          2607.226480            0.792486             9.509835                55.031474
   24      P1            39      3.145161       1240    89.193548          3985.439831            1.390093            16.681115                 2.262555
   24      P2           199      9.812623       2028    84.615385          3966.670266            1.314509            15.774103                 7.464861
   24      P3            83     26.433121        314    68.471338          3504.165637            1.251642            15.019710                21.118747
   36      P1             1     10.000000         10    80.000000          7951.147000            2.186174            26.234086                 4.574202

```

```
program    P1    P2     P3
term                      
3        0.53  8.51  48.78
6        1.79  9.59  49.90
12       2.11  8.86  55.03
24       2.26  7.46  21.12
36       4.57   NaN    NaN

```

![Chart 12](/home/ycnin/moore/charts/chart_12.png)

*Chart 12: [Description based on analysis]*

