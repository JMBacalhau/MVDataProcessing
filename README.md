# MVDataProcessing
##Missing Data Imputation Method for Medium Voltage Distribution Network Feeders

### Abstract

The energy sector’s investment aims to ensure a continuous, reliable, and quality
supply of electrical energy imposed by the electricity regulatory agency with maximum
economic-financial balance. The analysis of missing data and outliers is made on the three-phase voltage, current, and
power factor of 459 time series of real feeders shows that most missing data are three-phase, however, with a significant amount of single
and dual-phase loss that can be filled by the proportion between phases. Hence, the
challenge is to fill multiple weeks of missing three-phase data, and for that, the use of the
standard curve for each day of the week is proposed.
Therefore, this library proposes a method of
preprocessing, and missing data imputation using the unbalanced characteristic between
phases, interpolation, and the normalized scaled standard weekday curve. 

Article published on [Link](https://www.sba.org.br/open_journal_systems/index.php/cba/article/view/968).


### Structure
```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.cfg
├── src/
│   └── example_package/
│       ├── __init__.py
│       └── example.py
└── tests/
```
