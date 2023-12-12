# MVDataProcessing
## Missing Data Imputation Method for Medium Voltage Distribution Network Feeders

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
│   └── MVDataProcessing/
│       ├── __init__.py
│		├── Clean.py
│		├── Fill.py
│		├── Util.py
│       ├── Example.py
└── tests/
    └── test_lib.py
```
### Installation

```bash
pip install MVDataProcessing
```

### Example of performance and capability

#### Example 1
```python
import MVDataProcessing as mvp
    
#Simple process example. Check the console for explanation.  
#Run with plot=True to visualize plot and then with plot=False to check performance.

mvp.ShowExampleSimpleProcess(plot=True)
```

#### Example 2

```python
import MVDataProcessing as mvp

#NSSC process example. Check the console for explanation.  
#Run with plot=True to visualize plot and then with plot=False to check performance.   
 
mvp.ShowExampleNSSCProcess(plot=True)
```

### Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Licence

Copyright <2023> <JMBacalhau>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.