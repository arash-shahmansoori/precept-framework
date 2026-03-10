## Table 4: Cross-Episode Continuous Learning Results

**Domain**: integration | **N**: 5 | **β**: 1 | **Encounters**: 4 | **Seeds**: 9  

| Encounter | PRECEPT P₁ | ExpeL P₁ | FR P₁ | PRECEPT Steps | ExpeL Steps | FR Steps |
|-----------|-----------|----------|-------|--------------|------------|----------|
| 1st | **30.3%**±12.6* | 18.5%±9.0 | 16.8%±9.2 | **3.95**±1.19 | 4.38±1.28 | 4.45±1.31 |
| 2nd | **51.4%**±19.6** | 21.1%±8.5 | 17.7%±11.2 | **3.11**±1.05 | 4.43±1.29 | 4.49±1.34 |
| 3rd | **63.5%**±22.1*** | 27.2%±14.2 | 23.5%±19.2 | **2.61**±0.92 | 4.17±1.28 | 4.24±1.39 |
| 4th | **70.4%**±23.8*** | 30.6%±15.7 | 26.9%±17.5 | **2.33**±0.87 | 4.04±1.25 | 4.22±1.35 |

| Δ (1→4) | **+40.1pp** | +12.0pp | +10.1pp | **-1.62** | -0.35 | -0.23 |

### Statistical Significance (PRECEPT vs Full Reflexion at 4th encounter)

- **PRECEPT vs Full Reflexion**: p=0.0007 (Cohen's d=1.80) ***
- **PRECEPT vs ExpeL**: p=0.0016 (Cohen's d=1.55) **

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*