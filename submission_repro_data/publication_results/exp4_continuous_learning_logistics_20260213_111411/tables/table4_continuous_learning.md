## Table 4: Cross-Episode Continuous Learning Results

**Domain**: logistics | **N**: 5 | **β**: 1 | **Encounters**: 4 | **Seeds**: 10  

| Encounter | PRECEPT P₁ | ExpeL P₁ | FR P₁ | PRECEPT Steps | ExpeL Steps | FR Steps |
|-----------|-----------|----------|-------|--------------|------------|----------|
| 1st | **44.8%**±10.7 | 36.6%±12.7 | 38.8%±8.3 | **3.24**±0.26 | 4.19±0.48 | 4.12±0.26 |
| 2nd | **82.5%**±12.1** | 65.0%±19.2 | 62.5%±15.2 | **2.38**±0.26 | 3.05±0.57 | 3.40±0.63 |
| 3rd | **87.5%**±9.4** | 72.5%±13.2 | 60.0%±12.5 | **2.25**±0.19 | 2.80±0.42 | 3.50±0.58 |
| 4th | **100.0%**±0.0** | 85.0%±12.5 | 67.5%±17.0 | **2.00**±0.00 | 2.50±0.38 | 3.20±0.64 |

| Δ (1→4) | **+55.2pp** | +48.4pp | +28.7pp | **-1.24** | -1.69 | -0.92 |

### Statistical Significance (PRECEPT vs Full Reflexion at 4th encounter)

- **PRECEPT vs Full Reflexion**: p=0.0019 (Cohen's d=1.37) **
- **PRECEPT vs ExpeL**: p=0.0239 (Cohen's d=0.86) *

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*