# Logistics Domain Comparison (exp2_static_knowledge, run: 20260220_144634)

| Seed | Config | PRECEPT P1 | PRECEPT Pt | FR P1 | FR Pt | ExpeL P1 | ExpeL Pt | Conflicts | Static Reliability | Dynamic Reliability |
|------|--------|------------|------------|-------|-------|----------|----------|-----------|--------------------|---------------------|
| 42 | with_sk | 1.00 | 1.00 | 0.75 | 1.00 | 0.50 | 1.00 | 142/142 | 0.0329 | 0.98 |
| 42 | no_sk | 1.00 | 1.00 | 0.75 | 1.00 | 0.50 | 1.00 | 0/0 | 0.50 | 0.625 |
| 123 | with_sk | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 179/179 | 0.0265 | 0.984 |
| 123 | no_sk | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/0 | 0.50 | 0.625 |
| 456 | with_sk | 0.75 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 185/185 | 0.0256 | 0.984 |
| 456 | no_sk | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0/0 | 0.50 | 0.625 |
| 789 | with_sk | 0.75 | 1.00 | 0.50 | 0.75 | 1.00 | 1.00 | 114/114 | 0.0403 | 0.975 |
| 789 | no_sk | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0/0 | 0.50 | 0.625 |
| 999 | with_sk | 1.00 | 1.00 | 0.50 | 0.75 | 0.75 | 0.75 | 73/73 | 0.0602 | 0.963 |
| 999 | no_sk | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0/0 | 0.50 | 0.625 |
| 1337 | with_sk | 0.75 | 1.00 | 0.75 | 0.75 | 0.75 | 0.75 | 98/98 | 0.0463 | 0.972 |
| 1337 | no_sk | 1.00 | 1.00 | 0.75 | 0.75 | 1.00 | 1.00 | 0/0 | 0.50 | 0.625 |
| 2024 | with_sk | 1.00 | 1.00 | 0.25 | 0.50 | 0.75 | 1.00 | 81/81 | 0.0549 | 0.966 |
| 2024 | no_sk | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 | 1.00 | 0/0 | 0.50 | 0.625 |
| 3141 | with_sk | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 82/82 | 0.0543 | 0.967 |
| 3141 | no_sk | 0.75 | 0.75 | 0.50 | 1.00 | 0.50 | 0.75 | 0/0 | 0.50 | 0.625 |
| 7777 | with_sk | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 164/164 | 0.0287 | 0.983 |
| 7777 | no_sk | 1.00 | 1.00 | 0.50 | 0.75 | 0.75 | 1.00 | 0/0 | 0.50 | 0.625 |
| 8888 | with_sk | 1.00 | 1.00 | 0.75 | 0.75 | 0.75 | 1.00 | 118/118 | 0.0391 | 0.976 |
| 8888 | no_sk | 1.00 | 1.00 | 0.50 | 0.75 | 0.75 | 1.00 | 0/0 | 0.50 | 0.625 |

**Notes:**
- **P1** = first_try_success_rate; **Pt** = success_rate (overall)
- **FR** = Full Reflexion; **ExpeL** = ExpeL
- **Conflicts** = conflicts_detected / conflicts_resolved (from conflict_resolution.stats.summary)
- **Static Reliability** = conflict_resolution.stats.reliability.static_kb
- **Dynamic Reliability** = conflict_resolution.stats.reliability.dynamic_experience
- **with_sk** = static knowledge enabled; **no_sk** = static knowledge disabled (conflict resolution disabled, hence 0 conflicts and default reliability values)
