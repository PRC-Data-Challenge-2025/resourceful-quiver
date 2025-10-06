# What To Do

## Heuristic Rules
1. People have developed amazing ML models
2. Our job is to do feature engineering, focus on it!
3. Last year's winner basically translate trajectory data into tabular data.
4. Make it easily transferrable, it will save you time later in the process

## What to do
1. Check the "number of data in segment" distribution, do we have outliers?
2. Find relevant features for the data

## Relevant features
1. Number of data in segment
2. Nb Climb/Nb Cruise/Nb Descent -> Traffic/OpenAP
3. Min, max, fuel per ac type? But this might be irrelevant
4. Origin-Dest pair
5. Length of segment relative to great circle dist from Origin Dest pair 
6. Length of segment relative to great circle distance from segment
7. Segment distance
8. Sum of heading change (don't forget to wrap the degree)
9. Has holding or not
10. Mean/min/max distance from origin
11. Mean/min/max distance from dest
12. Stdev distance from origin
13. Stdev distance from dest
14. Mean/min/max ground speed
15. Stdev ground speed
16. Mean/min/max acceleration
17. Stdev acceleration
18. Initial and final altitude
19. Mean/median/min/max altitude
20. Stdev altitude
21. Mean/min/max/median vertical speed
22. Stdev vertical speed
23. Time in climb/cruise/descent
24. Time in segment
25. Mean/median/min/max mach number
26. Mean/median/min/max TAS
27. Mean/median/min/max CAS
28. Mean/median/min/max change in potential energy
29. Mean/median/min/max change in kinetic energy
30. Aircraft type
31. Take off weight (Hard one, but might be worth it)
32. Mean/min/max altitude from origin
33. Mean/min/max altitude from dest
34. Maximum fuel capacity