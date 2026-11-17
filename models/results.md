# Training results

Own data: 29407 frames, people: melis, mina, zeynep. HaGRID: 114000 hands from 19707 people.

Every number: leave-one-person-out over our 3 people (3 folds). P = precision, R = recall. 'range' = max - min over the folds.

## 1. Model comparison (train: own + HaGRID palm/none, test: held-out person)

### kNN  (2 s)
```
         acc  macroF1  point_right P  point_right R  point_left P  point_left R  open_palm P  open_palm R  none P  none R  point R (index)  point R (flat)
melis   0.89     0.89           0.99           0.89          0.92          0.88         0.99         1.00    0.65    0.84             0.90            0.84
mina    0.85     0.82           0.96           0.89          0.96          0.83         0.74         0.92    0.57    0.77             0.86            0.88
zeynep  0.96     0.95           0.97           0.99          0.98          0.97         0.95         0.97    0.89    0.85             0.99            0.96
MEAN    0.90     0.89           0.97           0.92          0.96          0.89         0.89         0.96    0.70    0.82             0.91            0.89
range   0.11     0.12           0.03           0.10          0.06          0.14         0.25         0.08    0.32    0.08             0.13            0.12
```
### RandomForest  (41 s)
```
         acc  macroF1  point_right P  point_right R  point_left P  point_left R  open_palm P  open_palm R  none P  none R  point R (index)  point R (flat)
melis   0.85     0.86           1.00           0.81          0.97          0.78         0.99         0.98    0.53    0.95             0.81            0.74
mina    0.78     0.78           0.96           0.78          0.98          0.70         0.88         0.87    0.43    0.91             0.73            0.78
zeynep  0.93     0.92           0.98           0.97          0.99          0.88         0.98         0.95    0.72    0.93             0.98            0.78
MEAN    0.85     0.85           0.98           0.86          0.98          0.79         0.95         0.93    0.56    0.93             0.84            0.77
range   0.15     0.14           0.04           0.19          0.03          0.18         0.11         0.11    0.30    0.03             0.25            0.04
```
### MLP  (22 s)
```
         acc  macroF1  point_right P  point_right R  point_left P  point_left R  open_palm P  open_palm R  none P  none R  point R (index)  point R (flat)
melis   0.75     0.78           0.98           0.55          0.96          0.76         0.99         0.99    0.40    0.94             0.64            0.69
mina    0.84     0.83           0.96           0.85          0.99          0.80         0.85         0.89    0.53    0.91             0.83            0.82
zeynep  0.93     0.91           0.97           0.98          0.99          0.90         0.96         0.86    0.72    0.90             0.99            0.80
MEAN    0.84     0.84           0.97           0.79          0.98          0.82         0.93         0.92    0.55    0.92             0.82            0.77
range   0.18     0.13           0.02           0.43          0.03          0.13         0.15         0.13    0.32    0.03             0.34            0.13
```

Best model by mean macro F1: **kNN**

## 2. Training data comparison (kNN)

### own only
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.92     0.91           0.97          0.86             0.97            0.99
mina    0.86     0.80           0.95          0.96             0.91            0.91
zeynep  0.95     0.94           0.94          0.97             0.99            0.98
MEAN    0.91     0.88           0.96          0.93             0.96            0.96
range   0.09     0.14           0.03          0.10             0.08            0.08
```
### own + HaGRID, points=none
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.89     0.89           0.99          0.92             0.90            0.84
mina    0.85     0.82           0.96          0.96             0.86            0.88
zeynep  0.96     0.95           0.97          0.98             0.99            0.96
MEAN    0.90     0.89           0.97          0.96             0.91            0.89
range   0.11     0.12           0.03          0.06             0.13            0.12
```
### HaGRID only, points=rot_one
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.68     0.72           1.00          0.99             0.73            0.01
mina    0.59     0.64           0.98          1.00             0.70            0.00
zeynep  0.79     0.81           1.00          1.00             0.98            0.00
MEAN    0.69     0.73           0.99          0.99             0.80            0.00
range   0.20     0.17           0.01          0.01             0.28            0.01
```
### own + HaGRID, points=rot_one
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.90     0.90           0.99          0.92             0.92            0.84
mina    0.85     0.82           0.96          0.96             0.86            0.88
zeynep  0.96     0.95           0.97          0.98             0.99            0.96
MEAN    0.90     0.89           0.97          0.96             0.92            0.89
range   0.11     0.12           0.04          0.06             0.13            0.12
```
### HaGRID only, points=rot_palm
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.55     0.61           0.99          0.93             0.22            0.83
mina    0.41     0.49           0.92          0.92             0.04            0.79
zeynep  0.39     0.49           0.93          0.94             0.00            0.85
MEAN    0.45     0.53           0.95          0.93             0.09            0.82
range   0.15     0.12           0.07          0.02             0.22            0.07
```
### own + HaGRID, points=rot_palm
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.91     0.91           0.99          0.92             0.92            0.90
mina    0.86     0.83           0.96          0.96             0.86            0.91
zeynep  0.96     0.95           0.96          0.98             0.99            0.96
MEAN    0.91     0.89           0.97          0.96             0.92            0.92
range   0.10     0.12           0.04          0.06             0.13            0.06
```
### Overview (means over the 3 people)
```
                                acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
own only                       0.91     0.88           0.96          0.93             0.96            0.96
own + HaGRID, points=none      0.90     0.89           0.97          0.96             0.91            0.89
HaGRID only, points=rot_one    0.69     0.73           0.99          0.99             0.80            0.00
own + HaGRID, points=rot_one   0.90     0.89           0.97          0.96             0.92            0.89
HaGRID only, points=rot_palm   0.45     0.53           0.95          0.93             0.09            0.82
own + HaGRID, points=rot_palm  0.91     0.89           0.97          0.96             0.92            0.92
```

Best data setup by mean macro F1: **own + HaGRID, points=none**

## 3. Feature comparison

### extra features = False
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.87     0.87           0.99          0.92             0.88            0.80
mina    0.82     0.79           0.95          0.96             0.81            0.88
zeynep  0.96     0.94           0.97          0.98             0.99            0.96
MEAN    0.88     0.87           0.97          0.95             0.89            0.88
range   0.14     0.15           0.04          0.06             0.17            0.15
```
### extra features = True
```
         acc  macroF1  point_right P  point_left P  point R (index)  point R (flat)
melis   0.89     0.89           0.99          0.92             0.90            0.84
mina    0.85     0.82           0.96          0.96             0.86            0.88
zeynep  0.96     0.95           0.97          0.98             0.99            0.96
MEAN    0.90     0.89           0.97          0.96             0.91            0.89
range   0.11     0.12           0.03          0.06             0.13            0.12
```
## 4. Why a random split overestimates (kNN, own data only)

```
                            acc  macroF1  point_right P  point_left P
random frames (80/20)     0.960    0.951          0.967         0.960
random takes (5-fold)     0.915    0.885          0.933         0.931
held-out person (3-fold)  0.910    0.880          0.960         0.930
```

Frames of one take are nearly identical, so with a random frame split the test
set contains near-copies of training frames: the model is tested on things it
has (almost) memorized. Holding out whole people is what the live clicker faces.

## Saved model

gesture_model.joblib: kNN, own=True, HaGRID points=none, features={'extra': True}, trained on 103407 rows.
