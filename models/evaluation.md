# Evaluation (kNN, own + HaGRID (points=none), extra features=True)

Leave-one-person-out: every frame is predicted by a model that never saw that person.

## Confusion matrix (all people, frame counts; rows = true, columns = predicted)

```
pred         point_right  point_left  open_palm  none
gesture                                              
point_right        10111          41         51   757
point_left            32        9419         35  1055
open_palm             72           3       3028    31
none                 210         382        267  3913
```

## Where do the 'none' frames go? (% of each activity's frames)

A 'none' frame predicted as a point is a potential false trigger.

```
pred                none  open_palm  point_left  point_right
note                                                        
face, cup           79.0        7.0        10.0          4.0
pointing at screen  86.0        9.0         3.0          2.0
resting             86.0        2.0         7.0          5.0
talking             77.0        5.0        12.0          6.0
```

## Confidence threshold trade-off (per frame, before the K-frames smoothing)

```
threshold  none->point (false-trigger rate)  points recognized
  0.0        12.4%                      90.8%
  0.6        10.9%                      89.7%
  0.7         8.0%                      86.9%
  0.8         6.6%                      85.1%
  0.9         3.8%                      79.3%
  1.0         2.5%                      71.8%
```

Precision of point_left/right looks high (~0.97) only because our test data has
about twice as many pointing frames as 'none' frames. In a real talk almost every
frame is 'none', so the none->point rate is the number that predicts false triggers.
