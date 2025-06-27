
### CosineAnnealingWarmRestarts

```yaml
scheduler:
  name: cosine_warm_restarts
  params:
    T_0: 10
    T_mult: 1
    eta_min: 0.000001
```


### CosineAnnealingLR

```yaml
scheduler:
  name: cosine
  params:
    T_max: 10
    eta_min: 0.000001
```