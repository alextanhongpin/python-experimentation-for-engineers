```python
import numpy as np


class Exchange:
    COST = 0

    def sample(self):
        return self.COST + np.random.normal()


class TimeOfDayEffectWrapper:
    def __init__(self, exchange: Exchange):
        self.exchange = exchange

    def sample(self, tod="afternoon"):
        bias = 2.5 if tod == "morning" else 0.0
        return self.exchange.sample() + bias


class ASDAQ(Exchange):
    COST = 12


class BYSE(Exchange):
    COST = 10


def pprint(value):
    print("${:.2f}".format(value))


asdaq = ASDAQ()
pprint(asdaq.sample())

asdaq = TimeOfDayEffectWrapper(asdaq)
pprint(asdaq.sample("morning"))
pprint(asdaq.sample("afternoon"))

byse = TimeOfDayEffectWrapper(BYSE())
pprint(byse.sample())
```

    $12.73
    $14.94
    $12.88
    $10.52



```python
pprint(np.mean([byse.sample() for _ in range(100)]))
pprint(np.mean([asdaq.sample() for _ in range(100)]))
```

    $9.99
    $12.01

