import statsmodels
import statsmodels.stats.api as sm
import math

# Ensure that the version is valid.
assert statsmodels.__version__ == "0.14.4"


def minimum_sample_size(
    conversion_rate=0.02, expected_improvement=0.15, power=0.8, alternative="two-sided"
):
    """
    Returns the similar value as https://abtestguide.com/abtestsize/
    """
    p1 = conversion_rate
    p2 = (1 + expected_improvement) * p1
    size = sm.samplesize_proportions_2indep_onetail(
        p2 - p1, p1, power, alternative=alternative
    )
    return math.ceil(size)


if __name__ == "__main__":
    assert minimum_sample_size() == 36693
    assert minimum_sample_size(alternative="larger") == 28903
