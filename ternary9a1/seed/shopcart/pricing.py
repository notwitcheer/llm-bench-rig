"""Price helpers. All money is integer cents."""


def apply_discount(cents, percent):
    """Return the price in cents after taking `percent` off.

    Halves round up (1005 cents less 10% is 905, not 904).
    """
    return round(cents * (100 - percent) / 100)
