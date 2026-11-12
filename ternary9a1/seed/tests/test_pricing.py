from shopcart.pricing import apply_discount


def test_discount_rounds_half_up():
    assert apply_discount(1005, 10) == 905
    assert apply_discount(2000, 25) == 1500
