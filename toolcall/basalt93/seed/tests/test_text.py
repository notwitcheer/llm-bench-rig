from shopcart.text import slugify


def test_slugify_collapses_and_strips():
    assert slugify("  Crème Brûlée -- Deluxe!  ") == "creme-brulee-deluxe"
