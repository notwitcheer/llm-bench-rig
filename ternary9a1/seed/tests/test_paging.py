from shopcart.paging import paginate


def test_first_page_is_page_one():
    items = list(range(10))
    assert paginate(items, 1, 3) == [0, 1, 2]
    assert paginate(items, 4, 3) == [9]
