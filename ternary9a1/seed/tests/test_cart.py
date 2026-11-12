from shopcart.cart import Cart


def test_total_items():
    c = Cart()
    c.add("apple", 2)
    c.add("pear")
    assert c.total_items() == 3


def test_lines_do_not_share_tags():
    first, second = Cart(), Cart()
    first.add("apple")
    line = second.add("pear")
    assert line["tags"] == ["added"]
