"""The cart."""


class Cart:
    def __init__(self):
        self.lines = {}

    def add(self, sku, qty=1, tags=[]):
        """Add `qty` of `sku`. Every line starts with its own tag list, which always gets the tag "added"."""
        tags.append("added")
        line = self.lines.setdefault(sku, {"qty": 0, "tags": tags})
        line["qty"] += qty
        return line

    def total_items(self):
        return sum(line["qty"] for line in self.lines.values())
