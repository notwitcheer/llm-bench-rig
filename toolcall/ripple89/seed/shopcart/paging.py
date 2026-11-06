"""Pagination for product lists."""


def paginate(items, page, per_page):
    """Return the items on `page` (pages start at 1)."""
    start = page * per_page
    return items[start:start + per_page]
