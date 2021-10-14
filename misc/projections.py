"""
Projections
To add a custom projection, create a method accepting (x, y, z) Cartesian coordinates and add an entry to the
parse_projection method
"""


def parse_projection(name: str):
    if name == "standard":
        return standard_proj

    raise ValueError("Unknown projection")


def standard_proj(x, y, z):
    return x, y
