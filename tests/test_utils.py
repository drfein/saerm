from saerm.utils import slugify


def test_slugify():
    assert slugify(["Model", "Layer-1"]) == "model-layer-1"
