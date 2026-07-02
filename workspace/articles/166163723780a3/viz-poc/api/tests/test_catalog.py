from app.catalog import DatasetCatalog
from app.config import CATALOG_PATH


def test_catalog_loads_sales_dataset() -> None:
    catalog = DatasetCatalog.load(CATALOG_PATH)

    assert catalog.has_dataset("sales")

    sales = catalog.get_dataset("sales")
    assert "region" in sales.dimensions
    assert "revenue" in sales.metrics
