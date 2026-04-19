# DM (Data Mining)

This directory contains the data mining pipeline for the NASA Meteorite Landings project.

## Subdirectories

- `data/raw/`: original source files
- `data/processed/`: cleaned and transformed datasets
- `data/external/`: external reference datasets (optional)
- `notebooks/`: staged notebooks for cleaning, EDA, clustering, and evaluation
- `src/`: reusable Python modules
- `outputs/`: generated figures, tables, and models
- `tests/`: basic tests for preprocessing and future modules

## Notes

- Prefer reusable logic in `src/` rather than duplicating code across notebooks.
- Use `pathlib` for all filesystem paths.
