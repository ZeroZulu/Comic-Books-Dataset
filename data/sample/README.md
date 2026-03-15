# Sample Data

This folder is reserved for small sample files used in testing and documentation.

## Full Dataset

Download the full dataset from Kaggle:  
👉 [Comic Books Dataset (10,000 entries)](https://www.kaggle.com/datasets/rudrakumargupta/comic-books-dataset-10000-entries)

Save the downloaded CSV as:
```
data/comic_books_10000_dataset.csv
```

The full dataset file is excluded from version control via `.gitignore`.

## Column Reference

| Raw Column Name     | Cleaned Name | Type   |
|---------------------|--------------|--------|
| Title               | title        | string |
| Studio/Publisher    | publisher    | string |
| Release Year        | year         | int    |
| Page Count          | pages        | int    |
| Rating (out of 10)  | rating       | float  |
| Theme (Color Style) | theme        | string |
| Country of Origin   | country      | string |
| Volume Count        | volume_count | int    |
| Genre               | genre        | string |
| Format              | format       | string |
| Status              | status       | string |
| Language            | language     | string |
| Age Rating          | age_rating   | string |
| Awards              | awards       | string |
| Writer              | writer       | string |
| Artist              | artist       | string |
| comic_id            | comic_id     | string |
