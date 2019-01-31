# github-selective-scraper

Scrape subsets of github repositories. Warning - this is a work in progress.

## Installation
First, make sure you have anaconda installed. Then
```
git clone git@github.com:ajthinking/github-selective-scraper.git
cd github-selective-scraper
conda env create
conda activate github-selective-scraper
source activate
python src/examples/migration-analysis/01_scrape.py
```

## Todo
- [ ] Parametirize the query string
- [ ] Fix try-except blocks to throw real errors

## Disclaimer
Please note to only use this repository in agreement with github API terms of service (scraping only ok for educational purposes).

## License
MIT