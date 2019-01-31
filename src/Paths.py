import os

root = os.path.realpath(
        os.path.join(
                os.path.dirname(
                        os.path.realpath(__file__)
                ),
                "../"
        )
)
raw = os.path.join(root, '../github-selective-scraper/data/raw')
raw_sample = os.path.join(root, '../github-selective-scraper/data/raw_sample')
processed = os.path.join(root, 'data/processed')