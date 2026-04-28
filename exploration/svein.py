import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Small
columns = ["NewsID", "Category", "SubCategory", "Title", "Abstract", "Url", "TitleEntities", "AbstractEntities"]

df_train = pd.read_csv("./small/MINDsmall_train/news.tsv", sep="\t", header=None, names=columns)
df_dev = pd.read_csv("./small/MINDsmall_dev/news.tsv", sep="\t", header=None, names=columns)

train_counts = df_train["Category"].value_counts()
dev_counts = df_dev["Category"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].bar(train_counts.index, train_counts.values)
axes[0].set_title("Train")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(dev_counts.index, dev_counts.values)
axes[1].set_title("Val")
axes[1].set_xlabel("Category")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Large
columns = ["NewsID", "Category", "SubCategory", "Title", "Abstract", "Url", "TitleEntities", "AbstractEntities"]

paths = {
    "Train": "./large/MINDlarge_train/news.tsv",
    "Val": "./large/MINDlarge_dev/news.tsv",
    "Test": "./large/MINDlarge_test/news.tsv",
}

for name, path in paths.items():
    df = pd.read_csv(path, sep="\t", header=None, names=columns)
    counts = df["Category"].value_counts()

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.title(name)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


news_columns = ["NewsID", "Category", "SubCategory", "Title", "Abstract", "Url", "TitleEntities", "AbstractEntities"]
beh_columns = ["ImpressionID", "UserID", "Time", "History", "Impressions"]

news_df = pd.read_csv("./large/MINDlarge_train/news.tsv", sep="\t", header=None, names=news_columns)
newsid_to_category = dict(zip(news_df["NewsID"], news_df["Category"]))

beh_df = pd.read_csv("./large/MINDlarge_train/behaviors.tsv", sep="\t", header=None, names=beh_columns)

category_clicks = defaultdict(int)

for impressions in beh_df["Impressions"].dropna():
    for imp in impressions.split():
        newsid, clicked = imp.rsplit("-", 1)
        if clicked == "1" and newsid in newsid_to_category:
            category = newsid_to_category[newsid]
            category_clicks[category] += 1

sorted_categories = sorted(category_clicks.items(), key=lambda x: x[1], reverse=True)
categories, counts = zip(*sorted_categories)

plt.figure()
plt.bar(categories, counts)
plt.xlabel("Category")
plt.ylabel("Number of Clicks")
plt.title("Clicked Categories")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()