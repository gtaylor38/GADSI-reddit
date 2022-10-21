# Project 3: Reddit Post Analysis

### Problem Statement:
Data Science Question: What characteristics of a sports-related post on Reddit are most predictive of the overall interaction on a thread (as measured by number of comments)?
- Success: Significant improvement over the baseline in predicting whether a post on reddit will exceed the median number of comments

General Question: Why do people care about sports so much, and are there any aspects of "sport" that span many sports and generate interest in them all?
- Success: Finding specific aspects of sport (word-related) that are generally important in predicting engagement

### Data Dictionary:
Data Collected for each Post:
- title (string): title text of each post
- subreddit (string): subreddit in which each post was made
- score (int): # of upvotes minus # of downvotes
- author_flair_text (string): text describing post author's flair, a sort of affiliation
- link_flair_text (string): text describing post link's flair, a sort of affiliation
- created_utc (float): Unix epoch time in seconds
- num_comments: # of comments on the post


### Python Libraries Used:
**Data Collection:**
import pandas as pd
import requests
import requests.auth
import json
import time
from time import sleep

**Data Cleaning:**
import pandas as pd

**Preliminary EDA:**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

**Text EDA:**
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

**Modeling:**
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

### Background:
Reddit is a news- and discussion-oriented website on which people can join communities ("subreddits") related to various topics. There are tens of thousands of subreddits, many of which are related to sports and have membership over one million. People may post anything that conforms to Reddit's guidelines and the guidelines of the subreddit in which they post it. Others may comment on posts, potentially generating discussion.

Sports of all kinds are very appealing to me personally, as they clearly are to many others. Theories are often put forward about why people enjoy sports so much, especially those related to narrative, community, and vicariousness. I wanted to know what the data had to say about this question.

Some framing of this project was inspired by Freakonomics Radio Episode 506: [Sportswashing](https://freakonomics.com/podcast/what-is-sportswashing-and-does-it-work/'). The episode in part discusses the strange power of sports to influence public opinion as well as the outsize interest in sports relative to the size of the industry.

### Analysis:
There are clear patterns to what type of words commonly appear in post titles according to the particular sports subreddit. In particular, competition words (vs, game, fight, race) are very common overall and especially where applicable. Mentions of specific players or athletes are also common across subreddits, and community words (those related to reviews, advice, or discussion) are common in sports that are more focused on personal hobbies and less on large competitions. These insights came primarily from EDA.

Ultimately, model performance was underwhelming. It was difficult to predict whether a given post would surpass the median number of comments on all posts in the training set (22.0), with a best model accuracy barely above 70%. It might be the case that there is more to be learned from the frequency of words that appear in posts than from their ability to predict engagement, i.e., that community members on Reddit might engage with each other more by making original posts than by commenting on the posts of others.

### Conclusions and Recommendations:
The problem statement's data science question is answered somewhat: posts that are more descriptive, explicitly encourage discussion, and are related to current sporting events do the best job of generating engagement on Reddit. The more general question concerning why people care about sports so much and whether there are common threads spanning and generating interest in many sports is mostly unanswered, though EDA on the frequency of certain words was enlightening. For further work on this topic, I have several recommendations:
- Perform thorough analysis of individual sport subreddits to find what those particular fans / enthusiasts like to engage with.
- Perform analysis by first classifying sports as game vs. hobby, team sport vs. individual, or a categorization in a similar vein, and analyzing each category.
- Gather more data and use a custom list of stop words when vectorizing post titles.