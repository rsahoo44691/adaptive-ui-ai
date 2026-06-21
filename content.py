"""Sample content corpus for the adaptive UI demo.

Each article is a small document the adaptive engine learns to rank for the
current user. The corpus is intentionally spread across distinct topics so
that adaptation (re-ranking by learned preference) is easy to observe.

The dataset is synthetic and written for this assignment; no external content
is reproduced.
"""

# reading_time is in minutes; items >= 6 minutes are treated as "long reads"
# and are used to demonstrate accessibility-oriented adaptation.
ARTICLES = [
    {
        "id": "a01",
        "title": "Getting Started with Neural Networks",
        "category": "AI & Machine Learning",
        "reading_time": 5,
        "body": (
            "A neural network learns patterns from data by adjusting weights "
            "through backpropagation. This introduction covers neurons, layers, "
            "activation functions, and how gradient descent trains a model to "
            "make accurate predictions."
        ),
    },
    {
        "id": "a02",
        "title": "Transformers and Attention Explained",
        "category": "AI & Machine Learning",
        "reading_time": 8,
        "body": (
            "The transformer architecture uses self-attention to weigh the "
            "importance of every token in a sequence. This long read explains "
            "queries, keys, values, multi-head attention, and why transformers "
            "replaced recurrent models for language tasks."
        ),
    },
    {
        "id": "a03",
        "title": "Designing Accessible Interfaces",
        "category": "Design & Accessibility",
        "reading_time": 6,
        "body": (
            "Accessible design ensures interfaces work for people with diverse "
            "abilities. Learn about color contrast, keyboard navigation, screen "
            "reader support, scalable text, and the WCAG guidelines that make a "
            "user interface inclusive and usable for everyone."
        ),
    },
    {
        "id": "a04",
        "title": "Color Theory for User Interfaces",
        "category": "Design & Accessibility",
        "reading_time": 4,
        "body": (
            "Color communicates meaning and guides attention in an interface. "
            "This article explores contrast, palettes, accessibility of color "
            "choices, and how thoughtful color design improves the overall user "
            "experience of a product."
        ),
    },
    {
        "id": "a05",
        "title": "Healthy Habits for Remote Workers",
        "category": "Health & Wellness",
        "reading_time": 5,
        "body": (
            "Working from home blurs the line between rest and productivity. "
            "These habits cover ergonomic posture, regular movement breaks, "
            "hydration, sleep routines, and setting boundaries to protect your "
            "mental and physical health."
        ),
    },
    {
        "id": "a06",
        "title": "A Beginner's Guide to Meditation",
        "category": "Health & Wellness",
        "reading_time": 4,
        "body": (
            "Meditation trains attention and reduces stress through simple "
            "breathing exercises. This guide explains mindfulness, body scans, "
            "and short daily practices that improve focus, calm, and emotional "
            "wellbeing over time."
        ),
    },
    {
        "id": "a07",
        "title": "Understanding the Stock Market",
        "category": "Finance & Investing",
        "reading_time": 7,
        "body": (
            "The stock market lets investors buy shares of public companies. "
            "This long read covers indexes, diversification, risk tolerance, "
            "compound growth, and the difference between investing for the long "
            "term and short term trading."
        ),
    },
    {
        "id": "a08",
        "title": "Budgeting Basics for Students",
        "category": "Finance & Investing",
        "reading_time": 4,
        "body": (
            "A budget tracks income against expenses so money lasts the month. "
            "Learn the fifty thirty twenty rule, how to cut recurring costs, "
            "build an emergency fund, and save consistently even on a small "
            "student income."
        ),
    },
    {
        "id": "a09",
        "title": "Quick Weeknight Pasta Recipes",
        "category": "Food & Cooking",
        "reading_time": 3,
        "body": (
            "Fast pasta dinners rely on a few pantry staples and bold flavor. "
            "These recipes use garlic, olive oil, tomatoes, and fresh herbs to "
            "put a satisfying meal on the table in under thirty minutes on a "
            "busy weeknight."
        ),
    },
    {
        "id": "a10",
        "title": "The Science of Baking Bread",
        "category": "Food & Cooking",
        "reading_time": 6,
        "body": (
            "Bread baking is chemistry you can eat. This long read explains how "
            "yeast ferments dough, why gluten gives structure, the role of "
            "hydration and temperature, and how time develops flavor in a "
            "homemade loaf."
        ),
    },
    {
        "id": "a11",
        "title": "Exploring National Parks on a Budget",
        "category": "Travel & Outdoors",
        "reading_time": 5,
        "body": (
            "National parks offer breathtaking scenery without a high price tag. "
            "This article shares tips on annual passes, off season travel, "
            "camping, scenic hiking trails, and planning an affordable outdoor "
            "adventure."
        ),
    },
    {
        "id": "a12",
        "title": "Reinforcement Learning in Plain English",
        "category": "AI & Machine Learning",
        "reading_time": 7,
        "body": (
            "Reinforcement learning trains an agent to make decisions by "
            "rewarding good actions. This long read introduces agents, states, "
            "rewards, exploration versus exploitation, and how the same ideas "
            "power game playing and recommendation systems."
        ),
    },
]


def get_articles():
    """Return a fresh copy of the article list."""
    return [dict(a) for a in ARTICLES]


def categories():
    """Return the distinct categories present in the corpus."""
    seen = []
    for a in ARTICLES:
        if a["category"] not in seen:
            seen.append(a["category"])
    return seen
