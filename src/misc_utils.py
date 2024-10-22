import pandas as pd
import numpy as np
def engagement_score(likes: pd.Series, retweets: pd.Series, followers: pd.Series) -> pd.Series:
    """
    Vectorized version of the engagement score function.

    Args:
        likes (pd.Series): Series of likes.
        retweets (pd.Series): Series of retweets.
        followers (pd.Series): Series of follower counts.

    Returns:
        pd.Series: Series of engagement scores.
    """
    # Calculate total engagement (likes + retweets)
    engagement = likes + retweets

    # Use NumPy to avoid division by zero errors
    engagement_rate = np.where(
        followers > 0, 
        100 + (engagement / followers) * 100, 
        0.0  # If followers == 0, set engagement rate to 0.0
    )

    return pd.Series(engagement_rate, index=followers.index)

def normalization(engagement: pd.Series, sentiment: pd.Series, confidence: pd.Series) -> pd.Series:
    """
    Calculate the normalized score by multiplying engagement, sentiment, and confidence.

    Args:
        engagement (pd.Series): Series of engagement scores.
        sentiment (pd.Series): Series of sentiment scores.
        confidence (pd.Series): Series of confidence values.

    Returns:
        pd.Series: Series of normalized scores.
    """
    # Ensure element-wise multiplication
    normalize = engagement * sentiment * confidence
    return normalize
    
