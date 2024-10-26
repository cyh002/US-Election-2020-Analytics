import pandas as pd
import numpy as np

# Engagement Metric Calculation and Normalization Adjustments
# -----------------------------------------------------------
# This code provides functions to calculate and normalize an engagement score that accounts for both engagement rates and follower count.

# 1. engagement_score function:
#    - Calculates an engagement score based on likes, retweets, and follower count.
#    - Normalizes engagement by follower count and scales it with a log transformation of followers.
#    - This log-based scaling gives weight to accounts with larger audiences without overly inflating their score.

# 2. normalization function:
#    - Transforms engagement using a log transformation to reduce the impact of high-engagement outliers.
#    - Adjusts sentiment scores from a -1 to 1 scale to a 0 to 1 range, preventing negative scores from dominating.
#    - Multiplies engagement, sentiment, and confidence scores and then standardizes to center the result.
#    - Applies a secondary normalization to map values to a standard range, improving consistency in score interpretation.

# 3. normalize_scores function:
#    - Maps final normalized scores to a [-1, 1] range for simplified visualization and easier interpretation.

# These adjustments allow for a more balanced, interpretable score, capturing both the magnitude and polarity of sentiment effectively across varying follower counts.


def engagement_score(likes: pd.Series, retweets: pd.Series, followers: pd.Series) -> pd.Series:
    """
    Calculate a scaled engagement score considering both engagement and follower count.

    Parameters:
        likes (pd.Series): Series representing the number of likes per post.
        retweets (pd.Series): Series representing the number of retweets per post.
        followers (pd.Series): Series representing the number of followers of each account.

    Returns:
        pd.Series: A Series containing the engagement score, scaled by the logarithm of follower count.
    """
    engagement = likes + retweets
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero and NaNs
        engagement_rate = (engagement / followers) * 100  # Calculate engagement rate
        engagement_rate = np.where(followers > 0, engagement_rate, 0.0)  # Replace NaN values with 0
    scaled_engagement = engagement_rate * np.log1p(followers)  # Scale by log of followers
    return pd.Series(scaled_engagement, index=followers.index)  # Return as Series

def normalization(engagement: pd.Series, sentiment: pd.Series, confidence: pd.Series) -> pd.Series:
    """
    Normalize engagement scores by adjusting for sentiment, confidence, and scaling.

    Parameters:
        engagement (pd.Series): Series of engagement scores.
        sentiment (pd.Series): Series representing sentiment scores, ranging from -1 to 1.
        confidence (pd.Series): Series representing the confidence level of sentiment, between 0 and 1.

    Returns:
        pd.Series: A Series containing the final normalized scores within a consistent range.
    """
    adjusted_sentiment = (sentiment + 1) / 2  # Transform sentiment to range [0, 1]
    log_engagement = np.log1p(engagement)  # Apply log transformation to engagement
    normalize = log_engagement * adjusted_sentiment * confidence  # Combine scaled engagement with sentiment and confidence
    standardized_score = (normalize - normalize.mean()) / normalize.std()  # Standardize scores
    final_score = normalize_scores(standardized_score)  # Normalize scores to a standard range
    return final_score

def normalize_scores(scores: pd.Series) -> pd.Series:
    """
    Normalize scores to the range [-1, 1] for consistency in interpretation.

    Parameters:
        scores (pd.Series): Series of standardized scores to be normalized.

    Returns:
        pd.Series: A Series containing the scores normalized to the range [-1, 1].
    """
    min_score = scores.min()
    max_score = scores.max()
    normalized = 2 * (scores - min_score) / (max_score - min_score) - 1  # Scale to [-1, 1]
    return normalized

