def engagement_score(likes: int, retweets: int, followers: int) -> float:
    """
    Calculate the engagement rate as 100% + (Engagement (Likes + Retweets) / Number of followers).
    
    Args:
        likes (int): The number of likes on the post.
        retweets (int): The number of retweets on the post.
        followers (int): The number of followers of the account.

    Returns:
        float: The engagement rate as a percentage.
        
    Example:
        >>> calculate_engagement_rate(50, 30, 1000)
        108.0
    """
    if followers == 0:
        raise ValueError("Number of followers cannot be zero.")
    
    engagement = likes + retweets
    engagement_rate = 100 + (engagement / followers) * 100
    
    print(f"Likes: {likes}, Retweets: {retweets}, Followers: {followers}, Engagement Rate: {engagement_rate}%")
    
    return engagement_rate
