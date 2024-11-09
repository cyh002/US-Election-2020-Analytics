import streamlit as st
from general_utils.app_state import init_state
import pandas as pd
import plotly.express as px
from app.general_utils.streamlit_filters import StreamlitFilters

class UserAnalysisPage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.filters = StreamlitFilters(self.data)
        self.selected_hashtag = []
        self.filtered_data = pd.DataFrame()
        self.main()

    def sidebar_filters(self):
        st.sidebar.header("Filters")
        with st.sidebar.expander("📊 Filter Options"):
            # Hashtag Filter
            hashtags = self.data['hashtag'].unique().tolist()
            st.markdown("**Select Hashtag(s):**")
            for hashtag in hashtags:
                if st.sidebar.checkbox(hashtag, value=True):
                    self.selected_hashtag.append(hashtag)

    def apply_filters(self):
        self.filtered_data = self.data[self.data['hashtag'].isin(self.selected_hashtag)]

    def user_influence(self):
        st.header("📈 User Influence Analysis")

        if self.filtered_data.empty:
            st.warning("No data available to display the user influence analysis.")
            return

        fig = px.scatter(
            self.filtered_data,
            x='user_followers_count',
            y='normalized_score',
            size='engagement',
            color='hashtag',
            color_discrete_map={
                'trump': 'red',
                'biden': 'blue',
                'both': 'green'
            },
            title="User Followers vs. Normalized Sentiment Scores",
            labels={
                'user_followers_count': 'User Followers Count',
                'normalized_score': 'Normalized Sentiment Score',
                'engagement': 'Engagement'
            },
            log_x=True,
            hover_data=['state', 'hashtag']
        )

        fig.update_layout(
            legend_title_text='Hashtag',
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )

        st.plotly_chart(fig, use_container_width=True)
        
    def analyze_bot_patterns(self):
        st.header("🤖 Bot Pattern Analysis")
        
        # Group by user_id to get user-level metrics
        user_metrics = self.filtered_data.groupby('user_id').agg({
            'days_from_join_date': 'first',  
            'user_id_post_count': 'first',   
            'user_followers_count': 'first',  
            'hashtag': lambda x: list(x.unique()),  
            'tweet_id': 'count',
            'clean_tweet': lambda x: ' | '.join(x.head(3))  # Join first 3 tweets with separator
        }).reset_index()
        
        # Calculate posts per day and suspicious score
        user_metrics['posts_per_day'] = user_metrics['user_id_post_count'] / user_metrics['days_from_join_date']
        user_metrics['suspicious_score'] = 0
        
        post_freq_threshold = user_metrics['posts_per_day'].quantile(0.95)
        follower_threshold = user_metrics['user_followers_count'].quantile(0.10)
        
        user_metrics.loc[user_metrics['posts_per_day'] > post_freq_threshold, 'suspicious_score'] += 1
        user_metrics.loc[user_metrics['user_followers_count'] < follower_threshold, 'suspicious_score'] += 1
        user_metrics.loc[user_metrics['days_from_join_date'] < 30, 'suspicious_score'] += 1

        # First plot - Account Age vs Post Activity
        st.subheader("Account Age vs Post Activity")
        fig1 = px.scatter(
            user_metrics,
            x='days_from_join_date',
            y='user_id_post_count',
            color='suspicious_score',
            size='user_followers_count',
            color_continuous_scale='RdYlBu_r',
            title="Account Age vs Total Posts",
            labels={
                'days_from_join_date': 'Account Age (days)',
                'user_id_post_count': 'Total Posts',
                'suspicious_score': 'Suspicious Score',
                'user_followers_count': 'Follower Count'
            },
            hover_data={
                'user_id': True,
                'user_followers_count': True,
                'posts_per_day': ':.2f',
                'clean_tweet': True
            }
        )
        
        fig1.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Second plot - Follower Count vs Post Frequency
        st.subheader("Follower Count vs Post Frequency")
        fig2 = px.scatter(
            user_metrics,
            x='user_followers_count',
            y='posts_per_day',
            color='suspicious_score',
            size='tweet_id',
            color_continuous_scale='RdYlBu_r',
            title="Follower Count vs Post Frequency",
            labels={
                'user_followers_count': 'Follower Count',
                'posts_per_day': 'Posts per Day',
                'suspicious_score': 'Suspicious Score',
                'tweet_id': 'Tweets in Dataset'
            },
            hover_data={
                'user_id': True,
                'days_from_join_date': True,
                'clean_tweet': True
            }
        )
        
        fig2.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Add table of suspicious accounts
        st.subheader("Most Suspicious Accounts")
        suspicious_accounts = user_metrics[user_metrics['suspicious_score'] >= 2].sort_values(
            by='suspicious_score', 
            ascending=False
        ).head(10)

        for idx, row in suspicious_accounts.iterrows():
            with st.expander(f"User ID: {row['user_id']} (Suspicious Score: {row['suspicious_score']})"):
                st.write(f"**Account Age:** {row['days_from_join_date']} days")
                st.write(f"**Followers:** {row['user_followers_count']:,}")
                st.write(f"**Posts per Day:** {row['posts_per_day']:.2f}")
                st.write("**Sample Tweets:**")
                for tweet in row['clean_tweet'].split(' | '):
                    st.write(f"- {tweet}")

        # Display summary statistics
        st.subheader("Suspicious Account Summary")
        total_users = len(user_metrics)
        suspicious_users = len(user_metrics[user_metrics['suspicious_score'] >= 2])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", f"{total_users:,}")
        with col2:
            st.metric("Suspicious Users", f"{suspicious_users:,}")
        with col3:
            st.metric("Suspicious Ratio", f"{(suspicious_users/total_users)*100:.1f}%")


    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.selected_hashtag = self.filters.selected_hashtag
        self.filtered_data = self.filters.filtered_data
        
        self.analyze_bot_patterns()
        self.user_influence()

if __name__ == "__main__":
    UserAnalysisPage()
