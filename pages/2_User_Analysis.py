import streamlit as st
from general_utils.app_state import init_state
import pandas as pd
import plotly.express as px
from general_utils.streamlit_filters import StreamlitFilters
class UserAnalysisPage:
    def __init__(self):
        
        init_state()
        self.data = st.session_state['data']
        self.filters = StreamlitFilters(self.data)
        self.selected_hashtag = []
        self.filtered_data = pd.DataFrame()
        self.user_metrics = None
        self.main()
        
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

    def calculate_user_metrics(self):
        """Calculate and prepare user-level metrics"""
        self.user_metrics = self.filtered_data.groupby('user_id').agg({
            'days_from_join_date': 'first',
            'user_id_post_count': 'first',
            'user_followers_count': 'first',
            'hashtag': lambda x: list(x.unique()),
            'tweet_id': 'count',
            'clean_tweet': lambda x: ' | '.join(x.head(3))
        }).reset_index()
        
        # Calculate derived metrics
        self.user_metrics['posts_per_day'] = self.user_metrics['user_id_post_count'] / self.user_metrics['days_from_join_date']
        self.user_metrics['suspicious_score'] = 0
        
        # Calculate suspicion scores
        post_freq_threshold = self.user_metrics['posts_per_day'].quantile(0.95)
        follower_threshold = self.user_metrics['user_followers_count'].quantile(0.10)
        
        self.user_metrics.loc[self.user_metrics['posts_per_day'] > post_freq_threshold, 'suspicious_score'] += 1
        self.user_metrics.loc[self.user_metrics['user_followers_count'] < follower_threshold, 'suspicious_score'] += 1
        self.user_metrics.loc[self.user_metrics['days_from_join_date'] < 30, 'suspicious_score'] += 1

    def display_distribution_analysis(self):
        """Display distribution analysis section"""
        st.header("📊 Distribution Analysis")
        
        # Account Age Distribution
        fig_dist_age = px.histogram(
            self.user_metrics,
            x='days_from_join_date',
            nbins=50,
            title="Distribution of Account Ages",
            labels={'days_from_join_date': 'Account Age (days)', 'count': 'Number of Users'}
        )
        fig_dist_age.add_trace(px.violin(self.user_metrics, x='days_from_join_date').data[0])
        fig_dist_age.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist_age, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Median Age", f"{self.user_metrics['days_from_join_date'].median():.0f} days")
        with col2:
            st.metric("Mean Age", f"{self.user_metrics['days_from_join_date'].mean():.0f} days")
        
        # Follower Count Distribution
        fig_dist_followers = px.histogram(
            self.user_metrics,
            x='user_followers_count',
            nbins=50,
            title="Log Distribution of Follower Counts",
            labels={'user_followers_count': 'Follower Count', 'count': 'Number of Users'},
            log_y = True
        )
        fig_dist_followers.add_trace(px.violin(self.user_metrics, x='user_followers_count').data[0])
        fig_dist_followers.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist_followers, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Median Followers", f"{self.user_metrics['user_followers_count'].median():,.0f}")
        with col2:
            st.metric("Mean Followers", f"{self.user_metrics['user_followers_count'].mean():,.0f}")
        
        # Posts Count Distribution
        fig_dist_posts_count = px.histogram(
            self.user_metrics,
            x='user_id_post_count',  
            nbins=50,
            title="Log Distribution of Total Posts",  
            labels={'user_id_post_count': 'Total Posts', 'count': 'Number of Users'},
            log_y= True  
        )
        
        fig_dist_posts_count.add_trace(px.violin(self.user_metrics, x='user_id_post_count').data[0])
        fig_dist_posts_count.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist_posts_count, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Median Total Posts", f"{self.user_metrics['user_id_post_count'].median():.0f}")
        with col2:
            st.metric("Mean Total Posts", f"{self.user_metrics['user_id_post_count'].mean():.0f}")
                
        
        # Posts per Day Distribution
        fig_dist_posts = px.histogram(
            self.user_metrics,
            x='posts_per_day',
            nbins=50,
            title="Log Distribution of Posts per Day",
            labels={'posts_per_day': 'Posts per Day', 'count': 'Number of Users'},
            log_y=True
        )
        fig_dist_posts.add_trace(px.violin(self.user_metrics, x='posts_per_day').data[0])
        fig_dist_posts.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist_posts, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Median Posts/Day", f"{self.user_metrics['posts_per_day'].median():.2f}")
        with col2:
            st.metric("Mean Posts/Day", f"{self.user_metrics['posts_per_day'].mean():.2f}")
            



    def display_activity_analysis(self):
        """Display user activity analysis section"""
        st.header("📈 User Activity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(
                self.user_metrics,
                x='days_from_join_date',
                y='user_id_post_count',
                color='suspicious_score',
                size='user_followers_count',
                color_continuous_scale='RdYlBu_r',
                title="Account Age vs Total Posts",
                labels={
                    'days_from_join_date': 'Account Age (days)',
                    'user_id_post_count': 'Total Posts',
                    'suspicious_score': 'Suspicious Score'
                }
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(
                self.user_metrics,
                x='user_followers_count',
                y='posts_per_day',
                color='suspicious_score',
                size='tweet_id',
                color_continuous_scale='RdYlBu_r',
                title="Follower Count vs Post Frequency",
                labels={
                    'user_followers_count': 'Follower Count',
                    'posts_per_day': 'Posts per Day',
                    'suspicious_score': 'Suspicious Score'
                }
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

    def display_suspicious_accounts(self):
        """Display suspicious accounts section"""
        st.header("🚨 Suspicious Accounts Analysis")
        with st.expander("ℹ️ How is suspicious activity calculated?"):
            st.write("""
            Suspicious activity is determined by a scoring system that considers three main factors:

            1. **Posting Frequency**: If a user's posts per day exceeds the 95th percentile of all users
            2. **Follower Count**: If a user's follower count is below the 10th percentile
            3. **Account Age**: If the account is less than 30 days old

            Each factor adds 1 point to the suspicious score. Users with a score of 2 or higher are considered suspicious.

            - **Score 0-1**: Normal activity
            - **Score 2**: Potentially suspicious
            - **Score 3**: Highly suspicious
            """)

        suspicious_accounts = self.user_metrics[self.user_metrics['suspicious_score'] >= 2].sort_values(
            by='suspicious_score',
            ascending=False
        ).head(10)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        total_users = len(self.user_metrics)
        suspicious_users = len(self.user_metrics[self.user_metrics['suspicious_score'] >= 2])
        
        with col1:
            st.metric("Total Users", f"{total_users:,}")
        with col2:
            st.metric("Suspicious Users", f"{suspicious_users:,}")
        with col3:
            st.metric("Suspicious Ratio", f"{(suspicious_users/total_users)*100:.1f}%")

        # Suspicious accounts details
        st.subheader("Most Suspicious Accounts")
        for idx, row in suspicious_accounts.iterrows():
            with st.expander(f"User ID: {row['user_id']} (Suspicious Score: {row['suspicious_score']})"):
                st.write(f"**Account Age:** {row['days_from_join_date']} days")
                st.write(f"**Followers:** {row['user_followers_count']:,}")
                st.write(f"**Posts per Day:** {row['posts_per_day']:.2f}")
                st.write("**Sample Tweets:**")
                for tweet in row['clean_tweet'].split(' | '):
                    st.write(f"- {tweet}")

    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.selected_hashtag = self.filters.selected_hashtag
        self.filtered_data = self.filters.filtered_data
        
        if not self.filtered_data.empty:
            self.calculate_user_metrics()
            self.display_distribution_analysis()
            self.display_activity_analysis()
            self.display_suspicious_accounts()
            self.user_influence()
        else:
            st.warning("No data available for analysis. Please select at least one hashtag.")

if __name__ == "__main__":
    UserAnalysisPage()
