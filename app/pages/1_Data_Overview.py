import streamlit as st
from app.general_utils.app_state import init_state
import pandas as pd
from app.general_utils.streamlit_filters import StreamlitFilters
from plotly import express as px

class DataTablePage:
    def __init__(self):
        """
        Initializes the DataTablePage class, setting up state and loading data.
        """
        init_state()
        self.data = st.session_state['data']
        self.data_loader = st.session_state['data_loader']
        self.filters = StreamlitFilters(self.data)
        self.main()

    def display_data_stats(self, df: pd.DataFrame) -> None:
        """
        Displays basic statistics about the filtered dataset, including hashtag distribution.
        """
        # Basic stats
        total_tweets = len(df)
        total_engagement = df['engagement'].sum()
        avg_engagement = df['engagement'].mean()
        unique_states = df['state'].nunique()

        # Create columns for basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tweets", f"{total_tweets:,}")
        with col2:
            st.metric("Total Engagement", f"{total_engagement:,.0f}")
        with col3:
            st.metric("Avg. Engagement", f"{avg_engagement:,.2f}")
        with col4:
            st.metric("States Covered", unique_states)

        # Add separator
        st.markdown("---")
        
        # Hashtag distribution
        st.subheader("Hashtag Distribution")
        
        # Count tweets by hashtag
        trump_tweets = len(df[df['hashtag'] == 'trump'])
        biden_tweets = len(df[df['hashtag'] == 'biden'])
        
        # Create columns for hashtag metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Trump Tweets",
                value=f"{trump_tweets:,}",
                help=f"Percentage: {(trump_tweets/total_tweets)*100:.1f}%"
            )
        with col2:
            st.metric(
                label="Biden Tweets",
                value=f"{biden_tweets:,}",
                help=f"Percentage: {(biden_tweets/total_tweets)*100:.1f}%"
            )
        with col3:
            st.metric(
                label="Total Tweets",
                value=f"{total_tweets:,}"
            )


    def display_data_table(self, df: pd.DataFrame) -> None:
        """
        Displays the filtered data in a table format with additional features.
        """
        st.subheader("📊 Data Table")

        # Add column selector
        available_columns = df.columns.tolist()
        default_columns = ['created_date', 'state', 'hashtag', 'clean_tweet', 
                          'engagement', 'normalized_score']
        
        # Ensure default columns exist in available columns
        default_columns = [col for col in default_columns if col in available_columns]
        
        selected_columns = st.multiselect(
            "Select Columns to Display",
            options=available_columns,
            default=default_columns
        )

        if not selected_columns:
            st.warning("Please select at least one column to display.")
            return

        # Display number of rows selector
        n_rows = st.number_input(
            "Number of rows to display",
            min_value=1,
            max_value=len(df),
            value=min(50, len(df))
        )

        # Sort options
        sort_column = st.selectbox(
            "Sort by",
            options=selected_columns,
            index=0 if selected_columns else None
        )

        sort_order = st.radio(
            "Sort order",
            options=["Descending", "Ascending"],
            horizontal=True
        )

        # Apply sorting
        df_display = df.copy()
        if sort_column:
            df_display = df_display.sort_values(
                by=sort_column,
                ascending=(sort_order == "Ascending")
            )

        # Display the filtered and sorted dataframe
        st.dataframe(
            df_display[selected_columns].head(n_rows),
            use_container_width=True,
            hide_index=True
        )

        # Add download button
        if not df_display.empty:
            csv = df_display[selected_columns].to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name="filtered_twitter_data.csv",
                mime="text/csv"
            )
    def plot_hashtag_distribution(self, df: pd.DataFrame) -> None:
        """
        Creates a bar plot for hashtag distribution
        """
        hashtag_counts = df['hashtag'].value_counts().reset_index()
        hashtag_counts.columns = ['Hashtag', 'Count']
        
        # Calculate percentages
        total = hashtag_counts['Count'].sum()
        hashtag_counts['Percentage'] = (hashtag_counts['Count'] / total * 100).round(1)
        
        # Create bar chart using plotly
        fig = px.bar(
            hashtag_counts,
            x='Hashtag',
            y='Count',
            text=hashtag_counts.apply(lambda x: f'{x["Count"]:,}<br>({x["Percentage"]}%)', axis=1),
            color='Hashtag',
            color_discrete_map={'trump': '#E41A1C', 'biden': '#377EB8'},
            title='Tweet Distribution by Hashtag'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Number of Tweets",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_state_distribution(self, df: pd.DataFrame) -> None:
        """
        Creates a bar plot for state distribution with hashtag breakdown
        """
        # Get top 10 states by total tweet count
        top_states = df['state'].value_counts().head(10).index.tolist()
        
        # Filter dataframe for top 10 states
        df_top = df[df['state'].isin(top_states)]
        
        # Create grouped counts
        state_hashtag_counts = (df_top.groupby(['state', 'hashtag'])
                            .size()
                            .reset_index(name='Count'))
        
        # Create bar chart using plotly
        fig = px.bar(
            state_hashtag_counts,
            x='state',
            y='Count',
            color='hashtag',
            text='Count',
            title='Top 10 States by Tweet Count (Hashtag Distribution)',
            color_discrete_map={'trump': '#E41A1C', 'biden': '#377EB8'},
            barmode='group'  # or 'stack' if you prefer stacked bars
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="State",
            yaxis_title="Number of Tweets",
            legend_title="Hashtag",
            xaxis={'categoryorder':'total descending'}  # Sort states by total tweets
        )
        
        st.plotly_chart(fig, use_container_width=True)


    def main(self) -> None:
        """
        Main function that runs the data table page.
        """
        st.title("📑 Data Table View")
        
        # Apply filters from sidebar
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        filtered_data = self.filters.filtered_data

        if filtered_data.empty:
            st.warning("No data available with the current filter settings.")
            return

        # Display the data table first
        self.display_data_table(filtered_data)
        
        # Add separator
        st.markdown("---")
        
        # Display data statistics
        st.subheader("📊 Data Statistics")
        self.display_data_stats(filtered_data)
        
        # Add plots below the metrics
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_hashtag_distribution(filtered_data)
        
        with col2:
            self.plot_state_distribution(filtered_data)

if __name__ == "__main__":
    DataTablePage()
