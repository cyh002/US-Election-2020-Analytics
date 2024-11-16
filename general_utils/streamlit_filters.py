import streamlit as st
import pandas as pd
from datetime import date
from typing import List, Tuple

class StreamlitFilters:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the StreamlitFilters class with data.
        
        Args:
            data (pd.DataFrame): Data to be filtered.
        """
        self.data = data
        self.selected_hashtag: List[str] = []
        self.selected_state: List[str] = []
        self.selected_days: int = 0
        self.min_followers: int = 10
        self.selected_date_range: Tuple[date, date] = (date.today(), date.today())
        self.filtered_data: pd.DataFrame = pd.DataFrame()
        self.disable_filters: List[str] = []

    def sidebar_filters(self) -> None:
        """
        Creates the sidebar filter interface in Streamlit, allowing the user to filter by hashtags, states, days,
        followers count, and date range.
        """
        st.sidebar.header("Filters")
        # Hashtag Filter
        hashtag_expander = st.sidebar.expander("📌 Select Hashtag(s)", expanded=True)
        with hashtag_expander:
            hashtags = self.data['hashtag'].unique().tolist()
            self.selected_hashtag = [
                hashtag for hashtag in hashtags if st.checkbox(hashtag, key=f"hashtag_{hashtag}", value=True)
            ]
        
        # State Filter
        state_expander = st.sidebar.expander("📍 Select State(s)", expanded=False)
        with state_expander:
            states = self.data['state'].unique().tolist()
            self.selected_state = [
                state for state in states if st.checkbox(state, key=f"state_{state}", value=True)
            ]

        # Days from Join Date Slider
        days_expander = st.sidebar.expander("📅 Days from Join Date", expanded=True)
        with days_expander:
            min_days = int(self.data['days_from_join_date'].min())
            max_days = int(self.data['days_from_join_date'].max())
            self.selected_days = st.slider(
                "Minimum days since user joined:",
                min_value=min_days,
                max_value=max_days,
                value=min_days
            )

        # User Followers Count Filter
        followers_expander = st.sidebar.expander("👥 Followers Count", expanded=True)
        with followers_expander:
            self.min_followers = st.number_input(
                "Minimum user followers count:",
                min_value=0,
                value=10,
                step=1
            )

        # Date Range Filter
        if 'date_range' not in self.disable_filters:
            date_expander = st.sidebar.expander("🕒 Select Date Range", expanded=True)
            with date_expander:
                min_date = self.data['created_date'].min()
                max_date = self.data['created_date'].max()
                self.selected_date_range = st.date_input(
                    "Select Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                # Ensure selected_date_range is a valid tuple of two dates
                if isinstance(self.selected_date_range, date):
                    self.selected_date_range = (self.selected_date_range, self.selected_date_range)
                elif len(self.selected_date_range) != 2:
                    st.error("Please select a start and end date.")
        else:
            self.selected_date_range = (self.data['created_date'].min(), self.data['created_date'].max())

    def apply_filters(self) -> None:
        """
        Applies the filters selected in the sidebar to the data, creating a filtered DataFrame.
        """
        # Check if selected_date_range contains a valid date range
        if isinstance(self.selected_date_range, tuple):
            if self.selected_date_range[0] is None or self.selected_date_range[1] is None:
                st.warning("Please select a valid start and end date range.")
                self.filtered_data = pd.DataFrame()  # Clear filtered data to prevent plotting
                return

            # Handle the case where only one date is selected
            if isinstance(self.selected_date_range[0], date) and self.selected_date_range[1] is None:
                self.selected_date_range = (self.selected_date_range[0], self.selected_date_range[0])
            elif isinstance(self.selected_date_range[1], date) and self.selected_date_range[0] is None:
                self.selected_date_range = (self.selected_date_range[1], self.selected_date_range[1])

        # Apply filters if the date range is valid
        self.filtered_data = self.data[
            (self.data['hashtag'].isin(self.selected_hashtag)) &
            (self.data['state'].isin(self.selected_state)) &
            (self.data['days_from_join_date'] >= self.selected_days) &
            (self.data['user_followers_count'] >= self.min_followers) &
            (self.data['created_date'] >= self.selected_date_range[0]) &
            (self.data['created_date'] <= self.selected_date_range[1])
        ].dropna(subset=['normalized_score'])

        # Warn if no data matches the filters
        if self.filtered_data.empty:
            st.warning("No data matches the selected filters.")
