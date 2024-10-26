# pages/1_Latest_Day_Analysis.py

import streamlit as st
from app.general_utils.app_state import init_state
import pandas as pd
from app.general_utils.streamlit_filters import StreamlitFilters

class LatestDayAnalysisPage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.data_loader = st.session_state['data_loader']
        self.llm_analyzer = st.session_state['llm_analyzer']
        self.llm_analyzer_count = self.data_loader.llm_analyzer_count
        self.selected_hashtag = []
        self.filtered_data = pd.DataFrame()
        self.llm_analyzer.model = self.data_loader.openai_model
        self.filters = StreamlitFilters(self.data)
        self.main()

    def perform_analysis(self):
        st.header("📅 Latest Day Analysis")
        st.subheader(f"Performing analysis on the Top {self.llm_analyzer_count} Tweets, using Model: {self.llm_analyzer.model}")

        if not self.filtered_data.empty:
            # Filter data for the latest day
            latest_day_data = self.llm_analyzer.filter_latest_day(self.filtered_data)
            columns_to_analyze = ['clean_tweet', 'engagement']
            latest_day_data = latest_day_data[columns_to_analyze]
            latest_date = self.filtered_data['created_date'].max()

            # Limiting tweets to prevent token overflow
            latest_day_data = latest_day_data.sort_values(by='engagement', ascending=False)
            latest_day_data = latest_day_data.head(self.llm_analyzer_count)
            st.subheader(f"Sample data: {latest_date}")
            
            # Show the table
            # reindex
            latest_day_data = latest_day_data.reset_index(drop=True)
            st.write(latest_day_data)

            if latest_day_data.empty:
                st.warning("No data available for the latest day.")
            else:
                with st.spinner("Performing analysis..."):
                    # Generate analysis report using LLMAnalyzer
                    analysis_report = self.llm_analyzer.generate_daily_report(latest_day_data)

                if analysis_report:
                    # Display the structured report
                    st.markdown(f"### 📄 Report for {latest_date}")

                    # Formatting the output for readability
                    st.subheader("Sentiment Analysis")
                    for sentiment in analysis_report.sentiments:
                        st.markdown(f"**Candidate**: {sentiment.candidate}")
                        st.markdown(f"**Sentiment**: {sentiment.sentiment.capitalize()}")
                        st.markdown(f"**Score**: {sentiment.score}")
                        st.markdown(f"**Key Topics**: {', '.join(sentiment.key_topics)}")
                        st.markdown("---")  # Separator for readability

                    st.subheader("Overall Analysis")
                    st.markdown(f"**Comparison**: {analysis_report.overall_analysis.comparison}")
                    st.markdown(f"**Confidence Score**: {analysis_report.overall_analysis.confidence_score}")

                    if analysis_report.additional_insights:
                        st.subheader("Additional Insights")
                        st.markdown(f"{analysis_report.additional_insights}")
        else:
            st.warning("No data available after applying the selected filters.")

    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.filtered_data = self.filters.filtered_data
        self.perform_analysis()

if __name__ == "__main__":
    LatestDayAnalysisPage()
