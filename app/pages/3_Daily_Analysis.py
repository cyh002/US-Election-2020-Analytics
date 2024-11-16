import streamlit as st
from app.general_utils.app_state import init_state
import pandas as pd
from app.general_utils.streamlit_filters import StreamlitFilters



class DailyAnalysisPage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.data_loader = st.session_state['data_loader']
        self.llm_analyzer = st.session_state['llm_analyzer']
        self.llm_analyzer_count = st.session_state['config']['streamlit']['llm_analyzer_count']
        self.selected_hashtag = []
        self.filtered_data = pd.DataFrame()
        self.llm_analyzer.model = self.data_loader.openai_model
        self.filters = StreamlitFilters(self.data)
        self.main()
        
    def additional_sidebar(self):
            # Add input field for the number of tweets to analyze per day
        self.llm_analyzer_count = st.sidebar.number_input(
            "Number of Tweets to Analyze",
            min_value=0, 
            max_value=self.llm_analyzer_count,
            value=0,
            help="Number of tweets to analyze for sentiment analysis per day. Max : " + str(self.llm_analyzer_count)
        )
        
        # update self.filters that date range is not required
        self.filters.disable_filters.append('date_range')
        
        # Store the selected date
        self.selected_date = st.sidebar.date_input(
            "Select Date",
            min_value=self.data['created_date'].min(),
            max_value=self.data['created_date'].max(),
            value=self.data['created_date'].max(),
            help="Select a date to analyze tweets for that day."
        )
        
    def perform_analysis(self):
        st.header("📅 Daily Analysis")
        st.subheader(f"Performing analysis on the Top {self.llm_analyzer_count} Tweets, using model: {self.llm_analyzer.model}")

        if not self.filtered_data.empty:
            # Filter data for the latest day
            selected_day_data = self.filtered_data[
            self.filtered_data['created_date'] == self.selected_date
            ].copy()
            columns_to_analyze = ['clean_tweet', 'engagement']
            selected_day_data = selected_day_data[columns_to_analyze]
            selected_date = self.selected_date

            # Limiting tweets to prevent token overflow
            selected_day_data = selected_day_data.sort_values(by='engagement', ascending=False)
            selected_day_data = selected_day_data.head(self.llm_analyzer_count)
            
            # Show the table
            # reindex
            selected_day_data = selected_day_data.reset_index(drop=True)
            

            if selected_day_data.empty:
                st.warning("No data available for the day.")
            else:
                with st.spinner("Performing analysis..."):
                    # Generate analysis report using LLMAnalyzer
                    analysis_report = self.llm_analyzer.generate_daily_report(selected_day_data)

                if analysis_report:
                    # Display the structured report
                    st.markdown(f"### 📄 Report for {selected_date}")
                    st.write(selected_day_data)

                    # Formatting the output for readability
                    st.subheader("Sentiment Analysis")
                    for sentiment in analysis_report.sentiments:
                        st.markdown(f"**Candidate**: {sentiment.candidate}")
                        st.markdown(f"**Sentiment**: {sentiment.sentiment.capitalize()}")
                        st.markdown(f"**Score**: {sentiment.score}")
                        st.markdown(f"**Key Topics**: {', '.join(sentiment.key_topics)}")
                        st.markdown(f"**Key Figures**: {', '.join(sentiment.key_figures)}")
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
        self.additional_sidebar()
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.filtered_data = self.filters.filtered_data
        self.perform_analysis()

if __name__ == "__main__":
    DailyAnalysisPage()
