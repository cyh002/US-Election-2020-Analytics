# pages/5_Word_Cloud.py

import streamlit as st
from general_utils.app_state import init_state
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from general_utils.streamlit_filters import StreamlitFilters

class WordCloudPage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.selected_hashtag = []
        self.filters = StreamlitFilters(self.data)
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

    def generate_word_cloud(self):
        st.header("☁️ Word Cloud by Hashtag")

        for hashtag in self.selected_hashtag:
            st.markdown(f"#### #{hashtag.capitalize()}")
            hashtag_data = self.filtered_data[self.filtered_data['hashtag'] == hashtag]
            text = " ".join(tweet for tweet in hashtag_data['clean_tweet'].astype(str))

            if text.strip():
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white'
                ).generate(text)
                plt.figure(figsize=(15, 7.5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
                plt.close()  # Close the figure after rendering
            else:
                st.info(f"No tweets available for #{hashtag}.")

    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.selected_hashtag = self.filters.selected_hashtag
        self.filtered_data = self.filters.filtered_data
        self.generate_word_cloud()

if __name__ == "__main__":
    WordCloudPage()
