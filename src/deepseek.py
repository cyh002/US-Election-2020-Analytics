# Please install OpenAI SDK first: `pip3 install openai`
import sys
import argparse
import os
from openai import OpenAI

def get_sentiment(text, api_key, base_url="https://api.deepseek.com"):
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Define the prompt for sentiment analysis
    prompt = (
        "You are to give a sentiment value and the confidence level of the user's message. "
        "Sentiment value is an integer between -1 and 1, where -1 is negative, 0 is neutral, and 1 is positive. "
        "Confidence level is a float between 0.00 and 1.00, where 0.00 is low confidence and 1.00 is high confidence. "
        "For example, if the user's message is 'I love this product', your response should be '1, 1.00'.\n\n"
        f"User message: \"{text}\"\n"
        "Response:"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )

        sentiment_output = response.choices[0].message.content.strip()  # Access 'content' directly
        return sentiment_output

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment of a given text.')
    parser.add_argument('text', type=str, help='The text to analyze for sentiment.')
    parser.add_argument('--api_key', type=str, help='DeepSeek API key. Alternatively, set the DEEPSEEK_API_KEY environment variable.')
    parser.add_argument('--base_url', type=str, default="https://api.deepseek.com", help='Base URL for DeepSeek API (if using a proxy or alternative endpoint).')

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DeepSeek API key not provided. Use --api_key argument or set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    sentiment = get_sentiment(args.text, api_key, args.base_url)
    print(sentiment)

if __name__ == "__main__":
    main()
