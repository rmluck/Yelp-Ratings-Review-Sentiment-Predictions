#!/usr/bin/env python3
"""
Yelp Dataset CA Restaurant Filter for Bayesian Regression Project

This script filters the Yelp dataset to focus on restaurants in CA with 50-1000 reviews.
It performs sentiment analysis and creates the dataset format needed for
predicting Yelp ratings from review sentiment and popularity using Bayesian regression.

Usage:
    python filter_city_data.py
"""

import pandas as pd
import json
import os
import numpy as np
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class YelpRestaurantFilter:
    def __init__(self, dataset_path: str = "Yelp JSON/yelp_dataset"):
        """
        Initialize the Yelp Restaurant Filter for Bayesian Regression
        
        Args:
            dataset_path: Path to the directory containing Yelp JSON files
        """
        self.dataset_path = dataset_path
        self.business_file = os.path.join(dataset_path, "yelp_academic_dataset_business.json")
        self.review_file = os.path.join(dataset_path, "yelp_academic_dataset_review.json")
        
        # Initialize VADER sentiment analyzer
        self._setup_sentiment_analyzer()
        
    def _setup_sentiment_analyzer(self):
        """Download VADER lexicon if needed and setup sentiment analyzer"""
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            print("Downloading VADER lexicon...")
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def load_ca_restaurants(self):
        """
        Load restaurants in CA with 50-1000 reviews (matching notebook filters)
        
        Returns:
            DataFrame containing restaurant businesses in CA
        """
        print("Loading restaurants in CA with 50-1000 reviews...")
        restaurants = []
        
        try:
            with open(self.business_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % 10000 == 0 and i > 0:
                        print(f"Processed {i} businesses, found {len(restaurants)} CA restaurants")
                    
                    try:
                        business = json.loads(line)
                        
                        # Apply the same filters as in the notebook
                        if business.get("categories"):
                            categories = business["categories"].split(", ")
                            
                            if ("Restaurants" in categories and 
                                business.get("state") == "CA" and 
                                50 < business.get("review_count", 0) < 1000):
                                restaurants.append(business)
                                
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Business file not found at {self.business_file}")
            return pd.DataFrame()
        
        print(f"Found {len(restaurants)} restaurants in CA with 50-1000 reviews")
        return pd.DataFrame(restaurants)
    
    def load_reviews_with_sentiment(self, business_ids):
        """
        Load reviews for restaurants and calculate sentiment scores
        
        Args:
            business_ids: Set of business IDs to get reviews for
            
        Returns:
            Dictionary mapping business_id to list of reviews with sentiment scores
        """
        print(f"Loading reviews and calculating sentiment for {len(business_ids)} restaurants...")
        reviews_by_business = defaultdict(list)
        total_reviews = 0
        
        try:
            with open(self.review_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % 50000 == 0 and i > 0:
                        print(f"Processed {i} reviews, found {total_reviews} restaurant reviews")
                    
                    try:
                        review = json.loads(line)
                        business_id = review.get('business_id')
                        
                        if business_id in business_ids:
                            # Calculate sentiment score using VADER (only compound score)
                            text = review.get('text', '')
                            if text:
                                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                                review['sentiment_compound'] = sentiment_scores['compound']
                            else:
                                # Default neutral sentiment if no text
                                review['sentiment_compound'] = 0.0
                            
                            reviews_by_business[business_id].append(review)
                            total_reviews += 1
                            
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Review file not found at {self.review_file}")
            return {}
        
        print(f"Found {total_reviews} reviews for CA restaurants")
        return dict(reviews_by_business)
    
    def create_bayesian_dataset(self):
        """
        Create the final dataset for Bayesian regression analysis
        
        Returns:
            DataFrame with columns: business_id, name, avg_rating, avg_sentiment_score, log_review_count
        """
        print("Creating Bayesian regression dataset...")
        
        # Load restaurants
        restaurants_df = self.load_ca_restaurants()
        
        if restaurants_df.empty:
            print("No restaurants found in CA with 50-1000 reviews")
            return pd.DataFrame()
        
        # Get business IDs
        business_ids = set(restaurants_df['business_id'])
        
        # Load reviews with sentiment
        reviews_by_business = self.load_reviews_with_sentiment(business_ids)
        
        # Create final dataset
        final_data = []
        
        for _, restaurant in restaurants_df.iterrows():
            business_id = restaurant['business_id']
            
            # Get basic info from business data
            avg_rating = restaurant.get('stars', 0)
            review_count = restaurant.get('review_count', 0)
            
            # Calculate average sentiment from reviews
            reviews = reviews_by_business.get(business_id, [])
            
            if reviews:
                sentiment_scores = [r['sentiment_compound'] for r in reviews]
                avg_sentiment_score = np.mean(sentiment_scores)
                actual_review_count = len(reviews)
            else:
                # Neutral sentiment if no reviews
                avg_sentiment_score = 0.0  
                actual_review_count = review_count 
            
            # Calculate log review count (adding 1 to avoid log(0))
            log_review_count = np.log(1 + actual_review_count)
            
            # Only include restaurants with at least some data
            if avg_rating > 0 and actual_review_count > 0:
                final_data.append({
                    'business_id': business_id,
                    'name': restaurant.get('name', ''),
                    'avg_rating': avg_rating,
                    'avg_sentiment_score': avg_sentiment_score,
                    'log_review_count': log_review_count,
                })
        
        result_df = pd.DataFrame(final_data)
        print(f"Created dataset with {len(result_df)} restaurants for Bayesian analysis")
        
        return result_df
    
    def print_dataset_statistics(self, df):
        """
        Print statistics about the final dataset
        """
        if df.empty:
            print("Dataset is empty")
            return
        
        print(f"Total restaurants: {len(df)}")
        print(f"Average rating: {df['avg_rating'].mean():.3f} +- {df['avg_rating'].std():.3f}")
        print(f"Average sentiment: {df['avg_sentiment_score'].mean():.3f} +- {df['avg_sentiment_score'].std():.3f}")
        print(f"Average log review count: {df['log_review_count'].mean():.3f} +- {df['log_review_count'].std():.3f}")
        
        print(f"\nRating distribution:")
        rating_counts = df['avg_rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  {rating} stars: {count} restaurants")
        
        print(f"\nSentiment score distribution:")
        print(f"  Positive (>0.1): {(df['avg_sentiment_score'] > 0.1).sum()} restaurants")
        print(f"  Neutral (-0.1 to 0.1): {((df['avg_sentiment_score'] >= -0.1) & (df['avg_sentiment_score'] <= 0.1)).sum()} restaurants")
        print(f"  Negative (<-0.1): {(df['avg_sentiment_score'] < -0.1).sum()} restaurants")


def main():
    """
    Main function to create the CA restaurant dataset for Bayesian regression
    """
    # Initialize the filter
    filter_tool = YelpRestaurantFilter()
    
    # Check if dataset files exist
    if not os.path.exists(filter_tool.business_file):
        print("Error: Yelp dataset not found!")
        print("Please ensure the dataset is extracted to 'Yelp JSON/yelp_dataset/'")
        return
    
    # Create the Bayesian regression dataset
    dataset_df = filter_tool.create_bayesian_dataset()
    
    if not dataset_df.empty:
        # Print statistics
        filter_tool.print_dataset_statistics(dataset_df)
        
        # Save the dataset
        output_file = "ca_restaurants_bayesian_dataset.csv"
        dataset_df.to_csv(output_file, index=False)
        print(f"\nDataset saved to: {output_file}")
        
        # Show a few examples
        print(f"\nFirst 5 restaurants:")
        print(dataset_df[['name', 'avg_rating', 'avg_sentiment_score', 'log_review_count']].head())
        
    else:
        print("No data found. Please check your dataset files.")


if __name__ == "__main__":
    main() 