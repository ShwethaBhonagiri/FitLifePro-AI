import numpy as np
from sklearn.cluster import KMeans
import json
import os

class FeedbackAnalyzer:
    """
    Analyzes user feedback to continuously improve fitness recommendations.
    """
    def __init__(self, feedback_file="user_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
        self._init_models()
    
    def _init_models(self):
        """Initialize the ML models for feedback analysis"""
        # Clustering model for user segmentation
        self.user_clustering = KMeans(n_clusters=4, random_state=42)
        
        # Feature weights for similarity calculations
        self.similarity_weights = {
            'age': 0.15,
            'gender': 0.05,
            'body_category': 0.20,
            'goal': 0.25,
            'activity_level': 0.15,
            'fitness_level': 0.20
        }
        
        # NEW: Enhanced model parameters
        self.sentiment_analysis_params = {
            'positive_keywords': ['love', 'great', 'excellent', 'good', 'enjoy', 'helpful', 'effective'],
            'negative_keywords': ['hard', 'difficult', 'too', 'not', 'can\'t', 'bad', 'unable'],
            'neutral_keywords': ['ok', 'fine', 'average', 'moderate'],
            'weight_factors': {
                'positive': 1.2,
                'negative': 0.8,
                'neutral': 1.0
            }
        }
        
        # NEW: Trend analysis parameters
        self.trend_analysis_params = {
            'compliance_threshold': 0.7,
            'progress_threshold': 0.5,
            'min_data_points': 3,
            'smoothing_factor': 0.2
        }
        
        # NEW: Initialize recommendation memory cache
        self.recommendation_cache = {}
    
    def _load_feedback(self):
        """Load existing feedback data from file"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except:
                return {"users": [], "workout_ratings": {}, "meal_ratings": {}}
        else:
            return {"users": [], "workout_ratings": {}, "meal_ratings": {}}
    
    def _save_feedback(self):
        """Save feedback data to file"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def add_user_feedback(self, user_id, user_data, feedback):
        """
        Add new user feedback to the database.
        
        Args:
            user_id (str): Unique identifier for the user
            user_data (dict): User profile data
            feedback (dict): Feedback on workouts and meals
                
        Returns:
            bool: Success status
        """
        # Check if user already exists
        user_exists = False
        for user in self.feedback_data["users"]:
            if user.get("user_id") == user_id:
                user["profile"] = user_data
                user["feedback_history"].append(feedback)
                user_exists = True
                break
        
        # Add new user if not found
        if not user_exists:
            self.feedback_data["users"].append({
                "user_id": user_id,
                "profile": user_data,
                "feedback_history": [feedback]
            })
        
        # Update workout ratings
        if "workout_ratings" in feedback:
            for workout_name, rating in feedback["workout_ratings"].items():
                if workout_name not in self.feedback_data["workout_ratings"]:
                    self.feedback_data["workout_ratings"][workout_name] = []
                
                self.feedback_data["workout_ratings"][workout_name].append({
                    "user_id": user_id,
                    "rating": rating,
                    "timestamp": feedback.get("timestamp", "")
                })
        
        # Update meal ratings
        if "meal_ratings" in feedback:
            for meal_name, rating in feedback["meal_ratings"].items():
                if meal_name not in self.feedback_data["meal_ratings"]:
                    self.feedback_data["meal_ratings"][meal_name] = []
                
                self.feedback_data["meal_ratings"][meal_name].append({
                    "user_id": user_id,
                    "rating": rating,
                    "timestamp": feedback.get("timestamp", "")
                })
        
        # NEW: Analyze sentiment in feedback comments
        if "comments" in feedback and feedback["comments"]:
            sentiment_score = self._analyze_sentiment(feedback["comments"])
            
            # Store sentiment analysis with the feedback
            if "sentiment_analysis" not in self.feedback_data:
                self.feedback_data["sentiment_analysis"] = []
            
            self.feedback_data["sentiment_analysis"].append({
                "user_id": user_id,
                "timestamp": feedback.get("timestamp", ""),
                "sentiment_score": sentiment_score,
                "comments": feedback["comments"]
            })
        
        # Save updated feedback
        self._save_feedback()
        
        # NEW: Clear cache to ensure fresh recommendations on next query
        self.recommendation_cache = {}
        
        return True
    
    def _analyze_sentiment(self, text):
        """
        NEW: Basic sentiment analysis for feedback comments
        
        Args:
            text (str): Feedback comment text
                
        Returns:
            float: Sentiment score (-1 to 1)
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count occurrences of positive, negative, and neutral words
        positive_count = sum(text_lower.count(word) for word in self.sentiment_analysis_params['positive_keywords'])
        negative_count = sum(text_lower.count(word) for word in self.sentiment_analysis_params['negative_keywords'])
        neutral_count = sum(text_lower.count(word) for word in self.sentiment_analysis_params['neutral_keywords'])
        
        # Apply weighting
        positive_score = positive_count * self.sentiment_analysis_params['weight_factors']['positive']
        negative_score = negative_count * self.sentiment_analysis_params['weight_factors']['negative']
        neutral_score = neutral_count * self.sentiment_analysis_params['weight_factors']['neutral']
        
        # Calculate total score (range from -1 to 1)
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0
        
        # Apply normalization to get a score between -1 and 1
        sentiment_score = (positive_score - negative_score) / (total_words + 0.001)
        sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to -1 to 1 range
        
        return sentiment_score
    
    def get_similar_users(self, user_data, limit=5):
        """
        Find users with similar profiles for collaborative filtering.
        
        Args:
            user_data (dict): User profile data to match against
            limit (int): Maximum number of similar users to return
                
        Returns:
            list: Similar user IDs and similarity scores
        """
        if not self.feedback_data["users"]:
            return []
        
        # Extract feature vectors for comparison
        target_vector = self._extract_user_features(user_data)
        similar_users = []
        
        for user in self.feedback_data["users"]:
            # Skip if no profile data
            if not user.get("profile"):
                continue
            
            # Calculate similarity score
            user_vector = self._extract_user_features(user["profile"])
            similarity = self._calculate_similarity(target_vector, user_vector)
            
            similar_users.append({
                "user_id": user["user_id"],
                "similarity": similarity
            })
        
        # Sort by similarity (highest first) and limit results
        similar_users.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_users[:limit]
    
    def _extract_user_features(self, user_data):
        """Extract normalized feature vector from user data"""
        features = {}
        
        # Age (normalize to 0-1 range)
        age = user_data.get("age", 30)
        features["age"] = min(1.0, age / 100)
        
        # Gender (one-hot encoding)
        gender = user_data.get("gender", "Male")
        features["gender_male"] = 1 if gender == "Male" else 0
        features["gender_female"] = 1 if gender == "Female" else 0
        features["gender_nonbinary"] = 1 if gender == "Non-binary" else 0
        
        # Body category
        height_m = user_data.get("height", 170) / 100
        weight_kg = user_data.get("weight", 70)
        bmi = weight_kg / (height_m ** 2)
        
        features["bmi"] = min(1.0, bmi / 40)  # Normalize BMI
        
        # Goal (one-hot encoding)
        goal = user_data.get("goal", "Weight Loss")
        features["goal_weight_loss"] = 1 if goal == "Weight Loss" else 0
        features["goal_muscle_gain"] = 1 if goal == "Muscle Gain" else 0
        features["goal_fitness"] = 1 if goal == "Improve Fitness" else 0
        features["goal_toning"] = 1 if goal == "Body Toning" else 0
        features["goal_endurance"] = 1 if goal == "Endurance Building" else 0
        
        # Activity level
        activity_level = user_data.get("activity_level", "Lightly Active")
        activity_scores = {
            "Sedentary": 0.2,
            "Lightly Active": 0.4,
            "Moderately Active": 0.6,
            "Very Active": 0.8,
            "Extremely Active": 1.0
        }
        features["activity_level"] = activity_scores.get(activity_level, 0.4)
        
        # NEW: Diet preference
        diet_type = user_data.get("diet_type", "Non-Vegetarian")
        features["diet_nonveg"] = 1 if diet_type == "Non-Vegetarian" else 0
        features["diet_vegetarian"] = 1 if diet_type == "Vegetarian" else 0
        features["diet_vegan"] = 1 if diet_type == "Vegan" else 0
        features["diet_keto"] = 1 if diet_type == "Keto" else 0
        features["diet_paleo"] = 1 if diet_type == "Paleo" else 0
        
        # NEW: Equipment availability
        has_equipment = user_data.get("home_equipment", ["None"])
        features["has_equipment"] = 0 if "None" in has_equipment and len(has_equipment) == 1 else 1
        
        # Return normalized feature vector
        return features
    
    def _calculate_similarity(self, vector1, vector2):
        """Calculate weighted cosine similarity between user vectors"""
        weighted_similarity = 0
        total_weight = 0
        
        # Process age similarity
        if "age" in vector1 and "age" in vector2:
            age_diff = abs(vector1["age"] - vector2["age"])
            age_similarity = 1 - min(1.0, age_diff)
            weighted_similarity += age_similarity * self.similarity_weights.get("age", 0.1)
            total_weight += self.similarity_weights.get("age", 0.1)
        
        # Process gender similarity
        if all(k in vector1 and k in vector2 for k in ["gender_male", "gender_female", "gender_nonbinary"]):
            gender_sim = 0
            for g in ["gender_male", "gender_female", "gender_nonbinary"]:
                if vector1[g] == vector2[g]:
                    gender_sim = 1
                    break
            
            weighted_similarity += gender_sim * self.similarity_weights.get("gender", 0.05)
            total_weight += self.similarity_weights.get("gender", 0.05)
        
        # Process BMI similarity
        if "bmi" in vector1 and "bmi" in vector2:
            bmi_diff = abs(vector1["bmi"] - vector2["bmi"])
            bmi_similarity = 1 - min(1.0, bmi_diff)
            weighted_similarity += bmi_similarity * self.similarity_weights.get("body_category", 0.15)
            total_weight += self.similarity_weights.get("body_category", 0.15)
        
        # Process goal similarity
        goal_similarity = 0
        goal_features = ["goal_weight_loss", "goal_muscle_gain", "goal_fitness", "goal_toning", "goal_endurance"]
        if all(k in vector1 and k in vector2 for k in goal_features):
            # Check if goals match
            for g in goal_features:
                if vector1[g] == 1 and vector2[g] == 1:
                    goal_similarity = 1
                    break
            
            weighted_similarity += goal_similarity * self.similarity_weights.get("goal", 0.25)
            total_weight += self.similarity_weights.get("goal", 0.25)
        
        # Process activity level similarity
        if "activity_level" in vector1 and "activity_level" in vector2:
            activity_diff = abs(vector1["activity_level"] - vector2["activity_level"])
            activity_similarity = 1 - min(1.0, activity_diff)
            weighted_similarity += activity_similarity * self.similarity_weights.get("activity_level", 0.15)
            total_weight += self.similarity_weights.get("activity_level", 0.15)
        
        # NEW: Process diet type similarity
        diet_similarity = 0
        diet_features = ["diet_nonveg", "diet_vegetarian", "diet_vegan", "diet_keto", "diet_paleo"]
        if all(k in vector1 and k in vector2 for k in diet_features):
            # Check if diet preferences match
            for d in diet_features:
                if vector1[d] == 1 and vector2[d] == 1:
                    diet_similarity = 1
                    break
            
            weighted_similarity += diet_similarity * 0.15
            total_weight += 0.15
        
        # NEW: Equipment similarity
        if "has_equipment" in vector1 and "has_equipment" in vector2:
            equipment_similarity = 1 if vector1["has_equipment"] == vector2["has_equipment"] else 0.5
            weighted_similarity += equipment_similarity * 0.05
            total_weight += 0.05
        
        # Normalize the final similarity score
        if total_weight > 0:
            return weighted_similarity / total_weight
        else:
            return 0
    
    def get_collaborative_recommendations(self, user_data, recommendation_type="workout", limit=5):
        """
        Get workout or meal recommendations based on similar users' ratings.
        
        Args:
            user_data (dict): User profile data
            recommendation_type (str): Type of recommendation ("workout" or "meal")
            limit (int): Maximum number of recommendations to return
                
        Returns:
            list: Recommended items with scores
        """
        # NEW: Check cache for existing recommendations
        cache_key = f"{hash(str(user_data))}-{recommendation_type}-{limit}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key].copy()
        
        # Get similar users
        similar_users = self.get_similar_users(user_data, limit=10)
        
        if not similar_users:
            return []
        
        # Get rated items based on recommendation type
        ratings_dict = self.feedback_data["workout_ratings"] if recommendation_type == "workout" else self.feedback_data["meal_ratings"]
        
        if not ratings_dict:
            return []
        
        # Calculate weighted ratings for items
        item_scores = {}
        
        for item_name, ratings in ratings_dict.items():
            item_scores[item_name] = {
                "weighted_score": 0,
                "weight_sum": 0
            }
            
            for rating_info in ratings:
                # Check if this rating is from a similar user
                user_id = rating_info.get("user_id")
                user_similarity = 0
                
                for similar_user in similar_users:
                    if similar_user["user_id"] == user_id:
                        user_similarity = similar_user["similarity"]
                        break
                
                if user_similarity > 0:
                    # Weight the rating by user similarity
                    rating = rating_info.get("rating", 3)
                    item_scores[item_name]["weighted_score"] += rating * user_similarity
                    item_scores[item_name]["weight_sum"] += user_similarity
        
        # Calculate final scores and filter items
        recommendations = []
        
        for item_name, score_info in item_scores.items():
            if score_info["weight_sum"] > 0:
                average_score = score_info["weighted_score"] / score_info["weight_sum"]
                
                # Only include items with good ratings
                if average_score >= 3.5:
                    recommendations.append({
                        "name": item_name,
                        "score": average_score
                    })
        
        # Sort by score (highest first) and limit results
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        result = recommendations[:limit]
        
        # NEW: Cache the results
        self.recommendation_cache[cache_key] = result.copy()
        
        return result
    
    def analyze_trends(self):
        """
        Analyze trends in user feedback to identify patterns.
        
        Returns:
            dict: Trend analysis results
        """
        trends = {
            "top_workouts": [],
            "top_meals": [],
            "user_segments": [],
            "insights": []
        }
        
        # Find top-rated workouts
        workout_avg_ratings = {}
        for workout_name, ratings in self.feedback_data["workout_ratings"].items():
            if ratings:
                avg_rating = sum(r.get("rating", 0) for r in ratings) / len(ratings)
                count = len(ratings)
                
                if count >= 3:  # Only include items with enough ratings
                    workout_avg_ratings[workout_name] = avg_rating
        
        # Sort workouts by rating
        sorted_workouts = [(name, rating) for name, rating in workout_avg_ratings.items()]
        sorted_workouts.sort(key=lambda x: x[1], reverse=True)
        
        trends["top_workouts"] = [{"name": name, "rating": rating} 
                                  for name, rating in sorted_workouts[:10]]
        
        # Find top-rated meals
        meal_avg_ratings = {}
        for meal_name, ratings in self.feedback_data["meal_ratings"].items():
            if ratings:
                avg_rating = sum(r.get("rating", 0) for r in ratings) / len(ratings)
                count = len(ratings)
                
                if count >= 3:  # Only include items with enough ratings
                    meal_avg_ratings[meal_name] = avg_rating
        
        # Sort meals by rating
        sorted_meals = [(name, rating) for name, rating in meal_avg_ratings.items()]
        sorted_meals.sort(key=lambda x: x[1], reverse=True)
        
        trends["top_meals"] = [{"name": name, "rating": rating} 
                               for name, rating in sorted_meals[:10]]
        
        # Analyze user segments using clustering
        if len(self.feedback_data["users"]) >= 10:  # Need enough users for meaningful clusters
            # Extract feature vectors for all users
            user_features = []
            user_ids = []
            
            for user in self.feedback_data["users"]:
                if user.get("profile"):
                    features = self._extract_user_features(user["profile"])
                    # Convert dict to list in a consistent order
                    feature_list = [
                        features.get("age", 0.3),
                        features.get("gender_male", 0),
                        features.get("gender_female", 0),
                        features.get("gender_nonbinary", 0),
                        features.get("bmi", 0.25),
                        features.get("goal_weight_loss", 0),
                        features.get("goal_muscle_gain", 0),
                        features.get("goal_fitness", 0),
                        features.get("goal_toning", 0),
                        features.get("goal_endurance", 0),
                        features.get("activity_level", 0.4)
                    ]
                    user_features.append(feature_list)
                    user_ids.append(user["user_id"])
            
            if user_features:
                # Fit clustering model
                features_array = np.array(user_features)
                clusters = self.user_clustering.fit_predict(features_array)
                
                # Analyze each cluster
                cluster_data = {}
                for i, cluster_id in enumerate(clusters):
                    if cluster_id not in cluster_data:
                        cluster_data[cluster_id] = {
                            "count": 0,
                            "users": [],
                            "features": {
                                "age": 0,
                                "bmi": 0,
                                "goals": {},
                                "activity_level": 0
                            }
                        }
                    
                    # Add user to cluster
                    cluster_data[cluster_id]["count"] += 1
                    cluster_data[cluster_id]["users"].append(user_ids[i])
                    
                    # Update cluster features
                    cluster_data[cluster_id]["features"]["age"] += user_features[i][0]
                    cluster_data[cluster_id]["features"]["bmi"] += user_features[i][4]
                    
                    # Process goals
                    goal_index = -1
                    max_val = 0
                    for j in range(5, 10):
                        if user_features[i][j] > max_val:
                            max_val = user_features[i][j]
                            goal_index = j
                    
                    goal_name = ["weight_loss", "muscle_gain", "fitness", "toning", "endurance"][goal_index - 5]
                    if goal_name not in cluster_data[cluster_id]["features"]["goals"]:
                        cluster_data[cluster_id]["features"]["goals"][goal_name] = 0
                    
                    cluster_data[cluster_id]["features"]["goals"][goal_name] += 1
                    cluster_data[cluster_id]["features"]["activity_level"] += user_features[i][10]
                
                # Calculate averages and determine dominant characteristics
                for cluster_id, data in cluster_data.items():
                    count = data["count"]
                    if count > 0:
                        data["features"]["age"] /= count
                        data["features"]["bmi"] /= count
                        data["features"]["activity_level"] /= count
                        
                        # Find dominant goal
                        dominant_goal = max(data["features"]["goals"].items(), key=lambda x: x[1])
                        data["features"]["dominant_goal"] = dominant_goal[0]
                        
                        # Create segment description
                        age_desc = "younger" if data["features"]["age"] < 0.35 else "middle-aged" if data["features"]["age"] < 0.6 else "older"
                        bmi_desc = "underweight" if data["features"]["bmi"] < 0.2 else "normal weight" if data["features"]["bmi"] < 0.25 else "overweight" if data["features"]["bmi"] < 0.3 else "obese"
                        activity_desc = "sedentary" if data["features"]["activity_level"] < 0.3 else "lightly active" if data["features"]["activity_level"] < 0.5 else "moderately active" if data["features"]["activity_level"] < 0.7 else "very active"
                        
                        data["description"] = f"Segment {cluster_id+1}: {count} {age_desc}, {bmi_desc}, {activity_desc} users focused on {dominant_goal[0].replace('_', ' ')}"
                
                # Add segments to trends
                trends["user_segments"] = [{"id": k, "description": v["description"], "count": v["count"]} 
                                           for k, v in cluster_data.items()]
        
        # Generate insights
        insights = []
        
        # Insight 1: Workout preferences by goal
        goal_workout_preferences = {}
        for user in self.feedback_data["users"]:
            if not user.get("profile") or not user.get("feedback_history"):
                continue
            
            goal = user["profile"].get("goal", "Weight Loss")
            if goal not in goal_workout_preferences:
                goal_workout_preferences[goal] = {}
            
            # Collect workout ratings
            for feedback in user["feedback_history"]:
                if "workout_ratings" in feedback:
                    for workout, rating in feedback["workout_ratings"].items():
                        if workout not in goal_workout_preferences[goal]:
                            goal_workout_preferences[goal][workout] = []
                        
                        goal_workout_preferences[goal][workout].append(rating)
        
        # Find top workout for each goal
        for goal, workouts in goal_workout_preferences.items():
            top_workout = None
            top_rating = 0
            min_ratings = 3  # Require at least 3 ratings
            
            for workout, ratings in workouts.items():
                if len(ratings) >= min_ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    if avg_rating > top_rating:
                        top_rating = avg_rating
                        top_workout = workout
            
            if top_workout:
                insights.append(f"Users focused on {goal} prefer '{top_workout}' workouts")
        
        # Insight 2: Meal preferences by diet type
        diet_meal_preferences = {}
        for user in self.feedback_data["users"]:
            if not user.get("profile") or not user.get("feedback_history"):
                continue
            
            diet_type = user["profile"].get("diet_type", "Non-Vegetarian")
            if diet_type not in diet_meal_preferences:
                diet_meal_preferences[diet_type] = {}
            
            # Collect meal ratings
            for feedback in user["feedback_history"]:
                if "meal_ratings" in feedback:
                    for meal, rating in feedback["meal_ratings"].items():
                        if meal not in diet_meal_preferences[diet_type]:
                            diet_meal_preferences[diet_type][meal] = []
                        
                        diet_meal_preferences[diet_type][meal].append(rating)
        
        # Find top meal for each diet type
        for diet, meals in diet_meal_preferences.items():
            top_meal = None
            top_rating = 0
            min_ratings = 3  # Require at least 3 ratings
            
            for meal, ratings in meals.items():
                if len(ratings) >= min_ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    if avg_rating > top_rating:
                        top_rating = avg_rating
                        top_meal = meal
            
            if top_meal:
                insights.append(f"{diet} users prefer '{top_meal}' meals")
        
        # NEW: Analyze sentiment trends if available
        if "sentiment_analysis" in self.feedback_data and len(self.feedback_data["sentiment_analysis"]) >= 5:
            sentiment_data = self.feedback_data["sentiment_analysis"]
            
            # Calculate average sentiment
            avg_sentiment = sum(item["sentiment_score"] for item in sentiment_data) / len(sentiment_data)
            
            # Generate insight based on sentiment
            if avg_sentiment > 0.5:
                insights.append(f"User feedback is very positive (sentiment score: {avg_sentiment:.2f}), indicating high satisfaction with plans")
            elif avg_sentiment > 0.2:
                insights.append(f"User feedback is generally positive (sentiment score: {avg_sentiment:.2f})")
            elif avg_sentiment > -0.2:
                insights.append(f"User feedback is neutral (sentiment score: {avg_sentiment:.2f}), suggesting room for improvement")
            else:
                insights.append(f"User feedback trends negative (sentiment score: {avg_sentiment:.2f}), indicating potential issues to address")
            
            # Look for common themes in negative feedback
            if avg_sentiment < 0:
                negative_comments = [item["comments"] for item in sentiment_data if item["sentiment_score"] < 0]
                if negative_comments:
                    combined_text = " ".join(negative_comments).lower()
                    
                    # Simple keyword analysis
                    problem_keywords = {
                        "too hard": "workout difficulty",
                        "too easy": "insufficient challenge",
                        "confusing": "clarity issues",
                        "too much time": "time constraints",
                        "equipment": "equipment concerns",
                        "complicated": "complexity issues",
                        "meals": "meal plan issues",
                        "diet": "dietary concerns",
                        "hungry": "calorie concerns",
                        "progress": "progress tracking issues"
                    }
                    
                    found_issues = []
                    for keyword, issue in problem_keywords.items():
                        if keyword in combined_text:
                            found_issues.append(issue)
                    
                    if found_issues:
                        insights.append(f"Common concerns in negative feedback: {', '.join(found_issues)}")
        
        trends["insights"] = insights
        return trends
    
    def update_recommendation_models(self):
        """
        Update machine learning models based on accumulated feedback.
        
        Returns:
            bool: Success status
        """
        # This method would be expanded in a production environment
        # to retrain models based on accumulated feedback
        
        # For now, just update the feature weights based on rating patterns
        if len(self.feedback_data["users"]) >= 20:  # Need enough data
            # Analyze which features most correlate with high ratings
            feature_correlations = {
                'age': 0.0,
                'gender': 0.0,
                'body_category': 0.0,
                'goal': 0.0,
                'activity_level': 0.0,
                'fitness_level': 0.0
            }
            
            # This is a simplified approach - in a real system, you would
            # use proper statistical methods to compute correlations
            
            # Update similarity weights based on correlations
            for feature, correlation in feature_correlations.items():
                # Adjust weight based on correlation strength
                # (this is a simplified approach)
                if abs(correlation) > 0.3:
                    self.similarity_weights[feature] *= (1 + abs(correlation))
            
            # Normalize weights to sum to 1
            total_weight = sum(self.similarity_weights.values())
            if total_weight > 0:
                for feature in self.similarity_weights:
                    self.similarity_weights[feature] /= total_weight
            
            return True
        
        return False
    
    def get_personalized_adjustment_recommendations(self, user_id, current_progress):
        """
        NEW: Generate personalized recommendations for plan adjustments based on progress.
        
        Args:
            user_id (str): User identifier
            current_progress (dict): Current progress metrics
            
        Returns:
            dict: Adjustment recommendations
        """
        # Find the user
        user_data = None
        feedback_history = []
        
        for user in self.feedback_data["users"]:
            if user.get("user_id") == user_id:
                user_data = user.get("profile", {})
                feedback_history = user.get("feedback_history", [])
                break
        
        if not user_data:
            return {"success": False, "message": "User not found"}
        
        # Extract goal
        goal = user_data.get("goal", "Weight Loss")
        
        # Analyze progress based on goal
        recommendations = {
            "workout_adjustments": [],
            "nutrition_adjustments": [],
            "adherence_suggestions": []
        }
        
        # Calculate workout adherence
        workout_compliance = current_progress.get("workout_compliance", 0)
        calorie_compliance = current_progress.get("calorie_compliance", 0)
        
        # Weight change analysis
        if "weight" in current_progress:
            current_weight = current_progress.get("weight", user_data.get("weight", 0))
            initial_weight = user_data.get("weight", 0)
            target_weight = user_data.get("target_weight", 0)
            
            if goal == "Weight Loss" and initial_weight > 0 and target_weight > 0:
                # Calculate progress towards weight loss goal
                total_to_lose = initial_weight - target_weight
                lost_so_far = initial_weight - current_weight
                
                if total_to_lose > 0:
                    progress_pct = (lost_so_far / total_to_lose) * 100
                    
                    # Categorize progress
                    if lost_so_far <= 0:
                        # Weight not decreasing
                        recommendations["nutrition_adjustments"].append(
                            "Increase your caloric deficit by 200-300 calories per day"
                        )
                        recommendations["workout_adjustments"].append(
                            "Add 10-15 minutes of moderate cardio after strength workouts for increased calorie burn"
                        )
                    elif progress_pct < 25:
                        # Early progress
                        recommendations["nutrition_adjustments"].append(
                            "Continue with your current calorie target and focus on protein-rich foods"
                        )
                        recommendations["workout_adjustments"].append(
                            "Maintain your current workout intensity and try to increase weights slightly each week"
                        )
                    elif progress_pct < 50:
                        # Good progress, may need adjustments for continued success
                        recommendations["nutrition_adjustments"].append(
                            "Recalculate calorie needs based on your new weight to maintain fat loss momentum"
                        )
                    elif progress_pct < 75:
                        # Approaching goal, focus on habits
                        recommendations["nutrition_adjustments"].append(
                            "Begin practicing maintenance-level eating several days per week"
                        )
                        recommendations["workout_adjustments"].append(
                            "Transition to a more maintenance-focused routine with emphasis on strength training"
                        )
                    else:
                        # Near goal
                        recommendations["nutrition_adjustments"].append(
                            "Gradually increase calories to maintenance level to prevent rebound weight gain"
                        )
            
            elif goal == "Muscle Gain" and initial_weight > 0 and target_weight > 0:
                # Calculate progress towards weight gain goal
                total_to_gain = target_weight - initial_weight
                gained_so_far = current_weight - initial_weight
                
                if total_to_gain > 0:
                    progress_pct = (gained_so_far / total_to_gain) * 100
                    
                    # Categorize progress
                    if gained_so_far <= 0:
                        # Weight not increasing
                        recommendations["nutrition_adjustments"].append(
                            "Increase caloric intake by 300-500 calories per day"
                        )
                        recommendations["workout_adjustments"].append(
                            "Focus on compound exercises with progressive overload"
                        )
                    elif progress_pct < 30:
                        # Early progress
                        recommendations["nutrition_adjustments"].append(
                            "Maintain your current calorie surplus and ensure adequate protein intake"
                        )
                    elif progress_pct < 60:
                        # Good progress
                        recommendations["workout_adjustments"].append(
                            "Consider increasing workout volume by adding an additional set to your main lifts"
                        )
                    else:
                        # Approaching goal, refine body composition
                        recommendations["nutrition_adjustments"].append(
                            "Consider a slight reduction in calories on rest days to minimize fat gain while maintaining muscle growth"
                        )
        
        # Compliance-based recommendations
        if workout_compliance < 60:
            recommendations["adherence_suggestions"].append(
                "Try scheduling workouts at consistent times and setting phone reminders"
            )
            recommendations["workout_adjustments"].append(
                "Consider shorter, more frequent workouts if time is a barrier"
            )
        
        if calorie_compliance < 60:
            recommendations["adherence_suggestions"].append(
                "Meal prep on weekends to make healthy eating more convenient"
            )
            recommendations["nutrition_adjustments"].append(
                "Simplify your meal plan with fewer ingredients and easier recipes"
            )
        
        return recommendations
