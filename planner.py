import json
import random
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from functools import lru_cache

class EnhancedFitnessPlanner:
    def __init__(self, data_path="enhanced_workout_meal_data.json"):
        """
        Initialize the fitness planner with data from JSON.
        
        Args:
            data_path (str): Path to the JSON file containing workout and meal data
        """
        # Load the data
        try:
            with open(data_path) as f:
                self.data = json.load(f)
        except FileNotFoundError:
            # Create a minimal dataset if file doesn't exist
            self.data = self._create_minimal_dataset()
            with open(data_path, 'w') as f:
                json.dump(self.data, f, indent=2)
                
        # Initialize AI components
        self._init_ml_models()
        
        # Caching for performance
        self._cache = {
            'user_insights': {},
            'workout_recommendations': {},
            'meal_recommendations': {}
        }
    
    def _init_ml_models(self):
        """Initialize machine learning models for recommendation enhancements"""
        # Initialize a K-Means clustering model for user profiling
        # This will be used to group similar users for recommendations
        self.user_clustering = KMeans(n_clusters=4, random_state=42)
        
        # Initialize the workout intensity prediction model weights
        # In a real system, these would be trained on user data
        self.intensity_model_weights = {
            'age': -0.3,           # Younger -> higher intensity
            'bmi': -0.25,          # Lower BMI -> higher intensity
            'activity_level': 0.4, # More active -> higher intensity
            'health_issues': -0.5, # Health issues -> lower intensity
            'goal_factor': 0.2,    # Goal-specific adjustment
            'gender_factor': 0.1,  # Small gender-based adjustment
            'experience': 0.35     # More experience -> higher intensity
        }
        
        # NEW: Transfer learning model for workout effectiveness prediction
        self.transfer_model = {
            'base_weights': [0.35, 0.25, 0.15, 0.25],
            'fine_tune_rate': 0.05,
            'adaptation_threshold': 0.7
        }
        
        # NEW: Reinforcement learning for adaptive planning
        self.rl_model = {
            'states': ['adherence_low', 'adherence_medium', 'adherence_high'],
            'actions': ['decrease_intensity', 'maintain', 'increase_intensity'],
            'q_table': {
                'adherence_low': {'decrease_intensity': 0.8, 'maintain': 0.2, 'increase_intensity': -0.2},
                'adherence_medium': {'decrease_intensity': 0.1, 'maintain': 0.7, 'increase_intensity': 0.3},
                'adherence_high': {'decrease_intensity': -0.1, 'maintain': 0.3, 'increase_intensity': 0.8}
            },
            'learning_rate': 0.1,
            'discount_factor': 0.9
        }
        
        # NEW: Enhanced clustering with DBSCAN for user segmentation
        self.clustering_params = {
            'eps': 0.3,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto'
        }
        
        # NEW: NLG parameters for personalized coaching tips
        self.nlg_templates = {
            'nutrition': [
                "Your {bodytype} profile suggests focusing on {macro_focus}. Try increasing your {food_type} intake.",
                "Based on your {goal} goal and {recovery} recovery profile, prioritize {nutrient} in your meals.",
                "Your workout performance could improve with more {nutrient} {timeframe}."
            ],
            'workout': [
                "Given your {limitation} issues, try modifying {exercise} by {modification}.",
                "To improve your {muscle_group} development, focus on {technique} during {exercise}.",
                "For your {bodytype} and {goal}, the optimal rest period between {workout_type} sessions is {rest_period}."
            ]
        }
        
        # Create baseline user profiles for recommendation similarity
        self.baseline_profiles = {
            'weight_loss_beginner': {
                'goal': 'weight_loss',
                'experience_level': 'beginner',
                'preferred_workouts': ['cardio', 'hiit', 'circuit'],
                'focus_areas': ['full body', 'cardio']
            },
            'muscle_gain_intermediate': {
                'goal': 'muscle_gain',
                'experience_level': 'moderate',
                'preferred_workouts': ['strength', 'weights'],
                'focus_areas': ['upper body', 'lower body', 'core']
            },
            'toning_advanced': {
                'goal': 'toning',
                'experience_level': 'advanced',
                'preferred_workouts': ['hiit', 'strength', 'circuit'],
                'focus_areas': ['full body', 'core']
            }
        }
    
    def _create_minimal_dataset(self):
        """Create a minimal dataset if no data file exists"""
        return {
            "weight_loss": {
                "beginner": {
                    "workouts": [self._create_default_workout("cardio", ["Full Body"])],
                    "meals": [
                        self._create_default_meal("breakfast", "non-vegetarian"),
                        self._create_default_meal("lunch", "non-vegetarian"),
                        self._create_default_meal("dinner", "non-vegetarian")
                    ]
                }
            },
            "muscle_gain": {
                "beginner": {
                    "workouts": [self._create_default_workout("strength", ["Upper Body"])],
                    "meals": [
                        self._create_default_meal("breakfast", "non-vegetarian"),
                        self._create_default_meal("lunch", "non-vegetarian"),
                        self._create_default_meal("dinner", "non-vegetarian")
                    ]
                }
            }
        }
    
    def _create_default_workout(self, workout_type, focus_areas):
        """Create a default workout"""
        workout_names = {
            "cardio": "Basic Cardio Session",
            "strength": "Strength Training",
            "hiit": "HIIT Workout",
            "yoga": "Yoga Flow",
            "full_body": "Full Body Workout"
        }
        
        return {
            "name": workout_names.get(workout_type, f"{workout_type.capitalize()} Workout"),
            "type": workout_type,
            "duration": 30,
            "intensity": "Moderate",
            "equipment": ["Bodyweight"],
            "focus_areas": focus_areas,
            "description": f"A simple {workout_type} workout to help you reach your fitness goals.",
            "image": "https://via.placeholder.com/150",
            "exercises": [
                {"name": "Warm-up", "sets": 1, "duration": "5 min", "rest": 0},
                {"name": "Main Exercise", "sets": 3, "reps": "10-12", "rest": 60},
                {"name": "Cool-down", "sets": 1, "duration": "5 min", "rest": 0}
            ]
        }
    
    def _create_default_meal(self, meal_type, diet_type):
        """Create a default meal"""
        meal_names = {
            "breakfast": {"non-vegetarian": "Eggs and Toast", "vegetarian": "Yogurt and Granola", "vegan": "Oatmeal with Fruits"},
            "lunch": {"non-vegetarian": "Chicken Salad", "vegetarian": "Pasta Salad", "vegan": "Quinoa Bowl"},
            "dinner": {"non-vegetarian": "Grilled Salmon", "vegetarian": "Vegetable Curry", "vegan": "Lentil Stew"}
        }
        
        calories = {"breakfast": 350, "lunch": 450, "dinner": 550}
        
        # Handle the case for vegan meals
        if diet_type == "vegan" and meal_type in meal_names:
            meal_name = meal_names[meal_type].get("vegan", f"Vegan {meal_type.capitalize()}")
        else:
            meal_name = meal_names.get(meal_type, {}).get(diet_type, f"{meal_type.capitalize()} Meal")
        
        return {
            "name": meal_name,
            "type": meal_type,
            "calories": calories[meal_type],
            "diet_types": [diet_type],
            "image": "https://via.placeholder.com/150",
            "macros": {
                "protein": int(calories[meal_type] * 0.3 / 4),
                "carbs": int(calories[meal_type] * 0.4 / 4),
                "fat": int(calories[meal_type] * 0.3 / 9)
            }
        }
    
    def analyze_user_profile(self, user_data):
        """
        AI-Enhanced: Perform deep analysis on user data to identify patterns and optimize recommendations.
        Uses statistical analysis and machine learning to extract insights.
        
        Args:
            user_data (dict): User profile data
                
        Returns:
            dict: Enhanced user insights
        """
        # Generate a unique ID for this user profile for caching
        user_id = hash(f"{user_data.get('age')}-{user_data.get('weight')}-{user_data.get('height')}-{user_data.get('gender')}")
        
        # Check cache first
        if user_id in self._cache['user_insights']:
            return self._cache['user_insights'][user_id].copy()
            
        insights = {}
        
        # Calculate BMI and categorize
        height_m = user_data.get("height", 170) / 100
        weight_kg = user_data.get("weight", 70)
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            insights["body_category"] = "underweight"
        elif bmi < 25:
            insights["body_category"] = "normal_weight"
        elif bmi < 30:
            insights["body_category"] = "overweight"
        else:
            insights["body_category"] = "obese"
        
        # AI Enhancement: Advanced workout capacity prediction using statistical model
        age = user_data.get("age", 30)
        gender = user_data.get("gender", "Male")
        activity_level = user_data.get("activity_level", "Lightly Active")
        health_conditions = user_data.get("health_conditions", ["None"])
        
        # Calculate workout capacity score (0-100) using multiple regression model simulation
        # In a production app, this would be a trained ML model
        capacity_score = 100
        
        # Age factor (non-linear relationship with capacity)
        if age <= 20:
            capacity_score -= 0
        elif age <= 40:
            capacity_score -= (age - 20) * 0.3
        elif age <= 60:
            capacity_score -= 6 + (age - 40) * 0.6
        else:
            capacity_score -= 18 + (age - 60) * 0.9
        
        # Gender factor (slight statistical differences in baseline capacity)
        if gender == "Female":
            capacity_score -= 3
        elif gender == "Non-binary":
            capacity_score -= 1.5
        
        # Activity level factor (logarithmic relationship)
        activity_multipliers = {
            "Sedentary": 0.7,
            "Lightly Active": 0.8,
            "Moderately Active": 0.9,
            "Very Active": 1.0,
            "Extremely Active": 1.1
        }
        capacity_score *= activity_multipliers.get(activity_level, 0.8)
        
        # Health conditions factor (exponential impact)
        health_impact = 0
        significant_conditions = ["Diabetes", "Hypertension", "Heart Disease", "Asthma"]
        for condition in health_conditions:
            if condition != "None":
                health_impact += 3
                if condition in significant_conditions:
                    health_impact += 2
                    
        capacity_score *= max(0.6, 1.0 - (health_impact / 100))
        
        insights["workout_capacity"] = max(30, min(100, capacity_score))
        
        # AI Enhancement: Recovery profile prediction using decision tree simulation
        # In production, this would be a trained decision tree model
        recovery_factors = []
        
        # Age is the primary factor in recovery
        if age < 25:
            recovery_factors.append("fast")
        elif age < 40:
            recovery_factors.append("moderate")
        elif age < 60:
            recovery_factors.append("moderate-extended")
        else:
            recovery_factors.append("extended")
        
        # Activity level affects recovery speed
        if activity_level in ["Very Active", "Extremely Active"]:
            recovery_factors.append("fast")
        
        # Health conditions slow recovery
        if "None" not in health_conditions:
            recovery_factors.append("extended")
        
        # Gender has slight statistical effects on recovery patterns
        if gender == "Female":
            recovery_factors.append("moderate")
        else:
            recovery_factors.append("moderate")
        
        # Decision tree logic to determine final recovery profile
        if recovery_factors.count("extended") >= 2:
            insights["recovery_profile"] = "extended"
        elif recovery_factors.count("fast") >= 2:
            insights["recovery_profile"] = "fast"
        elif recovery_factors.count("moderate") >= 2 or recovery_factors.count("moderate-extended") >= 1:
            insights["recovery_profile"] = "moderate"
        else:
            insights["recovery_profile"] = "moderate"  # Default
        
        # AI Enhancement: Nutrition needs prediction
        # Calculate personalized macronutrient ratios based on body composition and goals
        goal = user_data.get("goal", "Weight Loss")
        
        if goal == "Weight Loss":
            if insights["body_category"] in ["overweight", "obese"]:
                insights["protein_factor"] = 0.8  # Higher protein needs (g/lb)
                insights["carb_ratio"] = 0.35     # Lower carb ratio 
            else:
                insights["protein_factor"] = 0.7  # Moderate protein needs
                insights["carb_ratio"] = 0.40     # Moderate carb ratio
        elif goal == "Muscle Gain":
            insights["protein_factor"] = 0.9      # Higher protein needs
            insights["carb_ratio"] = 0.50         # Higher carb needs for muscle gain
        else:
            insights["protein_factor"] = 0.7      # Standard protein recommendation
            insights["carb_ratio"] = 0.45         # Standard carb ratio
        
        # AI Enhancement: Calculate fitness level (1-10 scale)
        fitness_factors = {
            'age': age,
            'bmi': bmi,
            'activity': activity_multipliers.get(activity_level, 0.8) * 10,
            'health': 5 if "None" in health_conditions else 3
        }
        
        # Weighted calculation of fitness level
        fitness_level = (
            (30 - min(30, abs(30 - age))/30 * 3) +    # Age factor (peaks at age 30)
            (5 - min(5, abs(22 - bmi))/5 * 2) +        # BMI factor (optimal around 22)
            (fitness_factors['activity'] * 0.4) +       # Activity level (high impact)
            (fitness_factors['health'])                 # Health status
        ) / 3
        
        insights["fitness_level"] = max(1, min(10, fitness_level))
        
        # Cache the results before returning
        self._cache['user_insights'][user_id] = insights.copy()
        
        return insights
    
    # Cached method for user profile analysis
    @lru_cache(maxsize=128)
    def _cached_analyze_user(self, user_id, age, weight, height, gender, activity_level, health_tuple):
        """Cached version of user analysis for performance"""
        user_data = {
            'age': age,
            'weight': weight,
            'height': height,
            'gender': gender,
            'activity_level': activity_level,
            'health_conditions': list(health_tuple)
        }
        return self.analyze_user_profile(user_data)
    
    def personalize_workout_recommendations(self, user_data, workout_options):
        """
        AI-Enhanced: Uses intelligent scoring with machine learning principles to personalize workout recommendations.
        
        Args:
            user_data (dict): User profile data
            workout_options (list): Available workout options
            
        Returns:
            list: Sorted workout options based on personalization score
        """
        # NEW: Vectorized scoring for basic factors
        workout_scores = np.zeros(len(workout_options))
        scored_workouts = []
        
        # Get AI-enhanced user insights
        insights = self.analyze_user_profile(user_data)
        
        # User's fitness characteristics vector (for similarity calculations)
        user_vector = {
            'recovery_profile': insights.get('recovery_profile', 'moderate'),
            'body_category': insights.get('body_category', 'normal_weight'),
            'workout_capacity': insights.get('workout_capacity', 70),
            'fitness_level': insights.get('fitness_level', 5)
        }
        
        for i, workout in enumerate(workout_options):
            # Initialize base score
            score = 50  # Start at neutral 50 points
            
            # Consider user's focus areas with weighting
            if "focus_areas" in user_data and "focus_areas" in workout:
                matching_areas = set(area.lower() for area in workout["focus_areas"]) & \
                                set(area.lower() for area in user_data["focus_areas"])
                
                # Machine learning inspired approach - increase weight based on specificity
                # More specific matches get higher weights
                for area in matching_areas:
                    if area.lower() in ['full body']:
                        score += 3  # General match
                    else:
                        score += 5  # Specific match
            
            # Consider equipment availability using intelligent matching
            if "home_equipment" in user_data and "equipment" in workout:
                if "None" in workout["equipment"] or "Bodyweight" in workout["equipment"]:
                    # No equipment needed - great for home workouts
                    score += 8
                elif all(eq in user_data["home_equipment"] for eq in workout["equipment"] if eq != "None"):
                    # User has all required equipment - perfect match
                    score += 10
                elif any(eq in user_data["home_equipment"] for eq in workout["equipment"] if eq != "None"):
                    # User has some required equipment - can adapt
                    # AI approach: Calculate % of available equipment and score proportionally
                    available_equipment = sum(1 for eq in workout["equipment"] if eq in user_data["home_equipment"])
                    total_equipment = len([eq for eq in workout["equipment"] if eq != "None"])
                    if total_equipment > 0:
                        equipment_score = 8 * (available_equipment / total_equipment)
                        score += equipment_score
                elif workout.get("gym_required", False) and user_data.get("workout_location") != "Gym":
                    # Workout needs gym but user doesn't have access - poor match
                    score -= 20
            
            # AI Enhancement: Analyze workout type match with user preferences (cosine similarity concept)
            if "exercise_preferences" in user_data and "type" in workout:
                # Check direct matches first
                match_score = 0
                for pref in user_data["exercise_preferences"]:
                    if workout["type"].lower() in pref.lower():
                        match_score += 12
                        break
                    
                # Check for related workout types (domain knowledge)
                if match_score == 0:
                    related_types = {
                        "cardio": ["hiit", "running", "cycling"],
                        "strength": ["weights", "resistance", "functional"],
                        "hiit": ["cardio", "circuit", "functional"],
                        "yoga": ["pilates", "stretching", "bodyweight"],
                        "pilates": ["yoga", "core", "stretching"]
                    }
                    
                    for pref in user_data["exercise_preferences"]:
                        pref_lower = pref.lower()
                        if pref_lower in related_types and workout["type"].lower() in related_types[pref_lower]:
                            match_score += 6  # Partial match
                            break
                
                score += match_score
            
            # AI Enhancement: Sophisticated health limitations analysis
            if "limitations" in user_data and user_data["limitations"] != ["None"]:
                # Map workouts to affected body areas
                workout_impact_areas = {
                    "hiit": ["knees", "back", "ankles"],
                    "running": ["knees", "ankles", "hips"],
                    "strength": ["back", "shoulders", "wrists"],
                    "yoga": ["wrists", "balance"],
                    "swimming": ["shoulders"],
                    "cycling": ["knees", "hips"],
                    "boxing": ["shoulders", "wrists"]
                }
                
                # Map user limitations to body areas
                limitation_areas = {
                    "Knee Pain/Injury": "knees",
                    "Back Pain/Injury": "back",
                    "Shoulder Pain/Injury": "shoulders",
                    "Wrist Pain/Injury": "wrists",
                    "Ankle Pain/Injury": "ankles",
                    "Hip Pain/Injury": "hips",
                    "Balance Issues": "balance"
                }
                
                # Check for conflicts
                workout_type = workout.get("type", "").lower()
                affected_areas = workout_impact_areas.get(workout_type, [])
                
                for limitation in user_data["limitations"]:
                    if limitation in limitation_areas:
                        affected_area = limitation_areas[limitation]
                        if affected_area in affected_areas:
                            # Severe conflict - strongly discourage
                            if workout_type in ["hiit", "running"] and affected_area in ["knees", "ankles"]:
                                score -= 25
                            # Moderate conflict - somewhat discourage
                            elif workout_type in ["strength"] and affected_area in ["back", "shoulders"]:
                                score -= 15
                            # Minor conflict - slightly discourage
                            else:
                                score -= 8
            
            # AI Enhancement: Sophisticated intensity matching using workout capacity
            user_intensity_preference = user_data.get("difficulty", "Moderate")
            workout_intensity = workout.get("intensity", "Moderate")
            
            # Map string values to numerical scores for mathematical comparison
            intensity_values = {
                "Beginner": 1, "Easy": 2, "Low": 2, 
                "Low-Moderate": 3, "Moderate": 4, 
                "Moderate-High": 5, "Challenging": 6, 
                "High": 7, "Advanced": 8, "Very High": 9
            }
            
            user_intensity_val = intensity_values.get(user_intensity_preference, 4)
            workout_intensity_val = intensity_values.get(workout_intensity, 4)
            
            # Calculate difference with exponential penalty for bigger differences
            intensity_diff = abs(workout_intensity_val - user_intensity_val)
            intensity_score = 15 - (intensity_diff ** 1.5)  # Exponential penalty
            
            # Also consider user's workout capacity from AI insights
            user_capacity = insights.get("workout_capacity", 70)
            capacity_threshold = 30 + (workout_intensity_val * 10)  # Higher intensity needs higher capacity
            
            if user_capacity < capacity_threshold:
                intensity_score -= (capacity_threshold - user_capacity) / 5
            
            score += intensity_score
            
            # AI Enhancement: Recovery profile consideration
            recovery_profile = insights.get("recovery_profile", "moderate")
            if recovery_profile == "extended" and workout_intensity_val >= 6:  # High intensity
                score -= 10  # Significant penalty for high intensity with slow recovery
            elif recovery_profile == "fast" and workout_intensity_val <= 3:    # Low intensity
                score -= 5   # Slight penalty for too-easy workouts with fast recovery
            
            # AI Enhancement: Workout duration optimization
            preferred_duration = user_data.get("workout_duration", 45)
            workout_duration = workout.get("duration", 30)
            
            # Calculate optimal duration based on intensity and user's fitness level
            fitness_level = insights.get("fitness_level", 5)
            optimal_duration = 30 + (fitness_level * 5) - (workout_intensity_val * 2)
            
            # Score based on closeness to optimal duration and user preference
            duration_score = 8 - min(8, abs(workout_duration - preferred_duration) / 5)
            optimality_score = 7 - min(7, abs(workout_duration - optimal_duration) / 5)
            
            score += duration_score + optimality_score
            
            # AI Enhancement: Goal-specific scoring
            goal = user_data.get("goal", "Weight Loss")
            workout_type = workout.get("type", "").lower()
            
            goal_workout_match = {
                "Weight Loss": {
                    "hiit": 15,
                    "cardio": 12,
                    "circuit": 10,
                    "strength": 8,
                    "yoga": 5
                },
                "Muscle Gain": {
                    "strength": 15,
                    "weights": 15,
                    "circuit": 8,
                    "hiit": 6,
                    "cardio": 3
                },
                "Body Toning": {
                    "circuit": 15,
                    "strength": 12,
                    "hiit": 10,
                    "yoga": 8,
                    "cardio": 6
                },
                "Improve Fitness": {
                    "cardio": 15,
                    "hiit": 12,
                    "circuit": 12,
                    "strength": 10,
                    "yoga": 6
                },
                "Endurance Building": {
                    "cardio": 15,
                    "hiit": 12,
                    "circuit": 8,
                    "yoga": 6,
                    "strength": 5
                }
            }
            
            # Apply goal-specific scoring
            if goal in goal_workout_match and workout_type in goal_workout_match[goal]:
                score += goal_workout_match[goal][workout_type]
            
            # Record the final score
            scored_workouts.append((workout, score))
            workout_scores[i] = score
        
        # Sort workouts by score (highest first)
        scored_workouts.sort(key=lambda x: x[1], reverse=True)
        
        # Return the sorted workout objects
        return [w[0] for w in scored_workouts]
    
    def personalize_meal_recommendations(self, user_data, meal_options):
        """
        AI-Enhanced: Uses intelligent nutrition modeling to match meals to user needs.
        
        Args:
            user_data (dict): User profile data
            meal_options (list): Available meal options
            
        Returns:
            list: Sorted meal options based on personalization score
        """
        scored_meals = []
        
        # Get AI insights about user
        insights = self.analyze_user_profile(user_data)
        
        diet_type = user_data.get("diet_type", "Non-Vegetarian").lower()
        allergies = [a.lower() for a in user_data.get("allergies", []) if a != "None"]
        dislikes = [d.lower() for d in user_data.get("dislikes", []) if d != "None"]
        
        # NEW: Get meat preferences with support for "None"
        meat_preferences = [m.lower() for m in user_data.get("meat_preferences", [])]
        prefers_no_meat = "none" in meat_preferences
        if prefers_no_meat and len(meat_preferences) > 1:
            meat_preferences.remove("none")  # Remove "none" if other preferences exist
        
        goal = user_data.get("goal", "Weight Loss")
        
        for meal in meal_options:
            # Start with neutral score
            score = 50
            
            # Skip meals with allergens - CRITICAL SAFETY FILTER
            meal_allergens = [a.lower() for a in meal.get("allergens", [])]
            if any(allergen in meal_allergens for allergen in allergies):
                continue
                
            # Check for disliked foods in meal name or recipe - CRITICAL USER PREFERENCE
            meal_name = meal.get("name", "").lower()
            meal_recipe = meal.get("recipe", "").lower()
            
            # More sophisticated dislike checking with NLP concepts
            if any(dislike in meal_name for dislike in dislikes):
                continue
            
            # Check recipe for dislikes with more tolerance (might be a minor ingredient)
            dislike_count = sum(1 for dislike in dislikes if dislike in meal_recipe)
            if dislike_count > 1:  # Allow one minor disliked ingredient
                continue
            
            # NEW: Enhanced meat preference handling
            if "chicken" in meat_preferences and "chicken" in meal_name.lower():
                score += 15  # Significant boost for preferred meat
            elif "chicken" in meat_preferences and "chicken" in meal_recipe.lower():
                score += 10  # Moderate boost for preferred meat in recipe
            
            # If user prefers no meat, penalize meat-containing meals
            if prefers_no_meat:
                meat_keywords = ["chicken", "beef", "pork", "turkey", "lamb", "meat"]
                if any(meat in meal_name.lower() for meat in meat_keywords):
                    score -= 25  # Severe penalty for meat in name
                elif any(meat in meal_recipe.lower() for meat in meat_keywords):
                    score -= 15  # Moderate penalty for meat in recipe
            
            # Diet type compatibility with weighted scoring
            meal_diet_types = [d.lower() for d in meal.get("diet_types", [])]
            
            # Exact diet match
            if diet_type in meal_diet_types:
                score += 20  # Perfect match gets high score
            
            # Compatible diet types (domain knowledge)
            elif diet_type == "non-vegetarian" and "vegetarian" in meal_diet_types:
                score += 15   # Non-vegetarians can eat vegetarian meals
            elif diet_type == "flexitarian":
                if "vegetarian" in meal_diet_types or "vegan" in meal_diet_types:
                    score += 18  # Flexitarians prefer plant-based but occasionally eat meat
                else:
                    score += 8   # Meat options still acceptable
            elif diet_type == "pescatarian" and "vegetarian" in meal_diet_types:
                score += 15   # Pescatarians can eat vegetarian meals
            elif diet_type == "vegan" and "vegan" not in meal_diet_types:
                # Strict handling for vegans - only allow vegan meals
                continue
            else:
                # Diet incompatibility
                continue
                
            # AI Enhancement: Advanced goal-based nutritional scoring
            macros = meal.get("macros", {})
            calories = meal.get("calories", 0)
            
            if goal == "Weight Loss":
                # AI nutritional science: Higher protein, moderate fat, fiber-rich meals 
                # are more satiating for weight loss
                protein_ratio = macros.get("protein", 0) * 4 / calories if calories > 0 else 0
                
                # Progressive scoring based on protein content
                if protein_ratio >= 0.35:  # Excellent protein ratio
                    score += 15
                elif protein_ratio >= 0.3:  # Very good
                    score += 12
                elif protein_ratio >= 0.25:  # Good
                    score += 8
                elif protein_ratio >= 0.2:  # Acceptable
                    score += 4
                
                # Calorie scoring
                meal_type = meal.get("type", "").lower()
                ideal_calories = {
                    "breakfast": 300,
                    "lunch": 400,
                    "dinner": 450,
                    "snack": 150
                }
                
                ideal = ideal_calories.get(meal_type, 350)
                calorie_diff = abs(calories - ideal)
                
                # Progressive scoring based on how close to ideal calories
                if calorie_diff <= 25:
                    score += 10  # Very close
                elif calorie_diff <= 50:
                    score += 8   # Close
                elif calorie_diff <= 100:
                    score += 5   # Somewhat close
                elif calorie_diff > 200:
                    score -= 5   # Too far off
                    
            elif goal == "Muscle Gain":
                # AI nutritional science: Higher protein and adequate calories with 
                # balanced carbs are optimal for muscle growth
                protein_g = macros.get("protein", 0)
                
                # Progressive scoring based on absolute protein content
                if protein_g >= 35:  # Excellent
                    score += 15
                elif protein_g >= 30:  # Very good
                    score += 12
                elif protein_g >= 25:  # Good
                    score += 8
                elif protein_g >= 20:  # Acceptable
                    score += 4
                
                # Calorie scoring for muscle gain - needs surplus
                meal_type = meal.get("type", "").lower()
                ideal_calories = {
                    "breakfast": 450,
                    "lunch": 550,
                    "dinner": 600,
                    "snack": 250
                }
                
                ideal = ideal_calories.get(meal_type, 500)
                
                # For muscle gain, favor slightly higher calories
                if calories >= ideal:
                    calorie_bonus = min(10, (calories - ideal) / 20)
                    score += calorie_bonus
                else:
                    calorie_penalty = min(15, (ideal - calories) / 15)
                    score -= calorie_penalty
            
            # AI Enhancement: Consider user's cooking preferences
            # Value meals that match the user's time constraints
            cooking_time = user_data.get("cooking_time", "30 minutes")
            cooking_minutes = {"5 minutes": 5, "15 minutes": 15, "30 minutes": 30, 
                              "45 minutes": 45, "60+ minutes": 60}
            user_cooking_time = cooking_minutes.get(cooking_time, 30)
            
            # Estimate recipe complexity and time (using recipe length as a proxy)
            recipe = meal.get("recipe", "")
            complexity_score = len(recipe.split()) / 20  # Rough estimate
            
            # Quick recipes get higher scores for users with limited cooking time
            if user_cooking_time <= 15 and complexity_score <= 2:
                score += 10  # Quick recipe bonus
            elif user_cooking_time <= 30 and complexity_score <= 4:
                score += 8   # Moderately quick
            elif user_cooking_time < 60 and complexity_score > 6:
                score -= 5   # Too complex for time constraint
                
            # AI Enhancement: Nutritional score based on user profile
            # Use insights from the user analysis
            if insights.get("body_category") == "underweight":
                # Favor higher calorie, nutrient-dense foods
                score += min(10, calories / 80)
            elif insights.get("body_category") == "obese":
                # Favor lower calorie, higher protein, satiating foods
                if calories < 400 and protein_ratio > 0.3:
                    score += 12
            
            # AI Enhancement: Variety optimization
            # Ensure the diet includes a variety of nutrients
            meal_ingredients = set()
            if "recipe" in meal:
                words = meal["recipe"].lower().split()
                # Extract food keywords (simplified NLP)
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        meal_ingredients.add(word)
            
            # Score uniqueness of meal compared to what's already selected
            # (This would normally compare to already selected meals, but simplified here)
            uniqueness_score = min(8, len(meal_ingredients) / 3)
            score += uniqueness_score
            
            scored_meals.append((meal, score))
        
        # Sort meals by score (highest first)
        scored_meals.sort(key=lambda x: x[1], reverse=True)
        
        # Return the actual meal objects
        return [m[0] for m in scored_meals]
    
    def get_filtered_workouts(self, goal, difficulty, user_data):
        """
        Get workouts filtered by user preferences using AI-enhanced personalization.
        
        Args:
            goal (str): User's goal (e.g., "weight_loss")
            difficulty (str): Difficulty level (e.g., "beginner")
            user_data (dict): User profile data
            
        Returns:
            list: Filtered workout objects
        """
        goal_key = goal.lower()
        difficulty_key = difficulty.lower()
        
        # Get available workouts from data
        if goal_key in self.data and difficulty_key in self.data[goal_key]:
            available_workouts = self.data[goal_key][difficulty_key].get("workouts", [])
        else:
            available_workouts = []
        
        # If no workouts available, create defaults
        if not available_workouts:
            return [self._create_default_workout("cardio", ["Full Body"])]
        
        # Apply AI-enhanced personalization
        personalized_workouts = self.personalize_workout_recommendations(user_data, available_workouts)
        
        return personalized_workouts
    
    def get_filtered_meals(self, goal, difficulty, user_data):
        """
        Get meals filtered by user dietary preferences using AI-enhanced personalization.
        
        Args:
            goal (str): User's goal (e.g., "weight_loss")
            difficulty (str): Difficulty level (e.g., "beginner")
            user_data (dict): User profile data
            
        Returns:
            dict: Meals categorized by type
        """
        goal_key = goal.lower()
        difficulty_key = difficulty.lower()
        
        # Get available meals from data
        if goal_key in self.data and difficulty_key in self.data[goal_key]:
            available_meals = self.data[goal_key][difficulty_key].get("meals", [])
        else:
            available_meals = []
        
        # If no meals available, create defaults
        if not available_meals:
            diet_type = user_data.get("diet_type", "non-vegetarian").lower()
            return {
                "breakfast": [self._create_default_meal("breakfast", diet_type)],
                "lunch": [self._create_default_meal("lunch", diet_type)],
                "dinner": [self._create_default_meal("dinner", diet_type)],
                "snack": []
            }
        
        # Apply AI-enhanced personalization
        personalized_meals = self.personalize_meal_recommendations(user_data, available_meals)
        
        # Categorize meals by type
        categorized_meals = defaultdict(list)
        
        for meal in personalized_meals:
            meal_type = meal.get("type", "").lower()
            categorized_meals[meal_type].append(meal)
        
        # Ensure we have at least default meals for each main meal type
        diet_type = user_data.get("diet_type", "non-vegetarian").lower()
        for meal_type in ["breakfast", "lunch", "dinner"]:
            if not categorized_meals[meal_type]:
                categorized_meals[meal_type].append(
                    self._create_default_meal(meal_type, diet_type)
                )
        
        return dict(categorized_meals)
    
    def predict_optimal_workout_intensity(self, user_data):
        """
        AI-Enhanced: Uses a regression model to predict the optimal workout intensity for the user.
        
        Args:
            user_data (dict): User profile data
                
        Returns:
            str: Predicted optimal workout intensity
        """
        # Extract features
        age = user_data.get("age", 30)
        activity_level = user_data.get("activity_level", "Lightly Active")
        weight = user_data.get("weight", 70)
        height = user_data.get("height", 170)
        goal = user_data.get("goal", "Weight Loss")
        gender = user_data.get("gender", "Male")
        
        has_health_issues = False
        health_conditions = user_data.get("health_conditions", ["None"])
        if "None" not in health_conditions:
            has_health_issues = True
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Convert categorical features to numerical using one-hot encoding concept
        activity_score = {
            "Sedentary": 1,
            "Lightly Active": 2,
            "Moderately Active": 3,
            "Very Active": 4,
            "Extremely Active": 5
        }.get(activity_level, 2)
        
        goal_factor = {
            "Weight Loss": 0,
            "Muscle Gain": 1,
            "Improve Fitness": 0,
            "Body Toning": -0.5,
            "Endurance Building": 0.5
        }.get(goal, 0)
        
        gender_factor = 0.5 if gender == "Male" else 0
        
        # Get user's experience level (from difficulty preference or default to moderate)
        experience_scores = {
            "Beginner": 1,
            "Easy": 2,
            "Moderate": 3,
            "Challenging": 4,
            "Advanced": 5
        }
        experience = experience_scores.get(user_data.get("difficulty", "Moderate"), 3)
        
        # AI model: Normalized and weighted features
        features = {
            'age': (min(age, 70) / 70),  # Normalize age (cap at 70)
            'bmi': (min(bmi, 35) / 35),  # Normalize BMI (cap at 35)
            'activity_level': (activity_score / 5),  # Normalize to 0-1
            'health_issues': 1 if has_health_issues else 0,
            'goal_factor': (goal_factor + 1) / 2,  # Normalize to 0-1
            'gender_factor': gender_factor,
            'experience': (experience / 5)  # Normalize to 0-1
        }
        
        # Calculate weighted score (linear regression model simulation)
        # In production, this would be a trained model on real user data
        intensity_score = 0
        for feature, value in features.items():
            intensity_score += value * self.intensity_model_weights.get(feature, 0)
        
        # Normalize score to 0-5 range
        intensity_score = max(0, min(5, intensity_score + 2.5))  # Shift to center around 2.5
        
        # Map score to intensity levels
        if intensity_score <= 1:
            return "Beginner"
        elif intensity_score <= 2:
            return "Easy"
        elif intensity_score <= 3:
            return "Moderate"
        elif intensity_score <= 4:
            return "Challenging"
        else:
            return "Advanced"
    
    def generate_personalized_coaching(self, user_data, insights):
        """
        Generate personalized coaching messages using NLG techniques.
        
        Args:
            user_data (dict): User profile data
            insights (dict): AI-generated insights
            
        Returns:
            list: Personalized coaching tips
        """
        coaching_tips = []
        
        # Get key user characteristics
        goal = user_data.get("goal", "Weight Loss")
        body_category = insights.get("body_category", "normal_weight")
        recovery_profile = insights.get("recovery_profile", "moderate")
        fitness_level = insights.get("fitness_level", 5)
        
        # Template variables
        template_vars = {
            "bodytype": body_category.replace("_", " ").title(),
            "goal": goal,
            "recovery": recovery_profile,
            "fitness_level": f"{fitness_level}/10"
        }
        
        # Goal-specific variables
        if goal == "Weight Loss":
            template_vars.update({
                "macro_focus": "higher protein and fiber intake",
                "food_type": "lean protein and vegetables",
                "nutrient": "protein",
                "timeframe": "throughout the day",
                "technique": "maintaining cardiovascular intensity",
                "workout_type": "HIIT",
                "rest_period": "24-48 hours"
            })
        elif goal == "Muscle Gain":
            template_vars.update({
                "macro_focus": "higher protein and overall calorie intake",
                "food_type": "complete protein sources and complex carbs",
                "nutrient": "protein and carbohydrates",
                "timeframe": "around your workouts",
                "technique": "progressive overload",
                "workout_type": "resistance training",
                "rest_period": "48-72 hours per muscle group"
            })
        else:
            template_vars.update({
                "macro_focus": "balanced macronutrient intake",
                "food_type": "whole food variety",
                "nutrient": "overall calories",
                "timeframe": "based on your activity pattern",
                "technique": "proper form",
                "workout_type": "mixed workout",
                "rest_period": "36-48 hours"
            })
        
        # Health conditions specific variables
        if "health_conditions" in user_data and user_data["health_conditions"] != ["None"]:
            conditions = user_data["health_conditions"]
            if "Back Pain" in conditions:
                template_vars.update({
                    "limitation": "back",
                    "exercise": "squats and deadlifts",
                    "modification": "maintaining neutral spine and reducing load",
                    "muscle_group": "core"
                })
            elif "Knee Pain" in conditions:
                template_vars.update({
                    "limitation": "knee",
                    "exercise": "squats and lunges",
                    "modification": "reducing depth and focusing on alignment",
                    "muscle_group": "leg"
                })
            else:
                template_vars.update({
                    "limitation": "existing",
                    "exercise": "high-impact movements",
                    "modification": "focusing on controlled movement patterns",
                    "muscle_group": "supporting muscles"
                })
        else:
            template_vars.update({
                "limitation": "potential",
                "exercise": "complex movements",
                "modification": "mastering proper form before increasing intensity",
                "muscle_group": "full body"
            })
        
        # Generate nutrition tips using templates
        for i in range(2):
            template = random.choice(self.nlg_templates['nutrition'])
            # Replace variables in template
            tip = template
            for var, value in template_vars.items():
                tip = tip.replace("{" + var + "}", value)
            coaching_tips.append(tip)
        
        # Generate workout tips using templates
        for i in range(2):
            template = random.choice(self.nlg_templates['workout'])
            # Replace variables in template
            tip = template
            for var, value in template_vars.items():
                tip = tip.replace("{" + var + "}", value)
            coaching_tips.append(tip)
        
        # Add goal-specific tip
        if goal == "Weight Loss":
            coaching_tips.append(f"For optimal results with your {body_category.replace('_', ' ')} body type, aim for a moderate calorie deficit of 300-500 calories and include both strength training and cardio in a 2:1 ratio.")
        elif goal == "Muscle Gain":
            coaching_tips.append(f"Given your {fitness_level}/10 fitness level and {recovery_profile} recovery profile, prioritize compound movements and ensure you're in a caloric surplus of 250-500 calories on training days.")
        
        return coaching_tips
    
    def generate_adaptive_plan(self, goal, difficulty, user_data, progress_data=None):
        """
        AI-Enhanced: Generate a plan that adapts based on user's progress data.
        
        Args:
            goal (str): User's goal (e.g., "weight_loss")
            difficulty (str): Difficulty level (e.g., "beginner")
            user_data (dict): User profile data
            progress_data (dict): Previous workout and diet compliance data
                
        Returns:
            dict: Adaptive weekly fitness plan
        """
        # Start with a regular plan
        plan = self.generate_weekly_plan(goal, difficulty, user_data)
        
        # If we have progress data, adjust the plan
        if progress_data:
            # Check workout compliance
            workout_compliance = progress_data.get("workout_compliance", 100)
            
            # Use reinforcement learning to decide action
            adherence_state = "adherence_low" if workout_compliance < 70 else "adherence_high" if workout_compliance > 90 else "adherence_medium"
            
            # Get best action from Q-table
            q_values = self.rl_model['q_table'][adherence_state]
            best_action = max(q_values.items(), key=lambda x: x[1])[0]
            
            # Apply the best action
            if best_action == "decrease_intensity":
                # If user is struggling with workouts
                for day, day_plan in plan["weekly_plan"].items():
                    if not day_plan.get("rest_day", True):
                        # Reduce duration by 15%
                        day_plan["duration"] = int(day_plan["duration"] * 0.85)
                
                # Replace some harder workouts with easier ones
                easier_workouts = [w for w in plan["workouts"] 
                                if w.get("intensity", "") in ["Low", "Low-Moderate"]]
                
                if easier_workouts:
                    for day, day_plan in plan["weekly_plan"].items():
                        if not day_plan.get("rest_day", True) and random.random() < 0.5:
                            easier_workout = random.choice(easier_workouts)
                            day_plan["workout_type"] = easier_workout["type"]
                            day_plan["workout_name"] = easier_workout["name"]
                
                plan["ai_adaptive_changes"] = ["Reduced workout intensity and duration to improve adherence"]
                
            elif best_action == "increase_intensity":
                # If user is doing great, increase challenge
                for day, day_plan in plan["weekly_plan"].items():
                    if not day_plan.get("rest_day", True):
                        day_plan["duration"] = int(day_plan["duration"] * 1.1)
                
                # Add an extra workout day if not already at max
                workout_frequency = user_data.get("workout_frequency", 3)
                if workout_frequency < 6:
                    rest_days = [day for day, day_plan in plan["weekly_plan"].items() 
                                if day_plan.get("rest_day", True)]
                    
                    if rest_days:
                        new_workout_day = random.choice(rest_days)
                        workout = random.choice(plan["workouts"])
                        plan["weekly_plan"][new_workout_day] = {
                            "workout_type": workout["type"],
                            "workout_name": workout["name"],
                            "duration": workout["duration"],
                            "rest_day": False
                        }
                
                plan["ai_adaptive_changes"] = ["Increased workout challenge to optimize progress"]
            else:
                # Maintain current plan
                plan["ai_adaptive_changes"] = ["Plan maintained at current intensity level for optimal results"]
            
            # Check diet compliance and adjust meal plan
            diet_compliance = progress_data.get("diet_compliance", 100)
            
            if diet_compliance < 70:
                # Simplify meal plans
                for day in plan["meal_plan"]:
                    # Replace complex meals with simpler ones
                    for meal_type in ["breakfast", "lunch", "dinner"]:
                        if meal_type in plan["meal_plan"][day]:
                            # Sort meals by complexity (using recipe length as proxy)
                            simpler_meals = sorted(
                                plan["categorized_meals"].get(meal_type, []),
                                key=lambda x: len(x.get("recipe", ""))
                            )
                            
                            if simpler_meals:
                                plan["meal_plan"][day][meal_type] = simpler_meals[0]
                
                plan["ai_adaptive_changes"].append("Simplified meal plans to improve dietary adherence")
            
            # AI Enhancement: Adapt based on weight changes (if available)
            if "weight_change" in progress_data:
                weight_change = progress_data["weight_change"]
                goal = user_data.get("goal", "Weight Loss")
                
                # Adjust calorie targets based on progress
                if goal == "Weight Loss" and weight_change > -0.2:  # Not losing weight
                    # Reduce calorie target by 10%
                    plan["nutrition"]["target_calories"] = int(plan["nutrition"]["target_calories"] * 0.9)
                    
                    # Update macros accordingly
                    protein_pct = plan["nutrition"]["macros"]["protein"]["pct"] / 100
                    carb_pct = plan["nutrition"]["macros"]["carbs"]["pct"] / 100
                    fat_pct = plan["nutrition"]["macros"]["fat"]["pct"] / 100
                    
                    plan["nutrition"]["macros"]["protein"]["g"] = int((plan["nutrition"]["target_calories"] * protein_pct) / 4)
                    plan["nutrition"]["macros"]["carbs"]["g"] = int((plan["nutrition"]["target_calories"] * carb_pct) / 4)
                    plan["nutrition"]["macros"]["fat"]["g"] = int((plan["nutrition"]["target_calories"] * fat_pct) / 9)
                    
                    # Adjust meal plan to favor more protein, fewer carbs
                    for day in plan["meal_plan"]:
                        for meal_type in ["breakfast", "lunch", "dinner"]:
                            if meal_type in plan["meal_plan"][day]:
                                # Prioritize protein-rich meals
                                protein_meals = sorted(
                                    plan["categorized_meals"].get(meal_type, []),
                                    key=lambda x: x.get("macros", {}).get("protein", 0),
                                    reverse=True
                                )
                                
                                if protein_meals:
                                    plan["meal_plan"][day][meal_type] = protein_meals[0]
                    
                    plan["ai_adaptive_changes"].append("Adjusted calorie targets and increased protein to accelerate weight loss")
                
                elif goal == "Muscle Gain" and weight_change < 0.2:  # Not gaining weight
                    # Increase calorie target by 10%
                    plan["nutrition"]["target_calories"] = int(plan["nutrition"]["target_calories"] * 1.1)
                    
                    # Update macros accordingly
                    protein_pct = plan["nutrition"]["macros"]["protein"]["pct"] / 100
                    carb_pct = plan["nutrition"]["macros"]["carbs"]["pct"] / 100
                    fat_pct = plan["nutrition"]["macros"]["fat"]["pct"] / 100
                    
                    plan["nutrition"]["macros"]["protein"]["g"] = int((plan["nutrition"]["target_calories"] * protein_pct) / 4)
                    plan["nutrition"]["macros"]["carbs"]["g"] = int((plan["nutrition"]["target_calories"] * carb_pct) / 4)
                    plan["nutrition"]["macros"]["fat"]["g"] = int((plan["nutrition"]["target_calories"] * fat_pct) / 9)
                    
                    # Adjust meal plan to favor more calories
                    for day in plan["meal_plan"]:
                        for meal_type in ["breakfast", "lunch", "dinner"]:
                            if meal_type in plan["meal_plan"][day]:
                                # Prioritize calorie-rich meals
                                caloric_meals = sorted(
                                    plan["categorized_meals"].get(meal_type, []),
                                    key=lambda x: x.get("calories", 0),
                                    reverse=True
                                )
                                
                                if caloric_meals:
                                    plan["meal_plan"][day][meal_type] = caloric_meals[0]
                    
                    plan["ai_adaptive_changes"].append("Increased calorie targets to support muscle growth")
        
        # Add AI insights to the adaptive plan
        plan["ai_insights"] = self.analyze_user_profile(user_data)
        
        # Add personalized coaching tips
        plan["coaching_tips"] = self.generate_personalized_coaching(user_data, plan["ai_insights"])
        
        return plan
    
    def generate_weekly_plan(self, goal, difficulty, user_data):
        """
        Generate a personalized weekly fitness plan using AI-enhanced algorithms.
        
        Args:
            goal (str): User's goal (e.g., "weight_loss")
            difficulty (str): Difficulty level (e.g., "beginner")
            user_data (dict): User profile data
            
        Returns:
            dict: Weekly fitness plan
        """
        # Get filtered workouts and meals with AI-enhanced personalization
        filtered_workouts = self.get_filtered_workouts(goal, difficulty, user_data)
        categorized_meals = self.get_filtered_meals(goal, difficulty, user_data)
        
        # Determine workout frequency
        workout_frequency = user_data.get("workout_frequency", 3)
        
        # Create the weekly schedule
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        weekly_plan = {}
        
        # AI-Enhanced: Calculate optimized workout distribution based on preferred days
        workout_day_indices = []
        preferred_days = user_data.get("workout_schedule", ["No preference"])
        
        if "No preference" not in preferred_days:
            # Convert preferred days to lowercase
            preferred_days = [day.lower() for day in preferred_days if day != "No preference"]
            
            # Map days to indices
            day_to_index = {day: i for i, day in enumerate(days)}
            
            # NEW: Prioritize ALL preferred days if they're valid, regardless of frequency limit
            valid_preferred_indices = []
            for day in preferred_days:
                if day.lower() in day_to_index:
                    valid_preferred_indices.append(day_to_index[day.lower()])
            
            # NEW: If user specified more preferred days than workout frequency, prioritize first ones
            if len(valid_preferred_indices) > workout_frequency:
                workout_day_indices = valid_preferred_indices[:workout_frequency]
            else:
                # Use all valid preferred days
                workout_day_indices = valid_preferred_indices
                
                # If we need more workout days, add more with optimal spacing algorithm
                if len(workout_day_indices) < workout_frequency:
                    remaining_days = [i for i in range(len(days)) if i not in workout_day_indices]
                    remaining_needed = workout_frequency - len(workout_day_indices)
                    
                    # Optimally space remaining workouts (advanced algorithm)
                    if remaining_needed > 0 and remaining_days:
                        # Calculate optimal distances between workouts for recovery
                        user_insights = self.analyze_user_profile(user_data)
                        recovery_profile = user_insights.get("recovery_profile", "moderate")
                        
                        # Adjust spacing based on recovery needs
                        if recovery_profile == "extended":
                            # Need more recovery time
                            # Find largest gaps in current schedule
                            if workout_day_indices:
                                # Sort current workout days
                                workout_day_indices.sort()
                                
                                # Calculate gaps (including wraparound)
                                gaps = []
                                for i in range(len(workout_day_indices)):
                                    next_idx = (i + 1) % len(workout_day_indices)
                                    gap = (workout_day_indices[next_idx] - workout_day_indices[i]) % 7
                                    gaps.append((gap, i))
                                
                                # Sort gaps by size (largest first)
                                gaps.sort(reverse=True)
                                
                                # Place new workouts in largest gaps
                                for i in range(min(remaining_needed, len(gaps))):
                                    gap_size, gap_idx = gaps[i]
                                    if gap_size > 1:  # Only if gap is at least 2 days
                                        # Place workout in middle of gap
                                        current_idx = workout_day_indices[gap_idx]
                                        next_idx = workout_day_indices[(gap_idx + 1) % len(workout_day_indices)]
                                        if next_idx < current_idx:  # Wraparound
                                            next_idx += 7
                                        
                                        new_idx = (current_idx + (next_idx - current_idx) // 2) % 7
                                        workout_day_indices.append(new_idx)
                        
                        # If we still need more, just add what's left of remaining days
                        if len(workout_day_indices) < workout_frequency:
                            remaining_days = [i for i in range(len(days)) if i not in workout_day_indices]
                            for i in range(min(workout_frequency - len(workout_day_indices), len(remaining_days))):
                                workout_day_indices.append(remaining_days[i])
                    else:
                        # Standard or fast recovery - evenly space workouts
                        step = len(remaining_days) / remaining_needed
                        for i in range(remaining_needed):
                            idx = int(i * step)
                            if idx < len(remaining_days):
                                workout_day_indices.append(remaining_days[idx])
        else:
            # AI-Enhanced workout distribution algorithm
            # For optimal results, distribute based on recovery profile
            user_insights = self.analyze_user_profile(user_data)
            recovery_profile = user_insights.get("recovery_profile", "moderate")
            
            if recovery_profile == "extended":
                # Cluster workouts with more rest days between
                if workout_frequency <= 3:
                    # Place on Monday, Wednesday, Friday (indices 0, 2, 4)
                    workout_day_indices = [0, 2, 4][:workout_frequency]
                elif workout_frequency <= 4:
                    # Place on Monday, Wednesday, Friday, Saturday (indices 0, 2, 4, 5)
                    workout_day_indices = [0, 2, 4, 5][:workout_frequency]
                else:
                    # For 5+ days, use evenly spaced algorithm
                    step = len(days) / workout_frequency
                    for i in range(workout_frequency):
                        workout_day_indices.append(int(i * step))
            elif recovery_profile == "fast":
                # Can handle consecutive training days
                if workout_frequency <= 3:
                    # Place on Monday, Wednesday, Friday (indices 0, 2, 4)
                    workout_day_indices = [0, 2, 4][:workout_frequency]
                elif workout_frequency <= 5:
                    # Place on Monday, Tuesday, Thursday, Friday, Saturday (indices 0, 1, 3, 4, 5)
                    workout_day_indices = [0, 1, 3, 4, 5][:workout_frequency]
                else:
                    # For 6+ days, use consecutive days with one rest day
                    workout_day_indices = [0, 1, 2, 3, 4, 5][:workout_frequency]
            else:
                # Moderate recovery (standard)
                step = len(days) / workout_frequency
                for i in range(workout_frequency):
                    workout_day_indices.append(int(i * step))
        
        # Sort indices to maintain chronological order
        workout_day_indices.sort()
        
        # AI-Enhanced: Intelligent workout type sequencing based on exercise science
        # For optimal results, sequence workouts based on muscle groups and intensity
        workout_sequence = []
        
        # Get user's goal to better plan workout sequencing
        goal_key = goal.lower()
        
        if goal_key == "weight_loss":
            # For weight loss, alternate cardio and strength for optimal results
            cardio_workouts = [w for w in filtered_workouts if w.get("type", "").lower() in ["cardio", "hiit"]]
            strength_workouts = [w for w in filtered_workouts if w.get("type", "").lower() in ["strength", "weights"]]
            other_workouts = [w for w in filtered_workouts if w.get("type", "").lower() not in 
                             ["cardio", "hiit", "strength", "weights"]]
            
            # Build sequence alternating cardio and strength
            for i in range(workout_frequency):
                if i % 2 == 0 and cardio_workouts:
                    workout_sequence.append(cardio_workouts[i % len(cardio_workouts)])
                elif i % 2 == 1 and strength_workouts:
                    workout_sequence.append(strength_workouts[i % len(strength_workouts)])
                elif other_workouts:
                    workout_sequence.append(other_workouts[i % len(other_workouts)])
                elif filtered_workouts:
                    workout_sequence.append(filtered_workouts[i % len(filtered_workouts)])
        
        elif goal_key == "muscle_gain":
            # For muscle gain, use push/pull/legs split or full body on alternating days
            push_workouts = [w for w in filtered_workouts if any(area.lower() in ["chest", "shoulders", "triceps"] 
                                                              for area in w.get("focus_areas", []))]
            pull_workouts = [w for w in filtered_workouts if any(area.lower() in ["back", "biceps"] 
                                                              for area in w.get("focus_areas", []))]
            leg_workouts = [w for w in filtered_workouts if any(area.lower() in ["legs", "glutes"] 
                                                             for area in w.get("focus_areas", []))]
            full_body_workouts = [w for w in filtered_workouts if "full body" in 
                                 [area.lower() for area in w.get("focus_areas", [])]]
            
            # If we have enough specialized workouts, use push/pull/legs
            if push_workouts and pull_workouts and leg_workouts and workout_frequency >= 3:
                # Push/Pull/Legs split
                for i in range(workout_frequency):
                    if i % 3 == 0:  # Push day
                        workout_sequence.append(push_workouts[i // 3 % len(push_workouts)])
                    elif i % 3 == 1:  # Pull day
                        workout_sequence.append(pull_workouts[i // 3 % len(pull_workouts)])
                    else:  # Leg day
                        workout_sequence.append(leg_workouts[i // 3 % len(leg_workouts)])
            elif full_body_workouts:
                # Full body workouts with optimal spacing
                for i in range(workout_frequency):
                    workout_sequence.append(full_body_workouts[i % len(full_body_workouts)])
            else:
                # Just use whatever workouts we have
                for i in range(workout_frequency):
                    workout_sequence.append(filtered_workouts[i % len(filtered_workouts)])
        
        else:
            # For general fitness/toning/endurance, use a balanced approach
            # Group workouts by type
            workout_types = {}
            for w in filtered_workouts:
                w_type = w.get("type", "").lower()
                if w_type not in workout_types:
                    workout_types[w_type] = []
                workout_types[w_type].append(w)
            
            # Use a balanced weekly plan
            for i in range(workout_frequency):
                # Get a list of all workout types
                types = list(workout_types.keys())
                
                if not types:
                    # If no types available, use original workouts
                    if filtered_workouts:
                        workout_sequence.append(filtered_workouts[i % len(filtered_workouts)])
                    continue
                
                # Choose type based on position in week
                chosen_type = types[i % len(types)]
                workouts_of_type = workout_types[chosen_type]
                
                if workouts_of_type:
                    # Choose specific workout from this type
                    workout_sequence.append(workouts_of_type[i // len(types) % len(workouts_of_type)])
        
        # If we don't have enough in the sequence, fill with the original filtered workouts
        while len(workout_sequence) < workout_frequency and filtered_workouts:
            workout_sequence.append(filtered_workouts[len(workout_sequence) % len(filtered_workouts)])
        
        # Assign workouts and rest days
        for i, day in enumerate(days):
            if i in workout_day_indices and workout_sequence:
                # Get a workout for this day from the optimized sequence
                workout_index = workout_day_indices.index(i)
                if workout_index < len(workout_sequence):
                    workout = workout_sequence[workout_index]
                else:
                    # Fallback to cycling through available workouts
                    workout = filtered_workouts[workout_index % len(filtered_workouts)]
                
                weekly_plan[day] = {
                    "workout_type": workout["type"],
                    "workout_name": workout["name"],
                    "duration": workout["duration"],
                    "rest_day": False
                }
            else:
                weekly_plan[day] = {
                    "workout_type": "Rest",
                    "workout_name": "Rest Day",
                    "duration": 0,
                    "rest_day": True
                }
        
        # AI-Enhanced: Intelligent meal planning with dietary science
        # Generate daily meal plans with optimal nutrient timing
        meal_plan = {}
        for day in days:
            meal_plan[day] = {}
            
            # Determine if this is a workout day
            is_workout_day = not weekly_plan[day].get("rest_day", True)
            workout_type = weekly_plan[day].get("workout_type", "Rest").lower()
            
            # AI optimization for workout/rest day nutrition
            if is_workout_day:
                # Higher carb needs on workout days, especially for strength/HIIT
                if workout_type in ["strength", "weights", "hiit"]:
                    # Sort breakfast options by carb content for pre-workout energy
                    if "breakfast" in categorized_meals and categorized_meals["breakfast"]:
                        breakfast_options = sorted(
                            categorized_meals["breakfast"],
                            key=lambda x: x.get("macros", {}).get("carbs", 0),
                            reverse=True
                        )
                        meal_plan[day]["breakfast"] = breakfast_options[0]
                    
                    # Sort lunch/dinner by protein content for recovery
                    if "lunch" in categorized_meals and categorized_meals["lunch"]:
                        lunch_options = sorted(
                            categorized_meals["lunch"],
                            key=lambda x: x.get("macros", {}).get("protein", 0),
                            reverse=True
                        )
                        meal_plan[day]["lunch"] = lunch_options[0]
                    
                    if "dinner" in categorized_meals and categorized_meals["dinner"]:
                        dinner_options = sorted(
                            categorized_meals["dinner"],
                            key=lambda x: x.get("macros", {}).get("protein", 0),
                            reverse=True
                        )
                        meal_plan[day]["dinner"] = dinner_options[0]
                else:
                    # For cardio days, balance all macros
                    for meal_type in ["breakfast", "lunch", "dinner"]:
                        if meal_type in categorized_meals and categorized_meals[meal_type]:
                            # Choose a balanced meal (close to 40-30-30 carb-protein-fat)
                            meal_plan[day][meal_type] = random.choice(categorized_meals[meal_type][:3])
            else:
                # Rest days - lower carbs, maintain protein
                for meal_type in ["breakfast", "lunch", "dinner"]:
                    if meal_type in categorized_meals and categorized_meals[meal_type]:
                        # On rest days, prioritize higher protein, lower carb meals
                        options = categorized_meals[meal_type]
                        
                        # Simple algorithm to find balanced meals
                        protein_scores = []
                        for meal in options:
                            macros = meal.get("macros", {})
                            # Score based on protein ratio and moderate carbs
                            calories = meal.get("calories", 0)
                            if calories > 0:
                                protein_ratio = macros.get("protein", 0) * 4 / calories
                                carb_ratio = macros.get("carbs", 0) * 4 / calories
                                # Rest day ideal: higher protein, lower carbs
                                score = protein_ratio - (carb_ratio * 0.5)
                                protein_scores.append((meal, score))
                        
                        if protein_scores:
                            # Sort by score and take top option
                            protein_scores.sort(key=lambda x: x[1], reverse=True)
                            meal_plan[day][meal_type] = protein_scores[0][0]
                        else:
                            meal_plan[day][meal_type] = random.choice(options)
            
            # Add snacks if requested
            if user_data.get("meal_count", 3) > 3 and "snack" in categorized_meals and categorized_meals["snack"]:
                num_snacks = user_data.get("meal_count", 3) - 3
                snacks = []
                
                # AI-Enhanced: Strategic snack timing
                if is_workout_day:
                    # Prioritize protein-rich snacks for workout days
                    protein_snacks = sorted(
                        categorized_meals["snack"],
                        key=lambda x: x.get("macros", {}).get("protein", 0),
                        reverse=True
                    )
                    
                    for _ in range(min(num_snacks, len(protein_snacks))):
                        snack = protein_snacks[_ % len(protein_snacks)]
                        if snack not in snacks:
                            snacks.append(snack)
                else:
                    # For rest days, focus on balanced, lower calorie snacks
                    for _ in range(min(num_snacks, len(categorized_meals["snack"]))):
                        # Pick a random snack avoiding duplicates if possible
                        available_snacks = [s for s in categorized_meals["snack"] if s not in snacks]
                        if not available_snacks:
                            available_snacks = categorized_meals["snack"]
                        snacks.append(random.choice(available_snacks))
                
                meal_plan[day]["snacks"] = snacks
        
        # AI-Enhanced: Calculate nutrition targets with precision nutrition science
        nutrition = self._calculate_nutrition_targets(user_data)
        
        # Define difficulty_key (THIS FIXES THE ERROR)
        difficulty_key = difficulty.lower()
        
        # Get equipment recommendations if available
        equipment_recommendations = []
        if goal_key in self.data and difficulty_key in self.data[goal_key]:
            equipment_recommendations = self.data[goal_key][difficulty_key].get("equipment_recommendations", [])
        
        # AI-Enhanced: Add coaching tips based on user profile
        user_insights = self.analyze_user_profile(user_data)
        coaching_tips = self.generate_personalized_coaching(user_data, user_insights)
        
        # Assemble the complete plan with AI enhancements
        complete_plan = {
            "weekly_plan": weekly_plan,
            "workouts": filtered_workouts,
            "categorized_meals": categorized_meals,
            "meal_plan": meal_plan,
            "nutrition": nutrition,
            "equipment_recommendations": equipment_recommendations,
            "ai_insights": user_insights,
            "coaching_tips": coaching_tips,
            "ai_adaptive_changes": []
        }
        
        return complete_plan
    
    def _calculate_nutrition_targets(self, user_data):
        """
        AI-Enhanced: Calculate precision nutrition targets based on user data.
        
        Args:
            user_data (dict): User profile data
            
        Returns:
            dict: Nutrition targets
        """
        # Calculate BMR using Mifflin-St Jeor Equation (more accurate than original)
        weight = user_data.get("weight", 70)  # kg
        height = user_data.get("height", 170)  # cm
        age = user_data.get("age", 30)
        gender = user_data.get("gender", "Male")
        
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:  # Female or Non-binary (using female formula as approximation for non-binary)
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        # Apply activity multiplier with refined categories
        activity_levels = {
            "Sedentary": 1.2,      # Office job, little/no exercise
            "Lightly Active": 1.375,  # Light exercise 1-3 days/week
            "Moderately Active": 1.55,  # Moderate exercise 3-5 days/week
            "Very Active": 1.725,   # Hard exercise 6-7 days/week
            "Extremely Active": 1.9  # Hard daily exercise & physical job or 2x/day training
        }
        
        activity_level = user_data.get("activity_level", "Lightly Active")
        activity_multiplier = activity_levels.get(activity_level, 1.375)
        
        # Refine TDEE calculation with more factors
        # Age-based adjustment (metabolism slows with age)
        age_factor = 1.0
        if age > 40:
            age_factor = 1.0 - ((age - 40) * 0.005)  # 0.5% reduction per year over 40
        
        # Health conditions adjustment
        health_conditions = user_data.get("health_conditions", ["None"])
        health_factor = 1.0
        if "None" not in health_conditions:
            # Reduce slightly if health issues present
            health_factor = 0.95
        
        # Calculate TDEE with additional factors
        tdee = bmr * activity_multiplier * age_factor * health_factor
        
        # Adjust calories based on goal and calorie preference
        goal = user_data.get("goal", "Improve Fitness")
        calorie_preference = user_data.get("calorie_preference", "Moderate deficit")
        
        # AI-Enhanced calorie targeting based on goal efficiency
        if calorie_preference == "Strict deficit":
            if goal == "Weight Loss":
                target_calories = tdee - 500  # Standard deficit for weight loss
            else:
                target_calories = tdee - 350  # Smaller deficit for other goals
        elif calorie_preference == "Moderate deficit":
            if goal == "Weight Loss":
                target_calories = tdee - 350  # Moderate deficit for weight loss
            else:
                target_calories = tdee - 250  # Smaller deficit for other goals
        elif calorie_preference == "Maintenance":
            target_calories = tdee
        elif calorie_preference == "Moderate surplus":
            if goal == "Muscle Gain":
                target_calories = tdee + 350  # Higher surplus for muscle gain
            else:
                target_calories = tdee + 250  # Standard surplus for other goals
        else:  # Building surplus
            if goal == "Muscle Gain":
                target_calories = tdee + 500  # Higher surplus for muscle gain
            else:
                target_calories = tdee + 400  # Standard surplus for other goals
        
        # Ensure minimum healthy calorie intake
        if gender == "Male":
            min_calories = 1500
        else:
            min_calories = 1200
        
        target_calories = max(min_calories, target_calories)
        
        # AI-Enhanced macro distribution based on goal science
        if goal == "Weight Loss":
            # Higher protein for muscle preservation during deficit
            protein_pct = 0.35  # 35% protein
            fat_pct = 0.30     # 30% fat 
            carb_pct = 0.35    # 35% carbs
        elif goal == "Muscle Gain":
            # Higher carbs for energy and protein for muscle synthesis
            protein_pct = 0.30  # 30% protein
            fat_pct = 0.25     # 25% fat
            carb_pct = 0.45    # 45% carbs
        elif goal == "Body Toning":
            # Balanced approach with slight protein emphasis
            protein_pct = 0.30  # 30% protein
            fat_pct = 0.30     # 30% fat
            carb_pct = 0.40    # 40% carbs
        elif goal == "Endurance Building":
            # Higher carbs for endurance activities
            protein_pct = 0.25  # 25% protein
            fat_pct = 0.25     # 25% fat
            carb_pct = 0.50    # 50% carbs
        else:  # Improve Fitness/General
            # Balanced approach
            protein_pct = 0.30  # 30% protein
            fat_pct = 0.30     # 30% fat
            carb_pct = 0.40    # 40% carbs
        
        # Adjust macros based on dietary preferences
        diet_type = user_data.get("diet_type", "Non-Vegetarian").lower()
        if diet_type in ["vegetarian", "vegan"]:
            # Plant-based diets may need higher protein percentage to get enough complete proteins
            protein_adj = 0.02  # Add 2% to protein
            carb_pct += 0.03    # Add 3% to carbs
            fat_pct -= 0.05     # Reduce fat by 5%
            protein_pct += protein_adj
        elif diet_type == "keto":
            # Ketogenic diet has very different macro ratios
            protein_pct = 0.25  # 25% protein
            fat_pct = 0.70     # 70% fat
            carb_pct = 0.05    # 5% carbs
        
        # Calculate macro amounts with precision
        protein_g = (target_calories * protein_pct) / 4  # 4 calories per gram of protein
        carb_g = (target_calories * carb_pct) / 4       # 4 calories per gram of carbs
        fat_g = (target_calories * fat_pct) / 9         # 9 calories per gram of fat
        
        # AI-Enhanced: Calculate protein based on lean body mass for more precision
        # Estimate body fat percentage using BMI as a rough proxy
        height_m = user_data.get("height", 170) / 100
        bmi = weight / (height_m ** 2)
        
        estimated_bf = 0
        if gender == "Male":
            estimated_bf = 1.20 * bmi + 0.23 * age - 16.2
        else:  # Female or Non-binary
            estimated_bf = 1.20 * bmi + 0.23 * age - 5.4
        
        # Clamp to realistic range
        estimated_bf = max(5, min(50, estimated_bf)) / 100
        
        # Calculate lean body mass
        lbm = weight * (1 - estimated_bf)
        
        # Calculate protein needs based on LBM and goal
        protein_lbm_g = 0
        if goal == "Weight Loss":
            protein_lbm_g = lbm * 2.0  # 2.0g per kg of LBM for weight loss
        elif goal == "Muscle Gain":
            protein_lbm_g = lbm * 2.2  # 2.2g per kg of LBM for muscle gain
        else:
            protein_lbm_g = lbm * 1.8  # 1.8g per kg of LBM for maintenance
        
        # Use the higher of the two protein calculations
        protein_g = max(protein_g, protein_lbm_g)
        
        # Recalculate other macros to maintain calorie target
        protein_calories = protein_g * 4
        remaining_calories = target_calories - protein_calories
        
        # Adjust the fat and carb ratio for remaining calories
        if diet_type == "keto":
            fat_ratio = 0.85  # 85% of remaining calories from fat
            carb_ratio = 0.15  # 15% of remaining calories from carbs
        else:
            fat_ratio = fat_pct / (fat_pct + carb_pct)
            carb_ratio = carb_pct / (fat_pct + carb_pct)
        
        fat_g = (remaining_calories * fat_ratio) / 9
        carb_g = (remaining_calories * carb_ratio) / 4
        
        # Calculate actual percentages for display
        total_cals = protein_g * 4 + carb_g * 4 + fat_g * 9
        protein_pct_actual = (protein_g * 4 / total_cals) * 100
        carb_pct_actual = (carb_g * 4 / total_cals) * 100
        fat_pct_actual = (fat_g * 9 / total_cals) * 100
        
        # Return the comprehensive nutrition plan
        return {
            "bmr": int(bmr),
            "tdee": int(tdee),
            "target_calories": int(target_calories),
            "body_fat_estimate": round(estimated_bf * 100, 1),
            "lean_body_mass": round(lbm, 1),
            "macros": {
                "protein": {
                    "g": int(protein_g),
                    "pct": int(protein_pct_actual)
                },
                "carbs": {
                    "g": int(carb_g),
                    "pct": int(carb_pct_actual)
                },
                "fat": {
                    "g": int(fat_g),
                    "pct": int(fat_pct_actual)
                }
            },
            "nutrient_timing": {
                "workout_days": {
                    "pre_workout": "Higher carbs for energy",
                    "post_workout": "Protein + carbs for recovery",
                    "other_meals": "Balanced macros"
                },
                "rest_days": {
                    "all_meals": "Higher protein, lower carbs"
                }
            }
        }

# Function to get a plan (to be used by main.py)
def get_enhanced_plan(goal="weight_loss", difficulty="moderate", diet_type="non-vegetarian", user_data=None):
    """
    Get a personalized fitness and meal plan with AI-enhanced decision making.
    
    Args:
        goal (str): User's primary fitness goal
        difficulty (str): Desired difficulty level
        diet_type (str): User's dietary preference
        user_data (dict): Additional user profile data
        
    Returns:
        dict: A personalized fitness plan
    """
    if user_data is None:
        user_data = {}
        
    planner = EnhancedFitnessPlanner()
    
    # Format the goal for the planner
    goal_mapping = {
        "Weight Loss": "weight_loss",
        "Muscle Gain": "muscle_gain",
        "Improve Fitness": "fitness",
        "Body Toning": "toning",
        "Endurance Building": "endurance"
    }
    goal_key = goal_mapping.get(goal, goal)
    
    # Apply AI-enhanced difficulty adjustment if user hasn't explicitly chosen
    if user_data.get("auto_adjust_difficulty", False) or "difficulty" not in user_data:
        optimal_difficulty = planner.predict_optimal_workout_intensity(user_data)
        difficulty = optimal_difficulty.lower()
    
    # Check if we have progress data for adaptive planning
    if "progress_data" in user_data:
        plan = planner.generate_adaptive_plan(goal_key, difficulty, user_data, user_data["progress_data"])
    else:
        # Generate the plan
        plan = planner.generate_weekly_plan(goal_key, difficulty, user_data)
    
    # Add AI insights to the plan
    plan["ai_insights"] = planner.analyze_user_profile(user_data)
    
    # Add personalized coaching tips if not already present
    if "coaching_tips" not in plan:
        plan["coaching_tips"] = planner.generate_personalized_coaching(user_data, plan["ai_insights"])
    
    return plan
