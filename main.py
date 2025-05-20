import streamlit as st
from planner import get_enhanced_plan
from display import display_enhanced_plan
import uuid
from datetime import datetime, timedelta
import pandas as pd
from feedback import FeedbackAnalyzer

st.set_page_config(page_title="FitLife Pro AI+", page_icon="🏋️", layout="wide")

# Initialize session state for multi-step form
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Generate unique user ID
if 'feedback_analyzer' not in st.session_state:
    st.session_state.feedback_analyzer = FeedbackAnalyzer()
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = {
        'weight_history': [],
        'workout_completions': {},
        'measurement_history': {
            'chest': [], 'waist': [], 'hips': [], 'arms': [], 'thighs': []
        },
        'performance_metrics': {
            'strength': [], 'endurance': [], 'flexibility': []
        },
        'nutrition_adherence': []
    }

# Custom CSS to enhance UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .step-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.8rem;
    }
    .step-indicator {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .step-dot {
        height: 25px;
        width: 25px;
        background-color: #bbb;
        border-radius: 50%;
        display: inline-block;
        margin: 0 5px;
        text-align: center;
        line-height: 25px;
        color: white;
    }
    .step-dot.active {
        background-color: #1E88E5;
    }
    .step-dot.completed {
        background-color: #4CAF50;
    }
    .note-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .ai-insight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
        margin-bottom: 1rem;
    }
    .ai-coaching-tip {
        background-color: #e5f6ff;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #0288d1;
    }
    .progress-card {
        background-color: #f5f9ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #42a5f5;
    }
    .adherence-high {
        border-left: 3px solid #4caf50;
    }
    .adherence-medium {
        border-left: 3px solid #ff9800;
    }
    .adherence-low {
        border-left: 3px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown('<div class="main-header">🎯🏋️ FitLife Pro AI+: Advanced Personalized Fitness Journey</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Create a deeply personalized fitness plan with cutting-edge AI technology</div>', unsafe_allow_html=True)

# Function to move to next step
def next_step():
    st.session_state.step += 1

# Function to move to previous step
def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

# Function to save form data
def save_form_data(form_data):
    st.session_state.user_data.update(form_data)

# Progress indicators
total_steps = 6  # Updated to include tracking
step_indicators = []
for i in range(1, total_steps + 1):
    if i < st.session_state.step:
        step_indicators.append(f'<span class="step-dot completed">{i}</span>')
    elif i == st.session_state.step:
        step_indicators.append(f'<span class="step-dot active">{i}</span>')
    else:
        step_indicators.append(f'<span class="step-dot">{i}</span>')

st.markdown(f'<div class="step-indicator">{"".join(step_indicators)}</div>', unsafe_allow_html=True)

# Step 1: Basic Information
if st.session_state.step == 1:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("Step 1: Tell us about yourself")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=16, max_value=90, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
    
    with col2:
        activity_level = st.select_slider(
            "Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
            value="Lightly Active",
            help="Sedentary: Office job, little exercise; Lightly Active: Light exercise 1-3 days/week; Moderately Active: Moderate exercise 3-5 days/week; Very Active: Hard exercise 6-7 days/week; Extremely Active: Hard daily exercise & physical job or 2x/day training."
        )
        health_conditions = st.multiselect(
            "Any health conditions?",
            ["None", "Diabetes", "Hypertension", "Heart Disease", "Arthritis", "Back Pain", "Knee Pain", "Asthma", "Pregnancy", "Other"],
            default=["None"],
            help="Select any health conditions that might affect your fitness regimen."
        )
        
        # Only show if 'Other' is selected
        other_condition = ""
        if "Other" in health_conditions:
            other_condition = st.text_input("Please specify your health condition")
            
        # Allow users to input current medications
        medications = st.text_input("Current medications (optional)", help="This helps us customize your plan to avoid potential issues")
    
    # Inform about health considerations
    if health_conditions and "None" not in health_conditions:
        st.markdown('<div class="note-box">', unsafe_allow_html=True)
        st.markdown("⚠️ **Important Note:** We'll customize your plan with your health conditions in mind, but please consult your healthcare provider before starting any new fitness program.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add AI insights about health profile
    if age > 0 and weight > 0 and height > 0:
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        bmi_category = "underweight" if bmi < 18.5 else "normal weight" if bmi < 25 else "overweight" if bmi < 30 else "obese"
        
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Health Insights")
        
        # Display BMI and insights
        st.markdown(f"**BMI:** {bmi:.1f} ({bmi_category.title()})")
        
        # Health-based recommendations
        if bmi_category == "underweight":
            st.markdown("Our AI recommends focusing on muscle building and nutritionally dense foods. Your plan will include higher calorie targets to help you achieve a healthy weight.")
        elif bmi_category == "overweight" or bmi_category == "obese":
            st.markdown("Our AI will design a plan focusing on sustainable fat loss while preserving muscle. We'll include cardio and strength training in an optimal balance for your goals.")
        
        # Age-based insights
        if age < 30:
            st.markdown("Your younger age is an advantage for fitness goals. Our AI will optimize your plan for faster recovery and higher intensity workouts.")
        elif age < 50:
            st.markdown("At your age, our AI will balance workout intensity with recovery needs, focusing on both performance and injury prevention.")
        else:
            st.markdown("Our AI will prioritize joint health and appropriate exercise selection for your age group, with adequate recovery periods.")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    form_data = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "health_conditions": health_conditions,
        "other_condition": other_condition if "Other" in health_conditions else "",
        "medications": medications
    }
    
    if st.button("Next: Fitness Goals"):
        save_form_data(form_data)
        next_step()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 2: Fitness Goals
elif st.session_state.step == 2:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("Step 2: Set your fitness goals")
    
    # More detailed explanations for goals
    goal_descriptions = {
        "Weight Loss": "Focus on reducing body fat while preserving muscle mass",
        "Muscle Gain": "Focus on building muscle size and strength",
        "Improve Fitness": "Enhance overall cardiovascular fitness and energy levels",
        "Body Toning": "Define muscles without significant size increase",
        "Endurance Building": "Increase stamina and exercise capacity"
    }
    
    # Display goal options with descriptions
    goal = st.selectbox(
        "Primary Goal", 
        list(goal_descriptions.keys()),
        help="Select the primary focus of your fitness journey"
    )
    
    # Show description of selected goal
    st.markdown(f"<div class='note-box'><strong>{goal}:</strong> {goal_descriptions[goal]}</div>", unsafe_allow_html=True)
    
    # Goal-specific options
    if goal == "Weight Loss":
        current_weight = st.session_state.user_data["weight"]
        
        # Calculate healthy weight loss range
        min_healthy_weight = max(current_weight * 0.9, 50)  # Limit to 10% loss or minimum healthy weight
        
        target_weight = st.number_input(
            "Target Weight (kg)", 
            min_value=min_healthy_weight, 
            max_value=current_weight - 0.5, 
            value=max(current_weight - 5, min_healthy_weight),
            step=0.1,
            help="A healthy weight loss goal is 0.5-1kg per week"
        )
        
        # Calculate weight loss percentage
        weight_loss_percentage = ((current_weight - target_weight) / current_weight) * 100
        
        # Provide feedback on goal
        if weight_loss_percentage > 20:
            st.warning(f"⚠️ Your target represents a {weight_loss_percentage:.1f}% weight loss. Consider a more moderate goal for sustainable results.")
        elif weight_loss_percentage > 10:
            st.info(f"Your target represents a {weight_loss_percentage:.1f}% weight loss. This is ambitious but achievable with consistency.")
        else:
            st.success(f"Your target represents a {weight_loss_percentage:.1f}% weight loss. This is a healthy, sustainable goal.")
            
    elif goal == "Muscle Gain":
        current_weight = st.session_state.user_data["weight"]
        
        target_weight = st.number_input(
            "Target Weight (kg)", 
            min_value=current_weight + 0.5, 
            max_value=current_weight * 1.2,  # Limit to 20% gain
            value=min(current_weight + 5, current_weight * 1.1),
            step=0.1,
            help="A realistic muscle gain is about 0.25-0.5kg per week"
        )
        
        # Calculate weight gain percentage
        weight_gain_percentage = ((target_weight - current_weight) / current_weight) * 100
        
        # Provide feedback on goal
        if weight_gain_percentage > 15:
            st.warning(f"⚠️ Your target represents a {weight_gain_percentage:.1f}% weight gain. Consider a more moderate goal for lean muscle gain.")
        elif weight_gain_percentage > 10:
            st.info(f"Your target represents a {weight_gain_percentage:.1f}% weight gain. This is ambitious but achievable with proper nutrition and training.")
        else:
            st.success(f"Your target represents a {weight_gain_percentage:.1f}% weight gain. This is a realistic goal for lean muscle gain.")
    else:
        # For other goals, just use current weight
        target_weight = st.session_state.user_data["weight"]
    
    timeline = st.slider(
        "Timeline (months)", 
        min_value=1, 
        max_value=24, 
        value=3,
        help="Realistic timeline for achieving your goal"
    )
    
    # Calculate and display weekly change
    if goal == "Weight Loss":
        weekly_change = (st.session_state.user_data["weight"] - target_weight) / (timeline * 4.33)  # 4.33 weeks per month on average
        st.markdown(f"Target weekly weight loss: **{weekly_change:.2f} kg/week**")
        
        # Advice on rate
        if weekly_change > 1:
            st.warning("⚠️ This rate of weight loss exceeds general recommendations. Consider extending your timeline for more sustainable results.")
        elif weekly_change > 0.5:
            st.info("This is an ambitious but achievable rate of weight loss with proper nutrition and exercise.")
        else:
            st.success("This is a sustainable rate of weight loss, which helps preserve muscle mass and increase success rate.")
            
    elif goal == "Muscle Gain":
        weekly_change = (target_weight - st.session_state.user_data["weight"]) / (timeline * 4.33)
        st.markdown(f"Target weekly weight gain: **{weekly_change:.2f} kg/week**")
        
        # Advice on rate
        if weekly_change > 0.5:
            st.warning("⚠️ This rate of muscle gain may lead to more fat gain than desired. Consider extending your timeline.")
        elif weekly_change > 0.25:
            st.info("This is an ambitious rate of muscle gain, requiring consistent training and nutrition.")
        else:
            st.success("This is a sustainable rate of lean muscle gain for most individuals.")
    
    # Option to let AI recommend optimal difficulty
    auto_adjust = st.checkbox("Let AI recommend optimal workout intensity", value=True,
                             help="Our AI will analyze your profile to suggest the most effective workout intensity")
    
    if auto_adjust:
        # Show AI is analyzing
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("🧠 **AI is analyzing your profile to recommend the optimal workout intensity.**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        difficulty = "Moderate"  # Default value, will be overridden by AI
    else:
        difficulty = st.select_slider(
            "Preferred Intensity",
            options=["Beginner", "Easy", "Moderate", "Challenging", "Advanced"],
            value="Moderate",
            help="Choose an intensity level that matches your fitness experience and comfort zone"
        )
    
        # Show description of selected difficulty
        difficulty_descriptions = {
            "Beginner": "New to exercise, focusing on form and building habits",
            "Easy": "Some experience with exercise, building strength and endurance",
            "Moderate": "Regular exerciser looking for balanced challenge",
            "Challenging": "Experienced exerciser ready for intense workouts",
            "Advanced": "Very experienced, looking for maximum intensity"
        }
        
        st.markdown(f"<div class='note-box'><strong>{difficulty}:</strong> {difficulty_descriptions[difficulty]}</div>", unsafe_allow_html=True)
    
    workout_frequency = st.slider(
        "Workout days per week", 
        min_value=1, 
        max_value=7, 
        value=4,
        help="How many days per week can you commit to structured exercise?"
    )
    
    workout_duration = st.slider(
        "Minutes per workout session", 
        min_value=15, 
        max_value=120, 
        value=45, 
        step=5,
        help="How long can you spend on each workout?"
    )
    
    # Get collaborative recommendations from feedback system
    if st.session_state.user_data:
        # Check if we have enough user data for recommendations
        similar_users = st.session_state.feedback_analyzer.get_similar_users(st.session_state.user_data)
        
        if similar_users:
            workout_recommendations = st.session_state.feedback_analyzer.get_collaborative_recommendations(
                st.session_state.user_data, recommendation_type="workout", limit=3)
            
            if workout_recommendations:
                st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
                st.markdown("### 🧠 AI Workout Recommendations")
                st.markdown("Based on successful users with similar profiles, our AI recommends these workouts:")
                
                for rec in workout_recommendations:
                    st.markdown(f"- **{rec['name']}** (Rating: {rec['score']:.1f}/5)")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    form_data = {
        "goal": goal,
        "target_weight": target_weight,
        "timeline": timeline,
        "difficulty": difficulty,
        "auto_adjust_difficulty": auto_adjust,
        "workout_frequency": workout_frequency,
        "workout_duration": workout_duration
    }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            prev_step()
    with col2:
        if st.button("Next: Diet Preferences"):
            save_form_data(form_data)
            next_step()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Dietary Preferences
elif st.session_state.step == 3:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("Step 3: Your dietary preferences")
    
    # Diet type options with descriptions
    diet_types = {
        "Non-Vegetarian": "Includes all food groups including meat and animal products",
        "Vegetarian": "Excludes meat but includes dairy and eggs",
        "Vegan": "Excludes all animal products",
        "Pescatarian": "Vegetarian diet plus seafood",
        "Flexitarian": "Primarily plant-based with occasional meat",
        "Keto": "High fat, low carb diet to achieve ketosis",
        "Paleo": "Based on foods presumed to be available to paleolithic humans"
    }
    
    diet_type = st.selectbox(
        "Diet Type", 
        list(diet_types.keys()),
        help="Select the dietary pattern that best matches your preferences"
    )
    
    # Show description of selected diet type
    st.markdown(f"<div class='note-box'><strong>{diet_type}:</strong> {diet_types[diet_type]}</div>", unsafe_allow_html=True)
    
    # AI Diet Analysis
    st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
    st.markdown("### 🧠 AI Diet Analysis")
    
    # Analyze diet compatibility with goal
    goal = st.session_state.user_data.get("goal", "Weight Loss")
    
    # Get diet compatibility score
    compatibility = {
        "Weight Loss": {
            "Non-Vegetarian": 90,
            "Vegetarian": 85,
            "Vegan": 80,
            "Pescatarian": 95,
            "Flexitarian": 90,
            "Keto": 85,
            "Paleo": 80
        },
        "Muscle Gain": {
            "Non-Vegetarian": 95,
            "Vegetarian": 85,
            "Vegan": 75,
            "Pescatarian": 90,
            "Flexitarian": 90,
            "Keto": 75,
            "Paleo": 85
        },
        "Improve Fitness": {
            "Non-Vegetarian": 90,
            "Vegetarian": 90,
            "Vegan": 85,
            "Pescatarian": 95,
            "Flexitarian": 95,
            "Keto": 75,
            "Paleo": 85
        }
    }
    
    # Get compatibility score or default to 80
    compat_score = compatibility.get(goal, {}).get(diet_type, 80)
    
    st.markdown(f"**Compatibility with {goal}:** {compat_score}/100")
    
    # Diet-specific recommendations
    if diet_type == "Vegan" and goal == "Muscle Gain":
        st.markdown("Our AI recommends focusing on plant-based protein sources like tempeh, seitan, and plant protein powders to meet your higher protein needs for muscle gain.")
    elif diet_type == "Keto" and goal == "Weight Loss":
        st.markdown("Our AI will optimize your keto meal plan for fat loss while ensuring adequate protein intake to preserve muscle mass.")
    elif diet_type == "Vegetarian" and goal == "Muscle Gain":
        st.markdown("Our AI recommends including eggs, Greek yogurt, and whey protein to meet your protein requirements for muscle growth.")
    
    # Display nutrients to monitor
    st.markdown("#### Key Nutrients to Monitor:")
    if diet_type == "Vegan":
        st.markdown("- **Vitamin B12**: Essential for nerve function and blood cell formation")
        st.markdown("- **Iron**: Critical for oxygen transport, plant sources are less bioavailable")
        st.markdown("- **Omega-3 Fatty Acids**: Important for heart and brain health")
    elif diet_type == "Keto":
        st.markdown("- **Electrolytes**: Sodium, potassium, and magnesium are critical on keto")
        st.markdown("- **Fiber**: Often lacking in keto diets, important for gut health")
    elif diet_type == "Paleo":
        st.markdown("- **Calcium**: May be lower without dairy products")
        st.markdown("- **Vitamin D**: Important for calcium absorption and immune function")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show additional options based on diet type
    if diet_type == "Non-Vegetarian":
        meat_preferences = st.multiselect(
            "Meat Preferences",
            ["None", "Chicken", "Turkey", "Beef", "Pork", "Lamb", "Fish", "Seafood"],
            default=["Chicken", "Fish"],
            help="Select the types of meat you prefer to include in your diet. Select 'None' if you want to exclude meat from your diet."
        )
        
        # Handle "None" selection
        if "None" in meat_preferences and len(meat_preferences) > 1:
            st.warning("You selected 'None' along with specific meat preferences. Only your specific preferences will be considered.")
            meat_preferences.remove("None")
        elif "None" in meat_preferences:
            st.info("You've selected 'None' for meat preferences. Your plan will minimize meat-based meals.")
            
    elif diet_type == "Pescatarian":
        meat_preferences = st.multiselect(
            "Seafood Preferences",
            ["None", "Fish", "Shellfish", "All Seafood"],
            default=["Fish"],
            help="Select the types of seafood you prefer to include in your diet. Select 'None' if you want to exclude seafood from your diet."
        )
        
        # Handle "None" selection
        if "None" in meat_preferences and len(meat_preferences) > 1:
            st.warning("You selected 'None' along with specific seafood preferences. Only your specific preferences will be considered.")
            meat_preferences.remove("None")
        elif "None" in meat_preferences:
            st.info("You've selected 'None' for seafood preferences. Your plan will minimize seafood-based meals.")
    else:
        meat_preferences = []
    
    # Food allergies and intolerances
    allergies = st.multiselect(
        "Food Allergies/Intolerances",
        ["None", "Gluten", "Dairy", "Nuts", "Eggs", "Soy", "Shellfish", "Peanuts", "Tree Nuts", "Fish", "Other"],
        default=["None"],
        help="Select any foods you need to avoid due to allergies or intolerances"
    )
    
    # Only show if 'Other' is selected
    other_allergy = ""
    if "Other" in allergies:
        other_allergy = st.text_input("Please specify your food allergies")
    
    # Foods to avoid (preferences)
    dislikes = st.multiselect(
        "Foods You Dislike",
        ["None", "Broccoli", "Mushrooms", "Onions", "Bell Peppers", "Eggplant", "Olives", "Tofu", "Quinoa", "Seafood", "Other"],
        default=["None"],
        help="Select foods you strongly dislike and want to avoid in your meal plan"
    )
    
    # Only show if 'Other' is selected
    other_dislike = ""
    if "Other" in dislikes:
        other_dislike = st.text_input("Please specify foods you dislike")
    
    # Meal frequency and timing
    meal_count = st.slider(
        "Meals per day", 
        min_value=2, 
        max_value=6, 
        value=3,
        help="Select the number of meals you prefer to eat each day, including snacks"
    )
    
    meal_timing = st.multiselect(
        "When do you prefer to eat?",
        ["Early Morning (5-7am)", "Morning (7-9am)", "Mid-Morning (9-11am)", 
         "Noon (11am-1pm)", "Afternoon (1-4pm)", "Evening (4-7pm)", "Night (7-10pm)", "Late Night (after 10pm)"],
        default=["Morning (7-9am)", "Noon (11am-1pm)", "Evening (4-7pm)"],
        help="Select your preferred meal times"
    )
    
    # Get appropriate default based on goal
    default_calorie_pref = "Moderate deficit" if st.session_state.user_data.get("goal") == "Weight Loss" else "Moderate surplus" if st.session_state.user_data.get("goal") == "Muscle Gain" else "Maintenance"
    
    calorie_preference = st.select_slider(
        "Calorie Control",
        options=["Strict deficit", "Moderate deficit", "Maintenance", "Moderate surplus", "Building surplus"],
        value=default_calorie_pref,
        help="Select your calorie target relative to your maintenance level"
    )
    
    # Explain calorie preference
    calorie_explanations = {
        "Strict deficit": "500-750 calorie deficit, faster weight loss",
        "Moderate deficit": "250-500 calorie deficit, sustainable weight loss",
        "Maintenance": "Eating at your maintenance level, weight stability",
        "Moderate surplus": "250-500 calorie surplus, gradual muscle gain",
        "Building surplus": "500-750 calorie surplus, faster muscle gain"
    }
    
    st.markdown(f"<div class='note-box'><strong>{calorie_preference}:</strong> {calorie_explanations[calorie_preference]}</div>", unsafe_allow_html=True)
    
    # Get collaborative meal recommendations from feedback system
    if st.session_state.user_data:
        # Check if we have enough user data for recommendations
        similar_users = st.session_state.feedback_analyzer.get_similar_users(st.session_state.user_data)
        
        if similar_users:
            meal_recommendations = st.session_state.feedback_analyzer.get_collaborative_recommendations(
                st.session_state.user_data, recommendation_type="meal", limit=3)
            
            if meal_recommendations:
                st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
                st.markdown("### 🧠 AI Meal Recommendations")
                st.markdown("Based on users with similar profiles, our AI recommends these meals:")
                
                for rec in meal_recommendations:
                    st.markdown(f"- **{rec['name']}** (Rating: {rec['score']:.1f}/5)")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Cooking preferences
    cooking_time = st.select_slider(
        "Maximum cooking time per meal",
        options=["5 minutes", "15 minutes", "30 minutes", "45 minutes", "60+ minutes"],
        value="30 minutes",
        help="Select how much time you can spend preparing each meal"
    )
    
    meal_prep = st.checkbox(
        "I prefer meal prepping (preparing multiple meals at once)",
        value=True,
        help="Meal prepping can save time and help with consistency"
    )
    
    form_data = {
        "diet_type": diet_type,
        "meat_preferences": meat_preferences if diet_type in ["Non-Vegetarian", "Pescatarian"] else [],
        "allergies": allergies,
        "other_allergy": other_allergy if "Other" in allergies else "",
        "dislikes": dislikes,
        "other_dislike": other_dislike if "Other" in dislikes else "",
        "meal_count": meal_count,
        "meal_timing": meal_timing,
        "calorie_preference": calorie_preference,
        "cooking_time": cooking_time,
        "meal_prep": meal_prep
    }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            prev_step()
    with col2:
        if st.button("Next: Equipment & Preferences"):
            save_form_data(form_data)
            next_step()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 4: Equipment & Preferences
elif st.session_state.step == 4:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("Step 4: Equipment & Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        workout_location = st.selectbox(
            "Where do you plan to workout?", 
            ["Home", "Gym", "Both", "Outdoors"], 
            help="This helps us select appropriate exercises for your available space and equipment"
        )
        
        # Only show equipment selection if working out at home or both
        if workout_location in ["Home", "Both"]:
            home_equipment = st.multiselect(
                "Equipment available at home",
                ["None", "Dumbbells", "Kettlebells", "Resistance Bands", "Yoga Mat", "Bench", 
                 "Pull-up Bar", "Treadmill", "Exercise Bike", "Jump Rope", "Medicine Ball", 
                 "Foam Roller", "Barbell and Weights", "Suspension Trainer (TRX)", "Other"],
                default=["None"],
                help="Select all equipment you have access to at home"
            )
            
            # Only show if 'Other' is selected
            other_equipment = ""
            if "Other" in home_equipment:
                other_equipment = st.text_input("Please specify other equipment")
                
            # Remove 'None' if other equipment is selected
            if "None" in home_equipment and len(home_equipment) > 1:
                home_equipment.remove("None")
        else:
            home_equipment = []
            other_equipment = ""
            
        # Exercise preferences
        exercise_preferences = st.multiselect(
            "Exercise types you enjoy",
            ["Weight Training", "Cardio", "HIIT", "Yoga", "Pilates", "Bodyweight", 
             "Circuit Training", "Running", "Swimming", "Cycling", "Sports", "Dancing"],
            default=["Weight Training", "Cardio"],
            help="Select exercise types you enjoy and would like to include in your plan"
        )
        
    with col2:
        # Limitations and injuries
        limitations = st.multiselect(
            "Do you have any physical limitations or injuries?",
            ["None", "Knee Pain/Injury", "Back Pain/Injury", "Shoulder Pain/Injury", 
             "Wrist Pain/Injury", "Ankle Pain/Injury", "Hip Pain/Injury", "Balance Issues", "Other"],
            default=["None"],
            help="These will help us customize exercises to avoid exacerbating any issues"
        )
        
        # Only show if 'Other' is selected
        other_limitation = ""
        if "Other" in limitations:
            other_limitation = st.text_input("Please specify your physical limitation")
            
        # Remove 'None' if limitations are selected
        if "None" in limitations and len(limitations) > 1:
            limitations.remove("None")
        
        # Focus areas
        focus_areas = st.multiselect(
            "Areas you'd like to focus on",
            ["Full Body", "Upper Body", "Lower Body", "Core", "Arms", "Chest", 
             "Back", "Shoulders", "Legs", "Glutes", "Cardiovascular Endurance"],
            default=["Full Body"],
            help="Select body areas you'd particularly like to target"
        )
        
        # Workout schedule preferences
        st.markdown("### Workout Schedule Preferences")
        preferred_days = st.multiselect(
            "Preferred workout days",
            ["No preference", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=["No preference"],
            help="Select days you prefer to workout. If no preference, we'll distribute optimally."
        )
        
        # Morning vs evening preference
        time_preference = st.radio(
            "Preferred workout time",
            ["Morning", "Afternoon", "Evening", "No preference"],
            index=3,
            help="When do you prefer to exercise?"
        )
    
    # AI insight box based on selections
    if workout_location and exercise_preferences:
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Workout Analysis")
        
        # Location-based insights
        if workout_location == "Home" and "None" in home_equipment:
            st.markdown("**Equipment-Free Plan:** Our AI will design an effective bodyweight routine requiring no equipment, perfect for home workouts.")
        elif workout_location == "Home" and home_equipment:
            st.markdown(f"**Home Gym Analysis:** Based on your available equipment, our AI will create optimized routines using {', '.join(home_equipment[:3])} and other items you have.")
        elif workout_location == "Gym":
            st.markdown("**Gym-Based Plan:** Our AI will incorporate a variety of equipment available at standard gyms for a comprehensive fitness plan.")
        elif workout_location == "Outdoors":
            st.markdown("**Outdoor Workout Plan:** Our AI will design workouts optimized for outdoor environments, incorporating natural elements and bodyweight exercises.")
        
        # Exercise preference insights
        pref_types = []
        if "Weight Training" in exercise_preferences or "Bodyweight" in exercise_preferences:
            pref_types.append("strength training")
        if "Cardio" in exercise_preferences or "Running" in exercise_preferences or "Swimming" in exercise_preferences or "Cycling" in exercise_preferences:
            pref_types.append("cardio")
        if "HIIT" in exercise_preferences or "Circuit Training" in exercise_preferences:
            pref_types.append("high-intensity interval work")
        if "Yoga" in exercise_preferences or "Pilates" in exercise_preferences:
            pref_types.append("flexibility and mobility work")
            
        if pref_types:
            st.markdown(f"**Exercise Analysis:** Our AI will prioritize {', '.join(pref_types)} based on your preferences, while ensuring balanced fitness development.")
        
        # Limitation insights
        if limitations and "None" not in limitations:
            st.markdown("**Adaptive Programming:** Our AI will modify exercises to accommodate your physical limitations while still providing effective workouts.")
            
            # Specific modifications based on limitations
            if "Knee Pain/Injury" in limitations:
                st.markdown("- We'll replace high-impact exercises with low-impact alternatives that protect your knees")
            if "Back Pain/Injury" in limitations:
                st.markdown("- Exercises will be selected to minimize spinal loading and strengthen core stabilizers")
            if "Shoulder Pain/Injury" in limitations:
                st.markdown("- Upper body pushing and pulling movements will be modified to accommodate shoulder health")
        
        # Focus area insights
        if "Full Body" in focus_areas:
            st.markdown("**Balance Analysis:** Our AI will ensure comprehensive whole-body development while still addressing your specific focus areas.")
        elif len(focus_areas) >= 3:
            st.markdown(f"**Specialization Analysis:** Our AI will create a split routine targeting your priority areas: {', '.join(focus_areas[:3])}.")
        elif focus_areas:
            st.markdown(f"**Target Area Analysis:** Our AI will emphasize {', '.join(focus_areas)} while maintaining overall fitness development.")
        
        # Scheduling insights
        if "No preference" not in preferred_days:
            st.markdown(f"**Schedule Optimization:** Our AI will arrange workout types to optimize recovery between similar muscle groups across your preferred days.")
        
        # Additional coaching tips
        st.markdown("### AI Coaching Tips")
        
        # Goal-specific tips
        goal = st.session_state.user_data.get("goal", "Weight Loss")
        
        if goal == "Weight Loss":
            st.markdown('<div class="ai-coaching-tip">For weight loss goals, our AI recommends incorporating both strength training and cardio in a 2:1 ratio for optimal fat loss while preserving muscle.</div>', unsafe_allow_html=True)
        elif goal == "Muscle Gain":
            st.markdown('<div class="ai-coaching-tip">For muscle building, our AI suggests focusing on progressive overload in your strength training, with strategic cardio to support recovery without interfering with gains.</div>', unsafe_allow_html=True)
        elif goal == "Improve Fitness":
            st.markdown('<div class="ai-coaching-tip">For overall fitness improvement, our AI recommends a balanced approach with equal emphasis on strength, cardio, and mobility work.</div>', unsafe_allow_html=True)
        
        # Equipment-specific tips
        if workout_location == "Home" and home_equipment and "None" not in home_equipment:
            st.markdown('<div class="ai-coaching-tip">Our AI will design creative workout variations to maximize the effectiveness of your home equipment, ensuring continuous progress without equipment limitations.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Form data to be saved
    form_data = {
        "workout_location": workout_location,
        "home_equipment": home_equipment if workout_location in ["Home", "Both"] else [],
        "other_equipment": other_equipment if "Other" in home_equipment else "",
        "exercise_preferences": exercise_preferences,
        "limitations": limitations,
        "other_limitation": other_limitation if "Other" in limitations else "",
        "focus_areas": focus_areas,
        "workout_schedule": preferred_days,
        "time_preference": time_preference
    }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            prev_step()
    with col2:
        if st.button("Generate Plan"):
            save_form_data(form_data)
            next_step()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 5: Results
elif st.session_state.step == 5:
    st.subheader("Your AI-Personalized Fitness Plan")
    
    with st.spinner("AI is generating your personalized plan... This may take a moment."):
        # Map UI-friendly values to data structure values
        goal_mapping = {
            "Weight Loss": "weight_loss",
            "Muscle Gain": "muscle_gain",
            "Improve Fitness": "fitness",
            "Body Toning": "toning",
            "Endurance Building": "endurance"
        }
        
        difficulty_mapping = {
            "Beginner": "beginner",
            "Easy": "easy",
            "Moderate": "moderate",
            "Challenging": "challenging",
            "Advanced": "advanced"
        }
        
        # Use the enhanced planner to get a personalized plan
        plan = get_enhanced_plan(
            goal=goal_mapping.get(st.session_state.user_data["goal"], "weight_loss"), 
            difficulty=difficulty_mapping.get(st.session_state.user_data["difficulty"], "moderate"),
            diet_type=st.session_state.user_data["diet_type"].lower(),
            user_data=st.session_state.user_data
        )
        
        if plan:
            # Store plan in session state for tracking
            st.session_state.plan = plan
            
            # Display the plan using the enhanced display function
            display_enhanced_plan(plan, st.session_state.user_data)
            
            # Add feedback mechanism
            st.markdown("### Help Us Improve Your Plan")
            st.markdown("Rate your satisfaction with this plan to help our AI system learn from your feedback:")
            
            satisfaction = st.slider("Overall Satisfaction", 1, 5, 5)
            feedback_text = st.text_area("Additional Comments (Optional)")
            
            if st.button("Submit Feedback"):
                # Prepare feedback data
                feedback = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "satisfaction": satisfaction,
                    "comments": feedback_text,
                    "workout_ratings": {},
                    "meal_ratings": {}
                }
                
                # For simplicity, use first workout as sample
                if "workouts" in plan and plan["workouts"]:
                    first_workout = plan["workouts"][0]
                    feedback["workout_ratings"][first_workout.get("name", "Workout")] = satisfaction
                
                # For simplicity, use first meal as sample
                if "categorized_meals" in plan and "breakfast" in plan["categorized_meals"] and plan["categorized_meals"]["breakfast"]:
                    first_meal = plan["categorized_meals"]["breakfast"][0]
                    feedback["meal_ratings"][first_meal.get("name", "Meal")] = satisfaction
                
                # Save feedback
                success = st.session_state.feedback_analyzer.add_user_feedback(
                    st.session_state.user_id,
                    st.session_state.user_data,
                    feedback
                )
                
                if success:
                    st.success("Thank you for your feedback! We'll use it to improve future recommendations.")
                else:
                    st.error("There was an error saving your feedback. Please try again.")
            
            # Option to save plan (simplified implementation)
            if st.button("📥 Save Plan as PDF"):
                st.success("This feature would generate a PDF of your plan. (Implementation pending)")
            
            # Option to share plan
            if st.button("📤 Share Your Plan"):
                st.success("This feature would allow you to share your plan. (Implementation pending)")
                
            # Option to move to tracking
            if st.button("📊 Track Your Progress"):
                st.session_state.step = 6
                st.rerun()
                
        else:
            st.error("No plan could be generated for your criteria. Please try different options.")
    
    # Option to restart
    if st.button("🔄 Start Over"):
        st.session_state.step = 1
        st.session_state.user_data = {}
        st.rerun()

# Step 6: Progress Tracking
elif st.session_state.step == 6:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.subheader("Progress Tracking & Analytics")
    
    # Tabs for different tracking areas
    track_tab1, track_tab2, track_tab3, track_tab4 = st.tabs(["Body Metrics", "Workout Performance", "Nutrition", "AI Analysis"])
    
    with track_tab1:
        st.subheader("Body Measurements")
        col1, col2 = st.columns(2)
        
        with col1:
            # Weight tracking
            current_weight = st.number_input("Current Weight (kg)", 
                                            min_value=30.0, max_value=200.0, 
                                            value=st.session_state.user_data.get("weight", 70.0))
            measure_date = st.date_input("Measurement Date", datetime.now())
            
            if st.button("Log Weight"):
                st.session_state.tracking_data['weight_history'].append({
                    'date': measure_date.strftime("%Y-%m-%d"),
                    'weight': current_weight
                })
                st.success("Weight logged successfully!")
                
        with col2:
            # Body measurements
            measurement_type = st.selectbox("Measurement Type", 
                                           ["Chest", "Waist", "Hips", "Arms", "Thighs"])
            measurement_value = st.number_input(f"{measurement_type} (cm)", 
                                              min_value=20.0, max_value=200.0, value=80.0)
            
            if st.button("Log Measurement"):
                key = measurement_type.lower()
                st.session_state.tracking_data['measurement_history'][key].append({
                    'date': measure_date.strftime("%Y-%m-%d"),
                    'value': measurement_value
                })
                st.success(f"{measurement_type} measurement logged!")
        
        # Display weight history chart
        if st.session_state.tracking_data['weight_history']:
            st.subheader("Weight History")
            weight_data = pd.DataFrame(st.session_state.tracking_data['weight_history'])
            st.line_chart(weight_data.set_index('date')['weight'])
    
    with track_tab2:
        st.subheader("Workout Tracking")
        
        # Display workouts from plan
        if 'plan' in st.session_state and 'workouts' in st.session_state.plan:
            workout_options = [w.get('name', f"Workout {i}") for i, w in 
                              enumerate(st.session_state.plan['workouts'])]
            selected_workout = st.selectbox("Select Completed Workout", workout_options)
            
            perceived_difficulty = st.slider("Perceived Difficulty", 1, 10, 5)
            performance_level = st.slider("Performance Level", 1, 10, 7)
            notes = st.text_area("Workout Notes")
            
            if st.button("Log Workout"):
                if selected_workout not in st.session_state.tracking_data['workout_completions']:
                    st.session_state.tracking_data['workout_completions'][selected_workout] = []
                
                st.session_state.tracking_data['workout_completions'][selected_workout].append({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'difficulty': perceived_difficulty,
                    'performance': performance_level,
                    'notes': notes
                })
                st.success(f"{selected_workout} logged successfully!")
                
            # Display workout completion history
            if st.session_state.tracking_data['workout_completions']:
                st.subheader("Workout History")
                for workout, logs in st.session_state.tracking_data['workout_completions'].items():
                    with st.expander(f"{workout} ({len(logs)} sessions)"):
                        for i, log in enumerate(logs):
                            st.markdown(f"**Session {i+1}:** {log['date']}")
                            st.markdown(f"Difficulty: {log['difficulty']}/10 | Performance: {log['performance']}/10")
                            if log['notes']:
                                st.markdown(f"Notes: {log['notes']}")
    
    with track_tab3:
        st.subheader("Nutrition Tracking")
        
        # Simple nutrition adherence tracking
        col1, col2 = st.columns(2)
        
        with col1:
            track_date = st.date_input("Date", datetime.now(), key="nut_date")
            calories_consumed = st.number_input("Calories Consumed", 
                                              min_value=0, max_value=5000, value=2000)
            protein_consumed = st.number_input("Protein Consumed (g)", 
                                             min_value=0, max_value=300, value=100)
        
        with col2:
            carbs_consumed = st.number_input("Carbs Consumed (g)", 
                                           min_value=0, max_value=500, value=200)
            fat_consumed = st.number_input("Fat Consumed (g)", 
                                         min_value=0, max_value=200, value=70)
            adherence = st.slider("Diet Adherence Level", 0, 100, 80)
        
        if st.button("Log Nutrition Data"):
            st.session_state.tracking_data['nutrition_adherence'].append({
                'date': track_date.strftime("%Y-%m-%d"),
                'calories': calories_consumed,
                'protein': protein_consumed,
                'carbs': carbs_consumed,
                'fat': fat_consumed,
                'adherence': adherence
            })
            st.success("Nutrition data logged successfully!")
            
        # Display nutrition history
        if st.session_state.tracking_data['nutrition_adherence']:
            st.subheader("Nutrition History")
            nutrition_data = pd.DataFrame(st.session_state.tracking_data['nutrition_adherence'])
            st.line_chart(nutrition_data.set_index('date')[['calories', 'protein', 'carbs', 'fat']])
            
            st.subheader("Diet Adherence")
            st.line_chart(nutrition_data.set_index('date')['adherence'])
    
    with track_tab4:
        st.subheader("AI Progress Analysis")
        
        if (len(st.session_state.tracking_data['weight_history']) > 1 or 
            len(st.session_state.tracking_data['workout_completions']) > 0):
            
            st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Progress Insights")
            
            # Weight change analysis
            if len(st.session_state.tracking_data['weight_history']) > 1:
                weight_data = st.session_state.tracking_data['weight_history']
                first_weight = weight_data[0]['weight']
                last_weight = weight_data[-1]['weight']
                weight_change = last_weight - first_weight
                
                goal = st.session_state.user_data.get("goal", "Weight Loss")
                
                if goal == "Weight Loss" and weight_change < 0:
                    st.markdown(f"✅ **Weight Loss Progress:** You've lost {abs(weight_change):.1f} kg, which is excellent progress toward your goal!")
                elif goal == "Weight Loss" and weight_change >= 0:
                    st.markdown(f"⚠️ **Weight Loss Challenge:** Your weight has {weight_change:.1f} kg changed. The AI recommends increasing your calorie deficit by 200-300 calories per day.")
                elif goal == "Muscle Gain" and weight_change > 0:
                    st.markdown(f"✅ **Muscle Gain Progress:** You've gained {weight_change:.1f} kg, which aligns with your muscle building goal.")
                elif goal == "Muscle Gain" and weight_change <= 0:
                    st.markdown(f"⚠️ **Muscle Gain Challenge:** You haven't gained weight yet. The AI recommends increasing your calorie intake by 300-500 calories per day and focusing on progressive overload.")
            
            # Workout analysis
            if st.session_state.tracking_data['workout_completions']:
                workout_count = sum(len(logs) for logs in st.session_state.tracking_data['workout_completions'].values())
                st.markdown(f"📊 **Workout Consistency:** You've completed {workout_count} workouts.")
                
                # Calculate average difficulty and performance
                all_difficulties = []
                all_performances = []
                for logs in st.session_state.tracking_data['workout_completions'].values():
                    for log in logs:
                        all_difficulties.append(log['difficulty'])
                        all_performances.append(log['performance'])
                
                if all_difficulties and all_performances:
                    avg_difficulty = sum(all_difficulties) / len(all_difficulties)
                    avg_performance = sum(all_performances) / len(all_performances)
                    
                    if avg_difficulty > 7 and avg_performance < 6:
                        st.markdown("🔄 **Training Adjustment:** Your workouts may be too challenging. The AI recommends decreasing intensity temporarily to improve form and performance.")
                    elif avg_difficulty < 5 and avg_performance > 8:
                        st.markdown("🔄 **Training Adjustment:** Your workouts may be too easy. The AI recommends increasing weights or intensity to continue progressing.")
                    elif avg_performance > 7:
                        st.markdown("🔄 **Training Adjustment:** Your performance is excellent! The AI recommends adding new exercise variations to continue challenging your muscles.")
            
            # Calculate adherence and project timeline
            if st.session_state.tracking_data['nutrition_adherence']:
                adherence_values = [entry['adherence'] for entry in st.session_state.tracking_data['nutrition_adherence']]
                avg_adherence = sum(adherence_values) / len(adherence_values)
                
                goal = st.session_state.user_data.get("goal", "Weight Loss")
                target_weight = st.session_state.user_data.get("target_weight", 0)
                current_weight = st.session_state.user_data.get("weight", 0)
                
                if goal == "Weight Loss" and target_weight > 0 and current_weight > 0:
                    weekly_loss = 0.5  # kg per week
                    if avg_adherence > 90:
                        weekly_loss = 0.7
                    elif avg_adherence < 70:
                        weekly_loss = 0.3
                    
                    weeks_needed = abs(current_weight - target_weight) / weekly_loss
                    st.markdown(f"⏱️ **Timeline Projection:** Based on your {avg_adherence:.1f}% adherence rate, you may reach your goal weight in approximately {int(weeks_needed)} weeks.")
                
                # Nutrition recommendations
                if avg_adherence < 70:
                    st.markdown("🍽️ **Nutrition Strategy:** Your adherence is challenging. The AI recommends simplifying your meal plan and focusing on meal prep to improve consistency.")
                elif 70 <= avg_adherence < 85:
                    st.markdown("🍽️ **Nutrition Strategy:** Your adherence is good. To improve further, try planning one 'free meal' per week to maintain sustainability.")
                else:
                    st.markdown("🍽️ **Nutrition Strategy:** Your adherence is excellent! Consider adjusting your calorie targets as your fitness improves.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Start logging your workouts and measurements to receive AI analysis of your progress.")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Plan"):
            st.session_state.step = 5
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.step = 1
            st.session_state.user_data = {}
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with updated copyright date
st.markdown('<div class="footer">FitLife Pro AI+ - Your Advanced Personal Fitness Journey © 2025</div>', unsafe_allow_html=True)
