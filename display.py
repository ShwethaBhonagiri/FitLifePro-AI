import streamlit as st
import pandas as pd
import math
import random
from datetime import datetime, timedelta

def display_enhanced_plan(plan, user_data):
    """
    Display an enhanced fitness plan with improved UI and AI-driven insights.
    
    Args:
        plan (dict): The fitness plan data
        user_data (dict): User profile data
    """
    # Calculate BMI and status
    bmi = user_data["weight"] / ((user_data["height"]/100) ** 2)
    bmi_status = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    
    # Get nutrition data from the plan
    nutrition = plan.get("nutrition", {})
    target_calories = nutrition.get("target_calories", 2000)
    
    # Display user stats with improved UI
    st.header("📊 Your Fitness Profile")
    
    # Profile card with custom CSS
    st.markdown("""
    <style>
    .profile-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .nutrition-card {
        background-color: #e9f7ef;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .workout-day {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 5px;
    }
    .rest-day {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 5px;
    }
    .ai-insight-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #4285f4;
    }
    .ai-coaching-tip {
        background-color: #e8f0fe;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        border-left: 3px solid #4285f4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add AI insights display
    if "ai_insights" in plan:
        insights = plan["ai_insights"]
        
        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Analysis of Your Fitness Profile")
        
        # Display key AI insights
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"**Fitness Level:** {insights.get('fitness_level', 5)}/10")
            st.markdown(f"**Workout Capacity:** {insights.get('workout_capacity', 70)}/100")
        
        with cols[1]:
            st.markdown(f"**Recovery Profile:** {insights.get('recovery_profile', 'moderate').title()}")
            st.markdown(f"**Body Category:** {insights.get('body_category', 'normal_weight').replace('_', ' ').title()}")
        
        with cols[2]:
            st.markdown(f"**Protein Needs:** {insights.get('protein_factor', 0.8)*2.2:.1f}g/kg of body weight")
            if "body_fat_estimate" in nutrition:
                st.markdown(f"**Body Fat Estimate:** {nutrition['body_fat_estimate']}%")
        
        # Display coaching tips if available
        if "coaching_tips" in plan:
            st.markdown("### AI Coaching Tips")
            for i, tip in enumerate(plan["coaching_tips"]):
                st.markdown(f'<div class="ai-coaching-tip">{tip}</div>', unsafe_allow_html=True)
        
        # Display adaptive changes if available
        if "ai_adaptive_changes" in plan and plan["ai_adaptive_changes"]:
            st.markdown("### AI Adaptive Adjustments")
            st.markdown("Your plan has been personalized based on:")
            for change in plan["ai_adaptive_changes"]:
                st.markdown(f"- {change}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # User profile in a nice card layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Age</p><p class='metric-value'>{user_data['age']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Gender</p><p class='metric-value'>{user_data['gender']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Weight</p><p class='metric-value'>{user_data['weight']} kg</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Height</p><p class='metric-value'>{user_data['height']} cm</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>BMI</p><p class='metric-value'>{bmi:.1f} ({bmi_status})</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Activity Level</p><p class='metric-value'>{user_data['activity_level']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Goal</p><p class='metric-value'>{user_data['goal']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Timeline</p><p class='metric-value'>{user_data.get('timeline', 3)} months</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Diet Type</p><p class='metric-value'>{user_data['diet_type']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Daily Calories</p><p class='metric-value'>{int(target_calories)} kcal</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Workout Frequency</p><p class='metric-value'>{user_data['workout_frequency']} days/week</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-label'>Workout Duration</p><p class='metric-value'>{user_data['workout_duration']} minutes</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display any health warnings
    if user_data.get("health_conditions") and "None" not in user_data["health_conditions"]:
        st.warning(f"⚠️ Please consult with your doctor before starting this program due to your health conditions: {', '.join(user_data['health_conditions'])}.")
    
    # Create tabs for different sections with more detailed content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Weekly Plan", "Workouts", "Nutrition", "Meal Planning", "Equipment"])
    
    # Tab 1: Weekly Plan with improved visualization
    with tab1:
        st.subheader("📅 Your Weekly Schedule")
        
        # Get the weekly plan
        weekly_plan = plan.get("weekly_plan", {})
        
        # Calculate the current week dates
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        
        # Create a weekly calendar with dates
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Display the weekly schedule in a nicer format
        for i, day in enumerate(days):
            day_lower = day.lower()
            current_date = start_of_week + timedelta(days=i)
            date_str = current_date.strftime("%b %d")
            
            day_plan = weekly_plan.get(day_lower, {"rest_day": True, "workout_type": "Rest"})
            is_rest_day = day_plan.get("rest_day", True)
            
            if is_rest_day:
                st.markdown(f"""
                <div class="rest-day">
                    <h4>{day} - {date_str}</h4>
                    <p>🛌 <b>Rest Day</b></p>
                    <p>Focus on recovery and stretching</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                workout_type = day_plan.get("workout_type", "General")
                workout_name = day_plan.get("workout_name", workout_type)
                duration = day_plan.get("duration", 30)
                
                # Get emoji based on workout type
                emoji = "🏋️" if workout_type.lower() in ["strength", "weights"] else "🏃" if workout_type.lower() in ["cardio", "running"] else "🧘" if workout_type.lower() in ["yoga", "stretching"] else "⚡" if workout_type.lower() in ["hiit", "circuit"] else "💪"
                
                st.markdown(f"""
                <div class="workout-day">
                    <h4>{day} - {date_str}</h4>
                    <p>{emoji} <b>{workout_name}</b></p>
                    <p>Type: {workout_type.title()}</p>
                    <p>Duration: {duration} minutes</p>
                </div>
                """, unsafe_allow_html=True)
            
        # Add a progress tracking section
        st.subheader("📈 Weekly Progress Tracker")
        
        # Sample metrics to track
        metrics = ["Weight (kg)", "Workout Compliance (%)", "Calorie Compliance (%)", "Water Intake (L)"]
        
        # Create editable dataframe for tracking
        if "progress_data" not in st.session_state:
            # Initialize with random sample data
            st.session_state.progress_data = pd.DataFrame({
                "Metric": metrics,
                "Goal": [user_data.get("target_weight", user_data["weight"] - 5), 100, 90, 2.5],
                "Current": [user_data["weight"], 0, 0, 0]
            })
        
        # Display editable dataframe
        edited_df = st.data_editor(
            st.session_state.progress_data,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "Goal": st.column_config.NumberColumn("Goal"),
                "Current": st.column_config.NumberColumn("Current", help="Enter your current values")
            },
            hide_index=True,
            key="progress_editor"
        )
        
        # Update the session state
        st.session_state.progress_data = edited_df
    
    # Tab 2: Workouts with more detailed information
    with tab2:
        st.subheader("🏋️ Your Workout Plan")
        
        # Get workouts from the plan
        workouts = plan.get("workouts", [])
        
        if not workouts:
            st.info("No specific workouts found in your plan. Please regenerate your plan.")
        else:
            # Create a search/filter option
            workout_names = [w.get("name", f"Workout {i+1}") for i, w in enumerate(workouts)]
            workout_types = list(set(w.get("type", "General").title() for w in workouts))
            
            col1, col2 = st.columns(2)
            with col1:
                selected_type = st.selectbox("Filter by type:", ["All"] + workout_types)
            with col2:
                search_term = st.text_input("Search workouts:", "")
            
            # Filter workouts based on selection
            filtered_workouts = []
            for workout in workouts:
                workout_type = workout.get("type", "").title()
                workout_name = workout.get("name", "")
                
                type_match = selected_type == "All" or workout_type == selected_type
                search_match = search_term.lower() in workout_name.lower() or search_term == ""
                
                if type_match and search_match:
                    filtered_workouts.append(workout)
            
            # Display the filtered workouts
            for i, workout in enumerate(filtered_workouts):
                with st.expander(f"{i+1}. {workout.get('name', 'Workout')} ({workout.get('type', 'General').title()})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(workout.get("image", "https://via.placeholder.com/150"), width=200)
                        st.markdown(f"**Duration:** {workout.get('duration', 30)} minutes")
                        st.markdown(f"**Intensity:** {workout.get('intensity', 'Moderate')}")
                        st.markdown(f"**Equipment:** {', '.join(workout.get('equipment', ['Bodyweight']))}")
                        if "video" in workout:
                            st.markdown(f"[Watch Tutorial]({workout['video']})")
                    
                    with col2:
                        st.markdown(f"**Focus Areas:** {', '.join(workout.get('focus_areas', ['Full Body']))}")
                        st.markdown(f"**Description:** {workout.get('description', 'No description available.')}")
                        
                        # Display exercises in a table
                        if "exercises" in workout:
                            exercise_data = []
                            for ex in workout["exercises"]:
                                exercise_data.append({
                                    "Exercise": ex["name"],
                                    "Sets": ex.get("sets", 1),
                                    "Reps/Duration": ex.get("reps", "") if "reps" in ex else f"{ex.get('duration', '30 sec')}",
                                    "Rest": f"{ex.get('rest', 60)} sec"
                                })
                            
                            st.markdown("**Exercise Plan:**")
                            st.table(pd.DataFrame(exercise_data))
                            
                            # Add a notes section
                            st.text_area("Workout Notes:", key=f"notes_{i}", height=100)
    
    # Tab 3: Nutrition with enhanced visualization
    with tab3:
        st.subheader("🥗 Your Nutrition Plan")
        
        # Display nutrition card
        st.markdown('<div class="nutrition-card">', unsafe_allow_html=True)
        
        # Display calorie details
        bmr = nutrition.get("bmr", int(target_calories * 0.8))
        tdee = nutrition.get("tdee", int(target_calories * 1.2))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Calorie Breakdown")
            st.markdown(f"**Basal Metabolic Rate (BMR):** {bmr} kcal")
            st.markdown(f"**Total Daily Energy Expenditure (TDEE):** {tdee} kcal")
            st.markdown(f"**Target Daily Calories:** {int(target_calories)} kcal")
            
            # Explain the calorie targets
            if target_calories < tdee:
                deficit = tdee - target_calories
                st.markdown(f"**Calorie Deficit:** {int(deficit)} kcal/day")
                st.markdown(f"**Expected Weight Loss:** ~{(deficit * 7 / 7700):.2f} kg/week")
            elif target_calories > tdee:
                surplus = target_calories - tdee
                st.markdown(f"**Calorie Surplus:** {int(surplus)} kcal/day")
                st.markdown(f"**Expected Weight Gain:** ~{(surplus * 7 / 7700):.2f} kg/week")
        
        with col2:
            # Get macro data
            macros = nutrition.get("macros", {
                "protein": {"g": int(target_calories * 0.3 / 4), "pct": 30},
                "carbs": {"g": int(target_calories * 0.4 / 4), "pct": 40},
                "fat": {"g": int(target_calories * 0.3 / 9), "pct": 30}
            })
            
            # Display macros
            st.markdown("### Macronutrient Targets")
            
            protein = macros.get("protein", {"g": 0, "pct": 0})
            carbs = macros.get("carbs", {"g": 0, "pct": 0})
            fat = macros.get("fat", {"g": 0, "pct": 0})
            
            # Display using metrics for better visualization
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Protein", f"{protein.get('g', 0)} g", f"{protein.get('pct', 0)}%")
            with c2:
                st.metric("Carbs", f"{carbs.get('g', 0)} g", f"{carbs.get('pct', 0)}%")
            with c3:
                st.metric("Fat", f"{fat.get('g', 0)} g", f"{fat.get('pct', 0)}%")
            
            # Show protein per kg of bodyweight
            protein_per_kg = protein.get('g', 0) / user_data["weight"]
            st.markdown(f"**Protein per kg of bodyweight:** {protein_per_kg:.2f} g/kg")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Nutrition tips based on goals
        st.subheader("Nutrition Tips & Guidelines")
        
        # Different tips based on user's goal
        if user_data["goal"] == "Weight Loss":
            tips = [
                "Focus on protein-rich foods to maintain muscle mass during weight loss",
                "Include plenty of fiber-rich vegetables to help with satiety",
                "Drink water before meals to help control hunger",
                "Limit processed foods and added sugars",
                "Consider intermittent fasting if it suits your lifestyle",
                "Plan meals ahead to avoid impulsive food choices"
            ]
        elif user_data["goal"] == "Muscle Gain":
            tips = [
                "Ensure you're getting enough protein (1.6-2.2g per kg of bodyweight)",
                "Don't skimp on carbs - they fuel your workouts and recovery",
                "Time your protein intake around workouts for optimal results",
                "Consider a pre-workout meal rich in carbs and protein",
                "Healthy fats are essential for hormone production",
                "Calories matter - make sure you're in a slight surplus"
            ]
        else:
            tips = [
                "Focus on whole, unprocessed foods for better overall health",
                "Aim for a diverse diet with plenty of colorful fruits and vegetables",
                "Stay hydrated throughout the day",
                "Balance your plate with protein, complex carbs, and healthy fats",
                "Consider timing your nutrition around your workouts",
                "Listen to your body's hunger and fullness cues"
            ]
        
        # Display tips
        for i, tip in enumerate(tips):
            st.markdown(f"**{i+1}.** {tip}")
        
        # Hydration recommendations
        st.subheader("Hydration Guidelines")
        
        # Calculate recommended water intake
        recommended_water = user_data["weight"] * 0.033  # 33ml per kg bodyweight
        
        st.markdown(f"**Daily water target:** {recommended_water:.1f} liters")
        st.markdown("""
        - Drink a glass of water first thing in the morning
        - Carry a water bottle with you throughout the day
        - Drink water before, during, and after workouts
        - Set reminders to drink water regularly
        """)
    
    # Tab 4: Meal Planning with sample meals
    with tab4:
        st.subheader("🍽️ Your Meal Plan")
        
        # Get the meal plan from the plan
        meal_plan = plan.get("meal_plan", {})
        categorized_meals = plan.get("categorized_meals", {})
        
        # Create day selector
        selected_day = st.selectbox("Select day:", days)
        selected_day_lower = selected_day.lower()
        
        if selected_day_lower in meal_plan:
            day_meals = meal_plan[selected_day_lower]
            
            # Display daily calorie total
            daily_calories = sum(meal.get("calories", 0) for meal_type, meal in day_meals.items() if meal_type != "snacks")
            if "snacks" in day_meals:
                daily_calories += sum(snack.get("calories", 0) for snack in day_meals["snacks"])
            
            calorie_percent = (daily_calories / target_calories) * 100
            
            st.markdown(f"**Daily Calorie Total:** {daily_calories} kcal ({calorie_percent:.1f}% of target)")
            
            # Display meals for the day
            st.markdown("### Meals for " + selected_day)
            
            meal_types = ["breakfast", "lunch", "dinner"]
            for meal_type in meal_types:
                if meal_type in day_meals:
                    meal = day_meals[meal_type]
                    st.markdown(f"**{meal_type.capitalize()}**")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(meal.get("image", "https://via.placeholder.com/150"), width=150)
                        st.markdown(f"**Calories:** {meal.get('calories', 0)} kcal")
                        
                        if "macros" in meal:
                            macros = meal["macros"]
                            st.markdown(f"**Protein:** {macros.get('protein', 0)}g")
                            st.markdown(f"**Carbs:** {macros.get('carbs', 0)}g")
                            st.markdown(f"**Fat:** {macros.get('fat', 0)}g")
                    
                    with col2:
                        st.markdown(f"**{meal.get('name', 'Meal')}**")
                        st.markdown(f"**Diet Type:** {', '.join(meal.get('diet_types', ['Standard']))}")
                        
                        if "recipe" in meal:
                            st.markdown("**Recipe:**")
                            st.markdown(meal["recipe"])
                        else:
                            st.markdown("*No recipe available*")
            
            # Display snacks if available
            if "snacks" in day_meals and day_meals["snacks"]:
                st.markdown("### Snacks")
                
                snacks = day_meals["snacks"]
                snack_cols = st.columns(min(len(snacks), 3))
                
                for i, snack in enumerate(snacks):
                    with snack_cols[i % len(snack_cols)]:
                        st.markdown(f"**{snack.get('name', 'Snack')}**")
                        st.image(snack.get("image", "https://via.placeholder.com/150"), width=100)
                        st.markdown(f"**Calories:** {snack.get('calories', 0)} kcal")
                        
                        if "macros" in snack:
                            macros = snack["macros"]
                            st.markdown(f"P: {macros.get('protein', 0)}g | C: {macros.get('carbs', 0)}g | F: {macros.get('fat', 0)}g")
        else:
            st.info(f"No meal plan available for {selected_day}.")
        
        # Allow users to customize/swap meals
        st.subheader("Meal Customization")
        
        # Let users select meal type to customize
        meal_type_to_swap = st.selectbox("Select meal to customize:", ["Breakfast", "Lunch", "Dinner", "Snack"])
        meal_type_lower = meal_type_to_swap.lower()
        
        # Get available alternatives
        alternatives = categorized_meals.get(meal_type_lower, [])
        
        if alternatives:
            st.markdown(f"### Alternative {meal_type_to_swap} Options")
            
            # Display in a scrollable area
            alt_cols = st.columns(min(len(alternatives), 3))
            for i, alt_meal in enumerate(alternatives):
                with alt_cols[i % len(alt_cols)]:
                    st.markdown(f"**{alt_meal.get('name', 'Meal Option')}**")
                    st.image(alt_meal.get("image", "https://via.placeholder.com/150"), width=100)
                    st.markdown(f"**Calories:** {alt_meal.get('calories', 0)} kcal")
                    
                    if "macros" in alt_meal:
                        macros = alt_meal["macros"]
                        st.markdown(f"P: {macros.get('protein', 0)}g | C: {macros.get('carbs', 0)}g | F: {macros.get('fat', 0)}g")
                    
                    # Create a key for each button that includes meal type and index
                    st.button(f"Swap with current {meal_type_to_swap}", key=f"swap_{meal_type_lower}_{i}")
        else:
            st.info(f"No alternative {meal_type_lower} options available.")
        
        # Grocery list generator
        st.subheader("Weekly Grocery List")
        
        if st.button("Generate Grocery List"):
            grocery_items = set()
            
            # Extract ingredients from meal recipes (simplified implementation)
            common_ingredients = [
                "Chicken breast", "Eggs", "Greek yogurt", "Oats", "Quinoa", "Brown rice",
                "Sweet potatoes", "Broccoli", "Spinach", "Bell peppers", "Onions", "Garlic",
                "Avocado", "Olive oil", "Almonds", "Berries", "Bananas", "Apples",
                "Whole grain bread", "Salmon", "Tuna", "Lentils", "Chickpeas", "Tofu"
            ]
            
            # Get diet-specific ingredients
            if user_data["diet_type"] == "Vegetarian":
                diet_specific = ["Tofu", "Tempeh", "Lentils", "Beans", "Dairy products", "Eggs"]
            elif user_data["diet_type"] == "Vegan":
                diet_specific = ["Tofu", "Tempeh", "Lentils", "Beans", "Nutritional yeast", "Plant milk"]
            else:
                diet_specific = ["Chicken", "Turkey", "Lean beef", "Fish", "Shrimp"]
            
            # Add common and diet-specific ingredients
            grocery_items.update(common_ingredients)
            grocery_items.update(diet_specific)
            
            # Special handling for users who selected "None" for meat preferences
            if "meat_preferences" in user_data and "none" in [pref.lower() for pref in user_data["meat_preferences"]]:
                # Remove meat items from grocery list
                for item in ["Chicken breast", "Chicken", "Turkey", "Lean beef", "Fish", "Shrimp", "Salmon", "Tuna"]:
                    if item in grocery_items:
                        grocery_items.remove(item)
            
            # Display the grocery list
            st.markdown("### Your Grocery List")
            
            # Group by category
            categories = {
                "Proteins": ["Chicken", "Turkey", "Beef", "Fish", "Shrimp", "Tofu", "Tempeh", "Eggs", "Greek yogurt"],
                "Carbohydrates": ["Oats", "Quinoa", "Brown rice", "Sweet potatoes", "Whole grain bread"],
                "Fruits & Vegetables": ["Broccoli", "Spinach", "Bell peppers", "Onions", "Garlic", "Avocado", "Berries", "Bananas", "Apples"],
                "Fats & Oils": ["Olive oil", "Almonds", "Avocado"],
                "Other": ["Spices", "Herbs", "Condiments"]
            }
            
            for category, keywords in categories.items():
                st.markdown(f"**{category}**")
                category_items = [item for item in grocery_items if any(keyword.lower() in item.lower() for keyword in keywords)]
                
                if category_items:
                    for i, item in enumerate(sorted(category_items)):
                        # Make sure each checkbox has a unique key
                        st.checkbox(item, key=f"grocery_{category}_{i}_{item}")
                else:
                    st.markdown("*No items in this category*")
            
            # Add a section for custom items
            st.markdown("**Custom Items**")
            custom_item = st.text_input("Add custom item:")
            if custom_item:
                st.checkbox(custom_item, key=f"grocery_custom_{custom_item}")
    
    # Tab 5: Equipment - Note this is the 5th tab, not the 4th tab!
    with tab5:
        st.subheader("🛒 Recommended Equipment")
        
        if "equipment_recommendations" in plan and plan["equipment_recommendations"]:
            for i, equipment in enumerate(plan["equipment_recommendations"]):
                with st.expander(f"{equipment['name']} - ${equipment['price']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(equipment["image"], width=150)
                    
                    with col2:
                        st.markdown(f"**Description:** {equipment['description']}")
                        st.markdown(f"**Why it's recommended:** {equipment['reason']}")
                        if "alternatives" in equipment:
                            st.markdown(f"**Alternatives:** {equipment['alternatives']}")
        else:
            # Generate basic equipment list based on workouts
            all_equipment = set()
            for workout in plan.get("workouts", []):
                all_equipment.update(workout.get("equipment", []))
            
            # Remove "None" and "Bodyweight" from the count of equipment needed
            equipment_needed = [item for item in all_equipment if item not in ["None", "Bodyweight"]]
            
            if equipment_needed:
                st.markdown("Based on your workout plan, you'll need the following equipment:")
                for item in sorted(equipment_needed):
                    st.markdown(f"- {item}")
                    
                # Add purchase links (example)
                st.markdown("### Where to Buy")
                st.markdown("""
                - [Amazon Fitness Equipment](https://www.amazon.com/Fitness-Equipment/b?node=3407831)
                - [Target Sports Equipment](https://www.target.com/c/sports-equipment/-/N-5xt8r)
                - [Dick's Sporting Goods](https://www.dickssportinggoods.com/c/exercise-fitness-equipment)
                """)
            else:
                st.markdown("Your plan primarily uses bodyweight exercises - no special equipment needed!")
        
        # Budget option
        st.markdown("### Budget-Friendly Alternatives")
        st.markdown("""
        If you're on a budget, consider these alternatives:
        - Use filled water bottles as dumbbells
        - Use a sturdy chair for step-ups and dips
        - Use a towel as a yoga mat
        - Use a backpack filled with books for weighted exercises
        - Use household items like cans or bags of rice for lighter weights
        - Use a broomstick or PVC pipe for mobility work
        - Use the stairs in your home for cardio
        """)
        
        # Home gym setup guide
        st.markdown("### Home Gym Setup Guide")
        
        # Different recommendations based on space and budget
        space_options = ["Small Space (e.g., apartment)", "Medium Space (e.g., spare room)", "Large Space (e.g., garage/basement)"]
        budget_options = ["Low (under $100)", "Medium ($100-$500)", "High ($500+)"]
        
        col1, col2 = st.columns(2)
        with col1:
            space = st.selectbox("Available Space:", space_options)
        with col2:
            budget = st.selectbox("Budget:", budget_options)
        
        # Provide customized equipment recommendations
        st.markdown("#### Recommended Setup")
        
        if space == space_options[0]:  # Small space
            if budget == budget_options[0]:  # Low budget
                st.markdown("""
                - Resistance bands set ($15-30)
                - Exercise mat ($15-25)
                - Door-mounted pull-up bar ($25-40)
                - Jump rope ($10-15)
                """)
            elif budget == budget_options[1]:  # Medium budget
                st.markdown("""
                - Adjustable dumbbells ($100-200)
                - Exercise mat ($15-25)
                - Door-mounted pull-up bar ($25-40)
                - Resistance bands set ($15-30)
                - Suspension trainer ($30-80)
                - Foldable bench ($50-100)
                """)
            else:  # High budget
                st.markdown("""
                - Premium adjustable dumbbells ($300-400)
                - Foldable squat rack ($300-500)
                - High-quality exercise mat ($30-50)
                - Smart fitness mirror or display ($1,000+)
                - Kettlebell set ($100-200)
                - Foldable bench ($100-150)
                """)
        elif space == space_options[1]:  # Medium space
            if budget == budget_options[0]:  # Low budget
                st.markdown("""
                - Resistance bands set ($15-30)
                - Exercise mat ($15-25)
                - Doorway pull-up bar ($25-40)
                - Jump rope ($10-15)
                - Kettlebell (1-2) ($20-40)
                - Stability ball ($15-25)
                """)
            elif budget == budget_options[1]:  # Medium budget
                st.markdown("""
                - Adjustable dumbbells ($100-200)
                - Kettlebell set ($70-150)
                - Exercise mat ($15-25)
                - Pull-up bar ($25-40)
                - Resistance bands set ($15-30)
                - Adjustable bench ($80-150)
                - Jump rope ($10-15)
                """)
            else:  # High budget
                st.markdown("""
                - Premium adjustable dumbbells ($300-400)
                - Half rack ($300-700)
                - Barbell and weight plates ($200-400)
                - High-quality adjustable bench ($150-300)
                - Exercise mat ($30-50)
                - Kettlebell set ($100-200)
                - Cable machine ($300-700)
                """)
        else:  # Large space
            if budget == budget_options[0]:  # Low budget
                st.markdown("""
                - Resistance bands set ($15-30)
                - Exercise mat ($15-25)
                - Pull-up bar ($25-40)
                - Jump rope ($10-15)
                - DIY plyo box ($20-30 in materials)
                - Kettlebell (1-2) ($20-40)
                """)
            elif budget == budget_options[1]:  # Medium budget
                st.markdown("""
                - Power tower ($100-200)
                - Adjustable dumbbells ($100-200)
                - Kettlebell set ($70-150)
                - Flat bench ($50-100)
                - Exercise mat ($15-25)
                - Resistance bands set ($15-30)
                - Jump rope ($10-15)
                - Stability ball ($15-25)
                """)
            else:  # High budget
                st.markdown("""
                - Power rack ($300-800)
                - Olympic barbell and weight plates ($300-600)
                - Adjustable bench ($150-300)
                - Premium adjustable dumbbells ($300-400)
                - Kettlebell set ($100-200)
                - Cardio equipment (treadmill/bike/rower) ($300-1,000)
                - Flooring ($100-300)
                - Cable machine or functional trainer ($700-2,000)
                """)
