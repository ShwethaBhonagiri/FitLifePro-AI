# FitLife Pro AI+

## Overview
FitLife Pro AI+ is an AI-powered fitness and nutrition planning system that personalizes workouts and meals based on user profiles. The system uses clustering, regression, reinforcement learning, and adaptive feedback mechanisms to generate tailored fitness journeys through a user-friendly Streamlit interface.

## Key Features
- Fitness level classification using health and demographic metrics
- Personalized workout recommendations based on goals, equipment, and experience
- Adaptive meal plan generation aligned with TDEE and macros
- Progress prediction using hybrid AI logic
- Interactive UI with six-step onboarding

## File Structure
```
FitLife_Pro/
├── main.py               # Streamlit UI with onboarding wizard
├── planner.py            # Core AI logic (classification, planning, adaptation)
├── feedback.py           # Feedback analysis and reinforcement learning logic
├── display.py            # Visual rendering and tabbed workout/nutrition display
├── data/                 # Contains JSON files for meals, workouts, and user profiles
├── assets/               # Optional image or CSS assets
├── tests/                # Includes test cases for planner and feedback modules
└── requirements.txt      # Python dependencies
```

## Setup Instructions
1. **Clone the repository**  
   ```bash
   git clone https://github.com/ShwethaBhonagiri/FitLifePro-AI
   cd FitLife_Pro
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**  
   ```bash
   streamlit run main.py
   ```

## Basic System Flow
1. User enters basic data through a six-step onboarding wizard.
2. AI modules classify user fitness level and generate recommendations.
3. System adapts suggestions based on simulated feedback and progress predictors.
4. Personalized plans are rendered dynamically in the interface.

## Testing & Validation
- **Manual testing** was performed using diverse mock user profiles to validate personalized output.
- Modules were tested independently (`planner.py`, `feedback.py`) with both valid and edge-case inputs.
- Sample test cases and data can be found in the `/tests/` folder.

## License
This project is developed for academic use under UTS guidelines.


---

> **Thank you for using FitLife Pro AI+**  
We hope this intelligent planner helped you explore how AI can enhance personalized fitness and nutrition design.

If you encounter any issues or have suggestions for improvement, please feel free to open an issue.  
Your feedback helps us improve and evolve this application further.

---
