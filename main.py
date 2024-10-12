import streamlit as st
import subprocess
import os

# Dictionary mapping choice numbers to script file names
league_scripts = {
    "Belgium 1": "Belgium1.py",
    "Germany 1": "germany1.py",
    "Germany 2": "germany2.py",
    "Italy A": "italyA.py",
    "Italy B": "italyB.py",
    "Turkey 1": "Turkey1.py",
    "Spain 1": "Spain1.py",
    "Spain 2": "Spain2.py",
    "Netherlands Eredivisie": "NetherlandsEredivisie.py",
    "France 1": "France1.py",
    "France 2": "France2.py",
    "Greece 1": "Greece1.py",
    "England League 1": "EnglandLeague1.py",
    "England Championship": "EnglandChampionship.py",
    "England League 2": "EnglandLeague2.py"
}

def choose_league():
    # Streamlit dropdown for league selection
    st.title("Choose a League")
    league_choice = st.selectbox("Select a league to load:", list(league_scripts.keys()))
    
    # Run button to execute the chosen script
    if st.button("Run Script"):
        # Get the current directory (adjust if necessary for your hosting)
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        
        # Get the script file corresponding to the selected league
        script_path = os.path.join(base_dir, league_scripts[league_choice])
        
        # Display the selected script and running message
        st.write(f"Running script for {league_choice}...")

        try:
            # Run the script using subprocess
            subprocess.run(["streamlit", "run", script_path], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred while trying to run the script: {e}")

# Streamlit runs this function when the app starts
choose_league()
