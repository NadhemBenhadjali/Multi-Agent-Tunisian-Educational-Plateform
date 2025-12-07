from app.crew.planner_crew import PlannerCrew
from app.handlers import get_parent_choices

def run():
    inputs = get_parent_choices()  
    result = PlannerCrew().crew().kickoff(inputs=inputs)
    print(result)
if __name__ == "__main__":
    run()
