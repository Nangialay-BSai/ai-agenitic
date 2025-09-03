def planner_prompt(user_prompt: str) -> str:
    PLANNER_PROMPT = f"""
you are the planner agend. Convert the user prompt into a complete enginnering project paln

user request: 
{user_prompt}


"""
    return PLANNER_PROMPT




