import os
from groq import Groq
from tavily import TavilyClient

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

MODEL_NAME = "llama-3.3-70b-versatile"


def extract_location(user_query):
    prompt = f"""
    Extract ONLY the city name from the user query.
    If no city found, return NONE.
    Query: {user_query}
    """

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def search_tata_dealers(city):
    search_query = f"""
    Tata Motors authorized car dealership showroom in {city}
    Include address and contact number.
    """

    results = tavily_client.search(
        query=search_query,
        search_depth="advanced",
        max_results=5
    )

    return results.get("results", [])


def tata_agent(user_query):

    city = extract_location(user_query)

    if city == "NONE":
        return "Please tell me your city so I can find the nearest Tata Motors dealership."

    dealers = search_tata_dealers(city)

    if not dealers:
        return f"I couldn't find Tata Motors dealerships in {city}."

    formatting_prompt = f"""
    Format the following search results into a clean chatbot answer.
    Show dealer name, address, phone and website.
    Results:
    {dealers}
    """

    final_response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": formatting_prompt}],
        temperature=0.3
    )

    return final_response.choices[0].message.content