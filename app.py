from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests

app = Flask(__name__)

# Load the content-based recommendations CSV
# Ensure the file exists and has the correct structure
content_df = pd.read_csv("content_recommendations.csv")

# Collaborative model: Dummy data for testing
def collaborative_model(user_id):
    """
    Returns mock recommendations for the collaborative model.
    """
    return [f"Item_{i}" for i in range(1, 6)]  # Waiting for CSV

# Azure ML Endpoint configuration
azure_endpoint = "http://d0e2662a-1035-40ed-ac8a-56af67503193.eastus2.azurecontainer.io/score"
api_key = "ZYIIAALV0Pm0LBtZyZk9BSuWfAJvxqIg"

def get_azure_recommendations(user_id):
    """
    Calls the Azure ML endpoint to get recommendations.
    """
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    data = {
        "Inputs": {
            "input1": [
                {"userID": str(user_id)}  # Adjust this to match the Azure schema
            ]
        }
    }
    try:
        response = requests.post(azure_endpoint, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('recommendations', [])
        else:
            print(f"Azure endpoint error: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Error calling Azure endpoint: {e}")
        return []

@app.route('/')
def home():
    """
    Renders the home page with the input form.
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handles the recommendation request and returns recommendations
    from the collaborative model, content-based model, and Azure ML.
    """
    user_id = request.json.get('user_id')  # Get the user ID from the request payload

    # Collaborative model recommendations (dummy data)
    collaborative_recommendations = collaborative_model(user_id)

    # Content-based recommendations from the CSV file
    try:
        # Ensure 'contentId' is treated as an integer
        content_df['contentId'] = content_df['contentId'].astype(int)

        # Filter the DataFrame for the given user_id
        filtered_df = content_df[content_df['contentId'] == int(user_id)]
        if not filtered_df.empty:
            # Extract the top 5 columns with the highest values for the given contentId
            content_recommendations = (
                filtered_df.iloc[0, 1:]  # Exclude 'contentId' column
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()  # Get the column names as recommendations
            )
        else:
            content_recommendations = []  # No match found
    except Exception as e:
        print(f"Error in content model: {e}")
        content_recommendations = []

    # Azure ML recommendations
    azure_recommendations = get_azure_recommendations(user_id)

    # Return all recommendations as a JSON response
    return jsonify({
        "collaborative": collaborative_recommendations,
        "content": content_recommendations,
        "azure": azure_recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
