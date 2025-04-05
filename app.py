from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load content-based recommendations
content_df = pd.read_csv("content_recommendations.csv")
cosine_sim = linear_kernel(content_df, content_df)
df_results = pd.DataFrame(cosine_sim, columns=content_df['contentId'], index=content_df['contentId'])

# Load collaborative filtering recommendations
collab_df = pd.read_csv('collab_recommendations.csv', index_col=0)


# Fixed user ID for Azure model
FIXED_USER_ID = "-9222795471790223670"

# Collaborative filtering function
def collaborative_model(item_id):
    try:
        item_id = int(item_id)  # Ensure item_id is an integer
        if item_id in collab_df.index:
            row = collab_df.loc[item_id]
            return row.iloc[1:6].tolist()  # Assuming the 1st column is title, then recs 1–5
        else:
            return ['Not found']
    except Exception as e:
        print(f"Error in collaborative model: {e}")
        return []

# Content-based filtering function
def content_model(item_id):
    try:
        item_id = int(item_id)  # Make sure it's an integer
        if item_id in df_results.columns:
            # Get top 5 similar items, excluding the item itself
            return df_results[item_id].sort_values(ascending=False).head(6)[1:].index.tolist()
        else:
            return ['Not found']
    except Exception as e:
        print(f"Error in content model: {e}")
        return []


# Azure ML function
def get_azure_recommendations(user_id):
    azure_endpoint = "http://d0e2662a-1035-40ed-ac8a-56af67503193.eastus2.azurecontainer.io/score"
    api_key = "ZYIIAALV0Pm0LBtZyZk9BSuWfAJvxqIg"

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    data = {
        "Inputs": {
            "input1": [
                {"userID": str(user_id)}
            ]
        }
    }

    try:
        response = requests.post(azure_endpoint, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('recommendations', [])[:5]
        else:
            print(f"Azure endpoint error: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        print(f"Error calling Azure endpoint: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    item_id = request.json.get("item_id")

    collab = collaborative_model(item_id)
    content = content_model(item_id)
    azure = get_azure_recommendations(FIXED_USER_ID)

    return jsonify({
        "collaborative": collab,
        "content": content,
        "azure": azure
    })

if __name__ == '__main__':
    app.run(debug=True)
