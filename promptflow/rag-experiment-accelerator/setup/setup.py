import os
import requests
import subprocess

from dotenv import load_dotenv

load_dotenv()

location = "westus2"
connection_name="rag_connection"
subscription_id = os.environ["AML_SUBSCRIPTION_ID"]
rg_name = os.environ["AML_RESOURCE_GROUP_NAME"]
workspace_name = os.environ["AML_WORKSPACE_NAME"]

access_token = subprocess.check_output("az account get-access-token | jq -r '.accessToken'", shell=True).decode("utf-8").rstrip()

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}

data = {
    "connectionType": "Custom",
    "customConfigs": {
        "AZURE_SEARCH_SERVICE_ENDPOINT": {
            "configValueType": "String",
            "value": os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
        },
        "OPENAI_API_TYPE": {
            "configValueType": "String",
            "value": os.environ["OPENAI_API_TYPE"]
        },
        "OPENAI_ENDPOINT": {
            "configValueType": "String",
            "value": os.environ["OPENAI_ENDPOINT"]
        },
        "OPENAI_API_VERSION": {
            "configValueType": "String",
            "value": os.environ["OPENAI_API_VERSION"]
        },
        "AZURE_SEARCH_ADMIN_KEY": {
            "configValueType": "Secret",
            "value": os.environ["AZURE_SEARCH_ADMIN_KEY"]
        },
        "OPENAI_API_KEY": {
            "configValueType": "Secret",
            "value": os.environ["OPENAI_API_KEY"]
        },
        "AZURE_SEARCH_ADMIN_KEY": {
            "configValueType": "Secret",
            "value": os.environ["AZURE_SEARCH_ADMIN_KEY"]
        },
        "AML_SUBSCRIPTION_ID": {
            "configValueType": "Secret",
            "value": os.environ["AML_SUBSCRIPTION_ID"]
        },
        "AML_WORKSPACE_NAME": {
            "configValueType": "Secret",
            "value": os.environ["AML_WORKSPACE_NAME"]
        },
        "AML_RESOURCE_GROUP_NAME": {
            "configValueType": "Secret",
            "value": os.environ["AML_RESOURCE_GROUP_NAME"]
        }
    },
}

connection_url_post=f"https://ml.azure.com/api/{location}/flow/api/subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/connections/{connection_name}?asyncCall=true"

response = requests.post(url=connection_url_post, headers=headers, json = data)

print(response.text)
