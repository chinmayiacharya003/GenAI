
from clarifai.client.workflow import Workflow

# Your PAT (Personal Access Token) can be found in the Account's Security section

# Initialize the workflow_url
workflow_url = "https://clarifai.com/ibl1wl2pz6pc/my-first-application-63en5/workflows/workflow-1f579e"
text_classfication_workflow = Workflow(
    url= workflow_url , pat="c463c82744504ce5a2f9955110b2c5fb"
)
result = text_classfication_workflow.predict_by_bytes(b"Who is your Daddy?", input_type="text")
print(result.results[0].outputs[0].data)
