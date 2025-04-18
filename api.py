import pandas as pd
import json
import boto3
import asyncio
import os
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, ModelSettings
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path
import io
from sentence_transformers import SentenceTransformer
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from sklearn.cluster import DBSCAN

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

s3_file_path = 'sysb_records.csv'
bucket = 'zenstatement'
object_name='not-found-sys-b'

base_dir = os.getcwd()
recon_file_name = 'recon_data_raw.csv'
recon_reply_name = 'recon_data_reply.csv'
data_dir = os.path.join(base_dir,'data')
input_dir = os.path.join(data_dir,'input')

directories_to_create = [data_dir,input_dir]
for directory in directories_to_create:
    Path(directory).mkdir(parents=True, exist_ok=True)

recon_file_path = os.path.join(input_dir,recon_file_name)
recon_reply_path = os.path.join(input_dir,recon_reply_name)



s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id='',
    aws_secret_access_key=''
)

os.environ["OPENAI_API_KEY"] = ""

keep_cols = ['txn_ref_id','sys_a_date','sys_a_amount_attribute_1','recon_sub_status']

missing_values_dict = {
     "sys_a_amount_attribute_2" : 0,
     "sys_b_amount_attribute_1" : 0,
     "sys_b_amount_attribute_2" : 0,
     "payment_method" : 'NA',
     "sys_b_date" : 'NA',
}

resolution_prompt = (
    "You are a financial resolution classifier agent."
    "Based on the input provided to you classify the resolution for user's query is True or False. True meaning it is resolved, False means unresolved"
    "If resolution is False or unresolved, tell what is needed to resolve it"
    "If resolution is True or resolved, put next steps as 'NA'"
)

class ResolutionResult(BaseModel):
    resolved: bool
    """Whether the output is resolved or not"""

    next_steps: str
    """If resolved is False, explain what is to be done next."""

agent_resolution = Agent(    
    name="resolution_agent",
    instructions=resolution_prompt,
    model="gpt-4o-mini",
    model_settings = ModelSettings(temperature=0.1),
    output_type=ResolutionResult
)

class PreProcess(BaseModel):
    recon_data_raw: str

class Resolution(BaseModel):
    recon_data_reply: str

app = FastAPI()

@app.get("/api/v1/zen/health")
async def health():
    return {"result": "API is UP!"}


@app.post("/api/v1/zen/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(input_dir,file.filename)
        with open(file_location, "wb") as f:
            contents = await file.read()
            f.write(contents)
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        print(str(e))
        return JSONResponse(status_code=500, content={"message": f"Failed to upload file: {str(e)}"})

@app.post("/api/v1/zen/preprocess")
async def preprocess_data(preprocess: PreProcess):
    try:
        df1 = pd.read_csv(os.path.join(input_dir,preprocess.recon_data_raw))
        for key,val in missing_values_dict.items():
            df1[key] = df1[key].fillna(value=val)
        def check_sysb(recon):
            rec = json.loads(recon)
            if rec['amount'].lower() == 'Not Found-SysB'.lower():
                return 1
            else:
                return 0
        df1['SYSB'] = df1['recon_sub_status'].apply(check_sysb)
        df1 = df1[df1['SYSB']==1]
        df1 = df1[keep_cols]
        csv_buffer = io.StringIO()
        df1.to_csv(csv_buffer)
        df1.to_csv(os.path.join(input_dir,'sysb_records.csv'),index=False)
        s3_object = s3.Object(bucket, f'not-found-sys-b/{s3_file_path}')
        s3_object.put(Body=csv_buffer.getvalue())
        return {"uploaded_file": f"s3://{bucket}/{object_name}/sysb_records.csv","success":True,"local_df_path": str(os.path.join(input_dir,'sysb_records.csv'))}
    except Exception as e:
        print(str(e))
        return JSONResponse(status_code=500, content={"message": f"Failed to upload file: {str(e)}"})

@app.post("/api/v1/zen/resolve")
async def resolve(res: Resolution):
    try:
        df1 = pd.read_csv(os.path.join(input_dir,'sysb_records.csv'))
        df2 = pd.read_csv(os.path.join(input_dir,res.recon_data_reply))
        df_concat = pd.concat([df1.set_index('txn_ref_id'),df2[['Transaction ID','Comments']].set_index('Transaction ID')], axis=1, join='inner').reset_index()
        df_concat.rename({"index":"Transaction ID"},axis=1,inplace=True)
        status_list = []
        def resolution_handling_direct(row,status_list):
            transaction_id = row['Transaction ID']
            comment = row['Comments']
            result = Runner.run_sync(agent_resolution, input=str(comment))
            res_status = result.final_output.resolved
            status_list.append(res_status)
            file_content = ''
            file_content+=f"Transaction ID : {transaction_id}\n"
            file_content+=f"sys_a_date : {str(row['sys_a_date'])}\n"
            file_content+=f"sys_a_amount_attribute_1 : {str(row['sys_a_amount_attribute_1'])}\n"
            file_content+=f"Comments : {str(comment)}\n"
            if res_status:
                #print(file_content)
                file_content+=f'Status : Resolved\n'
                s3_object = s3.Object(bucket, f'resolved/{transaction_id}.txt')
                s3_object.put(Body=file_content)
            else:
                #print(file_content)
                file_content+=f'Status : Unresolved\n'
                file_content+=f'Next Steps : {str(result.final_output.next_steps)}\n'
                s3_object = s3.Object(bucket, f'unresolved/{transaction_id}.txt')
                s3_object.put(Body=file_content)
        df_concat.apply(resolution_handling_direct,status_list=status_list,axis=1)
        df_concat['Resolved'] = status_list
        df_resolved = df_concat[df_concat['Resolved']==True]
        df_resolved.to_csv(os.path.join(input_dir,'resolved.csv'),index=False)
        return {"total_cases": len(df2),"resolved_cases": len(df_resolved),"unresolved_cases": len(df_concat)-len(df_resolved),"success":True,"local_df_path": str(os.path.join(input_dir,'resolved.csv'))}
    except Exception as e:
        print(str(e))
        return JSONResponse(status_code=500, content={"message": f"Failed to resolve: {str(e)}"})

@app.get("/api/v1/zen/cluster")
async def resolve():
    try:
        df_resolved = pd.read_csv(os.path.join(input_dir,'resolved.csv'))
        embeddings = model.encode(list(df_resolved['Comments']))
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        dbscan.fit(embeddings)

        # Get the cluster assignments
        labels_dbscan = dbscan.labels_
        df_resolved['cluster_dbscan'] = labels_dbscan
        df_cluster = df_resolved[['cluster_dbscan','Comments']].groupby('cluster_dbscan').agg(list).reset_index()
        text = []
        def get_cluster(row,text):
            if int(row['cluster_dbscan']) > 0:
                text.append(f'Cluster ID : {str(row["cluster_dbscan"])}\n')
                for i in row['Comments']:
                    text.append(f'{i}\n')
                text.append('\n')

        df_cluster.apply(get_cluster,text=text,axis=1)
        with open(os.path.join(input_dir,f'cluster_info.txt'),'w') as f:
            f.write(''.join(text))
        s3_object = s3.Object(bucket, f'cluster_info.txt')
        s3_object.put(Body=''.join(text))
        return {"uploaded_file": f"s3://{bucket}/cluster_info.txt","success":True}
    except Exception as e:
        print(str(e))
        return JSONResponse(status_code=500, content={"message": f"Failed to cluster: {str(e)}"})



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)