import gradio as gr
import httpx
import pandas as pd
import io

API_BASE = "http://127.0.0.1:8000/api/v1/zen"

# ---------- Page 1: Upload CSV and show preview ----------
def upload_csv(file):
    try:
        if file is None:
            return "Please select a file to upload", None

        # Read the file content
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f.read())}

        # Send file content to upload API
        response = httpx.post(f"{API_BASE}/upload", files=files)
        response.raise_for_status()

        # Preview the file in a DataFrame
        df = pd.read_csv(file.name)
        return df, file.name.split('\\')[-1]  # Return the filename for later use

    except pd.errors.EmptyDataError:
        return "The uploaded CSV file is empty", None
    except pd.errors.ParserError:
        return "Invalid CSV format", None
    except httpx.HTTPError as e:
        return f"API error: {str(e)}", None
    except Exception as e:
        return f"Upload failed: {str(e)}", None

# ---------- Page 2: Preprocess ----------
def preprocess_data(raw_filename):
    try:
        payload = {
            "recon_data_raw": raw_filename
        }
        response = httpx.post(f"{API_BASE}/preprocess", json=payload)
        response.raise_for_status()
        
        # Read the preprocessed file for preview
        try:
            local_df=response.json()['local_df_path']
            df = pd.read_csv(local_df)
            return "Preprocessing completed. Uploaded to S3.", df
        except Exception as e:
            return f"File preview failed: {str(e)}", None
            
    except Exception as e:
        return f"Preprocessing failed: {str(e)}", None

# ---------- Page 3: Upload and Resolve Queries ----------
def upload_and_resolve(file):
    try:
        files = {"file": (file.name, file)}
        upload_response = httpx.post(f"{API_BASE}/upload", files=files,timeout=60)
        upload_response.raise_for_status()
        
        payload = {
            "recon_data_reply": file.name
        }
        resolve_response = httpx.post(f"{API_BASE}/resolve", json=payload,timeout=180)
        resolve_response.raise_for_status()
        return f"Query resolution done for {file.name}",resolve_response.json()['local_df_path']
    except Exception as e:
        return f"Resolve failed: {str(e)}"

# ---------- Page 4: Clustering ----------
def cluster_queries():
    try:
        response = httpx.get(f"{API_BASE}/cluster")
        response.raise_for_status()
        return "Clustering completed successfully."
    except Exception as e:
        return f"Clustering failed: {str(e)}"

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    uploaded_filename = gr.State(value="")
    is_preprocessed = gr.State(value=False)  # Track preprocessing state
    with gr.Tabs():
        
        # -------- Page 1 --------
        with gr.Tab("1. Upload CSV"):
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            output_df = gr.Dataframe(label="CSV Preview")
            file_input.change(
                fn=upload_csv,
                inputs=file_input,
                outputs=[output_df, uploaded_filename]
            )

        # -------- Page 2 --------
        with gr.Tab("2. Preprocess Data and upload to S3"):
            with gr.Row():
                raw_name = gr.Text(label="Raw CSV filename (e.g. recon_data_raw.csv)")
            preprocess_btn = gr.Button("Preprocess")
            preprocess_output = gr.Textbox()
            preview_df = gr.Dataframe(label="Preprocessed Data Preview")
            preprocess_btn.click(
                preprocess_data, 
                inputs=raw_name, 
                outputs=[preprocess_output, preview_df]
            )

        # -------- Page 3 --------
        with gr.Tab("3. Resolve Queries"):
            resolve_file = gr.File(label="Upload reply CSV", file_types=[".csv"])
            resolve_output = gr.Textbox()
            preview_df = gr.Dataframe(label="Resolved Data Preview")
            resolve_file.change(upload_and_resolve, inputs=resolve_file, outputs=[resolve_output,preview_df])

        # -------- Page 4 --------
        with gr.Tab("4. Cluster Queries"):
            cluster_btn = gr.Button("Cluster Resolved Queries")
            cluster_output = gr.Textbox()
            cluster_btn.click(cluster_queries, outputs=cluster_output)

demo.launch()