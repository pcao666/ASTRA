import os
import chromadb
from sentence_transformers import SentenceTransformer  # Fallback: bge-m3 本地模型
from typing import List, Dict, Any, Tuple
import sys
from mcp.server.fastmcp import FastMCP
import io
import time
import torch
import traceback
import logging

from multiprocessing import Process
import concurrent.futures
import threading
import requests

mcp = FastMCP("LOCAL", timeout=6000, request_timeout=6000, execution_timeout=6000)

# --- Database Connection (Assumes DB is already built) ---
db_path = "./database"  # Your Database data path
collection_name = "my_collection"

# --- Embedding Config ---
# SiliconFlow API 优先，失败则 fallback 到本地 bge-m3
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"

# --- Lazy Loading ---
_rag_model = None  # 本地 bge-m3 model (仅 fallback 时加载)
_chroma_client = None
_collection = None
print("ChromaDB client will be loaded on first use. Embeddings: SiliconFlow API (priority) -> local bge-m3 (fallback).")


def get_db_collection():
    """
    Initializes and returns the ChromaDB client and collection on first call.
    """
    global _chroma_client, _collection
    if _collection is None:
        print("Connecting to ChromaDB for the first time...")
        try:
            _chroma_client = chromadb.PersistentClient(path=db_path)
            _collection = _chroma_client.get_collection(name=collection_name)
            print(f"✅ Successfully connected to collection '{collection_name}' with {_collection.count()} documents.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not connect to ChromaDB collection.")
            print(f"Please run 'build_database.py' script first.")
            print(f"Details: {e}")
            # Raise an exception to stop the tool execution
            raise e
    return _collection


# --- MCP Tools ---

@mcp.tool()
async def rag_query(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Queries the analog circuit design knowledge base. The knowledge base contains:
    - gm/Id design tables and methodology for analog OTA sizing
    - Compensation network design rules (Miller, Ahuja, etc.)
    - Layout-aware design considerations and parasitic effects
    - PVT corner analysis methodology

    Call this tool BEFORE making sizing decisions when you need:
    - Recommended gm/Id values for specific transistor roles (input pair, current mirror, etc.)
    - Heuristics for choosing initial L/W ranges
    - Compensation strategy selection guidance
    - Knowledge of process-specific design constraints

    Args:
        query: Natural language question about analog circuit design
        num_results: Number of relevant document chunks to return (default 3)

    Returns:
        Dict with 'results' list, each containing 'content' (document text)
    """
    try:
        collection = get_db_collection()
    except Exception as e:
        return {"results": [{"content": f"Error: Could not connect to DB. {e}"}]}

    print(f"Received query: {query}")

    # --- Embedding: SiliconFlow API 优先，失败 fallback 本地 bge-m3 ---
    query_embedding = None

    # 1) Try SiliconFlow API
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("SILICONFLOW_API_KEY", "")
        if api_key:
            resp = requests.post(
                SILICONFLOW_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": SILICONFLOW_EMBEDDING_MODEL,
                    "input": [query],
                    "encoding_format": "float",
                },
                timeout=30,
            )
            result = resp.json()
            if "data" in result and len(result["data"]) > 0:
                query_embedding = result["data"][0]["embedding"]
                print("Embedding via SiliconFlow API (Qwen3-VL-Embedding-8B).")
            else:
                print(f"SiliconFlow API returned no data, falling back to local model.")
        else:
            print("SILICONFLOW_API_KEY not set, falling back to local model.")
    except Exception as e:
        print(f"SiliconFlow API failed ({e}), falling back to local model.")

    # 2) Fallback: local bge-m3
    if query_embedding is None:
        global _rag_model
        if _rag_model is None:
            print(f"Loading local embedding model ({LOCAL_EMBEDDING_MODEL})...")
            try:
                _rag_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
                print("Local model loaded.")
            except Exception as e:
                return {"results": [{"content": f"Error: Could not load local embedding model. {e}"}]}
        query_embedding = _rag_model.encode([query], normalize_embeddings=True)[0].tolist()
        print("Embedding via local bge-m3 (fallback).")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )

    # Organize query results
    response = {
        "results": []
    }

    # Check if results are valid and not empty
    if results and results.get('documents') and results['documents'][0]:
        for doc in results['documents'][0]:
            response["results"].append({
                "content": doc
            })
    else:
        print("No documents found for the query.")

    return response





# --- find_initial_design Background Task ---

def _run_find_initial_design_task(
        gmid1: int,
        gmid2: int,
        gmid3: int,
        gmid4: int,
        gmid5: int,
        iterations: int,
        task_id: str,
        output_filename: str  # .log file path
):
    """
    A standalone function to execute the find_initial_design task in the background.
    This function will run in a new process and includes detailed debugging logs.
    """
    logger = logging.getLogger(task_id)
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers due to retries
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(output_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        # Lazy loading to prevent long startup time
        from Find_Initial_Design.bo_logic import BayesianOptimization

        logger.info(f"--- Background Task 'find_initial_design' Started: {time.ctime()} ---")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"GMID Parameters: gmid1={gmid1}, gmid2={gmid2}, gmid3={gmid3}, gmid4={gmid4}, gmid5={gmid5}")
        logger.info(f"Max optimization iterations: {iterations} (plus 10 initial samples)")
        logger.info("\n")

        try:
            SEED = 5
            logger.info(f"Using Seed set to: {SEED}")

            store_path = "./store"
            os.makedirs(store_path, exist_ok=True)
            file_path_x = os.path.join(store_path, f"design_{task_id}_SEED_{SEED}_x.csv")
            file_path_y = os.path.join(store_path, f"design_{task_id}_SEED_{SEED}_y.csv")

            logger.info(f"Result X will be saved to: {os.path.abspath(file_path_x)}")
            logger.info(f"Result Y will be saved to: {os.path.abspath(file_path_y)}")

            mace = BayesianOptimization(iterations)

            start_time = time.time()
            logger.info(f"Optimization start time: {time.ctime(start_time)}")

            # Execute find, passing the logger
            resultx, resulty = mace.find(
                gmid1, gmid2, gmid3, gmid4, gmid5,
                file_path_x, file_path_y,
                logger
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"\n--- Optimization Complete ---")
            logger.info(f"Optimization end time: {time.ctime(end_time)}")
            logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

            # Print final results to log
            mace.print_results(resultx, resulty, logger)

            logger.info(f"Streaming CSV files saved.")

        except Exception as e:
            logger.error(f"\n--- Task 'find_initial_design' Terminated Unexpectedly: {time.ctime()} ---")
            logger.error("A fatal error caused the process to crash. Full traceback:")
            logger.error(traceback.format_exc())

    except Exception as e:
        error_msg = f"Could not set up log file '{output_filename}' or failed to import 'bo_logic', 'find_initial_design' task aborted. Error: {e}"
        print(error_msg)
        try:
            # Try a final log write
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        except:
            pass

    print(f"Background task 'find_initial_design' finished, logs written to: {os.path.abspath(output_filename)}")
    # Close logger handlers
    file_handler.close()
    logger.removeHandler(file_handler)


# --- FocalOpt Background Task ---

def _run_focal_opt_task(
        initial_design_task_id: str,
        iterations: int,
        task_id: str,
        output_filename: str  # .log file path
):
    """
    A standalone function to execute the FocalOpt (Stage 2 optimization) task in the background.
    This function will run in a new process and includes detailed debugging logs.
    """
    # --- LAZY IMPORT ---
    # Import simulation functions only when the task runs
    from examples.simulation_OTA_two import OTA_two_simulation_all, write_data_OTA_two_all

    logger = logging.getLogger(task_id)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(output_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        # Lazy load FocalOpt logic
        from FocalOpt.focal_opt_main import run_focal_optimization

        logger.info(f"--- Background Task 'FocalOpt' (Stage 2 Optimization) Started: {time.ctime()} ---")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Using Initial Design ID: {initial_design_task_id}")
        logger.info(f"Max optimization iterations: {iterations} (will be distributed within FocalOpt stages)")
        logger.info("\n")

        try:
            SEED = 5  # Assume FocalOpt internally uses SEED 5

            store_path = "./store"
            os.makedirs(store_path, exist_ok=True)

            # Construct file paths for the Stage 1 output
            initial_x_csv_path = os.path.join(store_path, f"design_{initial_design_task_id}_SEED_{SEED}_x.csv")
            initial_y_csv_path = os.path.join(store_path, f"design_{initial_design_task_id}_SEED_{SEED}_y.csv")

            if not os.path.exists(initial_y_csv_path):
                logger.error(f"FATAL: Initial design Y CSV file not found: {initial_y_csv_path}")
                raise FileNotFoundError(f"Initial design file not found: {initial_y_csv_path}")

            if not os.path.exists(initial_x_csv_path):
                logger.error(f"FATAL: Initial design X CSV file not found: {initial_x_csv_path}")
                raise FileNotFoundError(f"Initial design file not found: {initial_x_csv_path}")

            start_time = time.time()
            logger.info(f"FocalOpt optimization start time: {time.ctime(start_time)}")

            # Main FocalOpt call
            final_csv_path, final_best_result, final_best_x = run_focal_optimization(
                initial_x_csv_path,
                initial_y_csv_path,
                OTA_two_simulation_all,  # Pass the full unbinding simulation function
                task_id,
                logger,
                total_iterations=iterations
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"\n--- FocalOpt Optimization Complete ---")
            logger.info(f"Optimization end time: {time.ctime(end_time)}")
            logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"Final results saved in: {os.path.abspath(final_csv_path)}")
            logger.info(f"Best performance metrics: {final_best_result}")
            logger.info(f"Best X parameters: {final_best_x}")

            # --- Generate final netlist from best parameters ---
            if final_best_x and len(final_best_x) == 12:
                param_names = ['cap', 'L1', 'L2', 'L3', 'L4', 'L5', 'r', 'W1', 'W2', 'W3', 'W4', 'W5']
                x_dict = dict(zip(param_names, final_best_x))

                netlist_template = os.path.join("examples", "netlists", "ICCAD_OTA_two_new.cir")
                netlist_output = os.path.join("netlist_working", f"focalopt_{task_id}_final.cir")
                os.makedirs("netlist_working", exist_ok=True)

                write_data_OTA_two_all(
                    netlist_template,
                    cap=x_dict['cap'], l1=x_dict['L1'], l2=x_dict['L2'],
                    l3=x_dict['L3'], l4=x_dict['L4'], l5=x_dict['L5'],
                    r=x_dict['r'], w1=x_dict['W1'], w2=x_dict['W2'],
                    w3=x_dict['W3'], w4=x_dict['W4'], w5=x_dict['W5']
                )

                # write_data_OTA_two_all writes to a fixed temp path; copy to our output path
                import shutil
                temp_netlist = os.path.join("examples", "temp_retest_OTA_two_all_netlist_new.cir")
                if os.path.exists(temp_netlist):
                    shutil.copy2(temp_netlist, netlist_output)
                    logger.info(f"Final netlist saved to: {os.path.abspath(netlist_output)}")
                else:
                    logger.warning("Temp netlist not found after write_data_OTA_two_all call.")
            else:
                logger.warning(f"Could not generate netlist: final_best_x invalid (len={len(final_best_x) if final_best_x else 0})")


        except Exception as e:
            logger.error(f"\n--- Task 'FocalOpt' Terminated Unexpectedly: {time.ctime()} ---")
            logger.error("A fatal error caused the process to crash. Full traceback:")
            logger.error(traceback.format_exc())

    except Exception as e:
        error_msg = f"Could not set up log file '{output_filename}' or failed to import 'focal_opt_logic', 'FocalOpt' task aborted. Error: {e}"
        print(error_msg)
        try:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        except:
            pass

    print(f"Background task 'FocalOpt' finished, logs written to: {os.path.abspath(output_filename)}")
    # Close logger handlers
    file_handler.close()
    logger.removeHandler(file_handler)


# --- Task Management Tools ---

# Global thread pool and task dictionary
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
running_tasks = {}





@mcp.tool()
async def find_initial_design(
        gmid1: int,
        gmid2: int,
        gmid3: int,
        gmid4: int,
        gmid5: int,
        iterations: int = 1200
) -> Dict[str, Any]:
    """
    Starts a background Bayesian Optimization task (find_initial_design) to find an initial feasible circuit design.

    Args:
        gmid1: gmid value for transistors M1 and M2.
        gmid2: gmid value for transistors M3 and M4.
        gmid3: gmid value for transistors M5 and M6.
        gmid4: gmid value for transistors M7 and M8.
        gmid5: gmid value for transistor M9.
        iterations: Maximum number of Bayesian Optimization iterations (default 1200).

    Returns:
        A dictionary containing the task ID and status information.
    """
    task_id = f"{int(time.time())}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"find_design_results_{timestamp}_{task_id}.log"

    future = executor.submit(
        _run_find_initial_design_task,
        gmid1, gmid2, gmid3, gmid4, gmid5,
        iterations,
        task_id,
        output_filename
    )

    running_tasks[task_id] = {
        'future': future,
        'output_file': output_filename,
        'start_time': time.time()
    }

    return {
        "status": "task_started",
        "task_id": task_id,
        "message": f"Task started, ID: {task_id}",
        "output_file": os.path.abspath(output_filename)
    }


@mcp.tool()
async def FocalOpt(
        initial_design_task_id: str,
        iterations: int = 450  # 450 = 50 + 100 + 100 + 200 (approx. total for all stages)
) -> Dict[str, Any]:
    """
    Starts the ASTRA-FocalOpt (Stage 2 optimization) task, focusing the optimization on the Stage 1 results.

    Args:
        initial_design_task_id: The Task ID from the Stage 1 (find_initial_design) task.
        iterations: The total maximum number of iterations for the FocalOpt stages (default 450).

    Returns:
        A dictionary containing the task ID and status information.
    """
    task_id = f"focalopt_{int(time.time())}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"focalopt_results_{timestamp}_{task_id}.log"

    future = executor.submit(
        _run_focal_opt_task,
        initial_design_task_id,
        iterations,
        task_id,
        output_filename
    )

    running_tasks[task_id] = {
        'future': future,
        'output_file': output_filename,
        'start_time': time.time()
    }

    return {
        "status": "task_started",
        "task_id": task_id,
        "message": f"FocalOpt task started, ID: {task_id}",
        "output_file": os.path.abspath(output_filename)
    }


@mcp.tool()
async def check_task_status(task_id: str) -> Dict[str, Any]:
    """
    Checks the status of a background task (find_initial_design, or FocalOpt).

    Args:
        task_id: The ID of the task to check.

    Returns:
        A dictionary containing the task status ("running", "completed", "failed", "not_found").
    """
    if task_id not in running_tasks:
        return {"status": "not_found", "message": "Task not found"}

    task = running_tasks[task_id]
    future = task['future']

    if future.done():
        try:
            future.result()
            return {"status": "completed", "output_file": os.path.abspath(task['output_file'])}
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            return {"status": "failed", "error": str(e), "output_file": os.path.abspath(task['output_file'])}
    else:
        runtime = time.time() - task['start_time']
        return {"status": "running", "runtime_seconds": runtime}


# Start the server
if __name__ == "__main__":
    # Ensure stdout supports UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            print("Stdout encoding set to UTF-8")
        except:
            print("Warning: Could not set stdout encoding to UTF-8")

    print("RAG Server is running, waiting for client connection...")
    mcp.run()

