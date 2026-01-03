import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

# import anthropic
# import openai
import requests

# import src.ranchocordova.chatbot_enhanced as chatbot_enhanced
import torch
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)

import document_processors.mcp_logic as mcp_logic

# Import both chatbots
# import src.ranchocordova.chatbot as chatbot_original
import src.ranchocordova.chatbot_unified
from document_processors.claude_processor import ClaudeLikeDocumentProcessor
from document_processors.mcp_logic import (
    # call_mcp_tool,
    # call_tool_with_sql,
    detect_visualization_intent,
    # discover_tools,
    # generate_llm_response,
    # generate_table_description,
    generate_visualization,
    # parse_user_query,
)
from document_processors.specific_folder_reader import (
    OperationsDocumentProcessor,
    ProjectDocumentProcessor,
)
from src.ranchocordova.chatbot_unified import _llm, generate_answer, initialize_models

print("🔥 Warming models at startup")
src.ranchocordova.chatbot_unified.initialize_models()

model, tokenizer = src.ranchocordova.chatbot_unified._llm  # ✅ THIS WORKS

inputs = tokenizer("warmup", return_tensors="pt").to(model.device)
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=1)

print("🔥 Warm-up complete")


request_tracker = defaultdict(list)
request_lock = threading.Lock()
# openai_client = (
#   openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#  if os.getenv("OPENAI_API_KEY")
# else None
# )
# print("🔑  OpenAI client initialized:", openai_client is not None)


app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["JSON_AS_ASCII"] = False

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

MCP_GDRIVE_URL = os.getenv("MCP_GDRIVE_URL", "http://127.0.0.1:8000")

# Initialize Anthropic client
# try:
#    claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#    print("Claude client initialized successfully")
# except Exception as e:
#    print(f"Warning: Claude client not initialized: {e}")
claude_client = None

# Cache for file contents and folder structure
file_content_cache = {}
folder_structure_cache = {}

# We use this to for the project management and operation Q&A pages, to read the files
project_processor = None
operations_processor = None


def clean_and_format_response(response_text):
    """Clean up response and ensure proper markdown formatting"""
    import re

    # Remove metadata lines
    metadata_patterns = [
        r"---[\s\S]*?---",
        r"Iterative Analysis Complete.*?\.",
        r"Session total:.*?\.",
        r"Files accessed:.*?\.",
        r"Analysis based on.*?\.",
        r"\d+ iterations using.*?\.",
    ]

    for pattern in metadata_patterns:
        response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)

    # Remove ASCII box-drawing characters
    ascii_chars = ["│", "├", "┤", "┼", "─", "┬", "┴", "╋", "║", "═", "╔", "╗", "╚", "╝"]
    for char in ascii_chars:
        response_text = response_text.replace(char, "")

    # Fix malformed tables (pipes without proper spacing)
    # Convert: | Project | Cost to proper markdown table
    lines = response_text.split("\n")
    fixed_lines = []
    in_table = False

    for i, line in enumerate(lines):
        # Detect table lines (contains | but not formatted properly)
        if "|" in line and line.count("|") >= 2:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]

            if not in_table:
                # First row - add header
                fixed_lines.append("| " + " | ".join(cells) + " |")
                # Add separator
                fixed_lines.append("|" + "|".join(["---" for _ in cells]) + "|")
                in_table = True
            else:
                # Data rows
                fixed_lines.append("| " + " | ".join(cells) + " |")
        else:
            if in_table and line.strip() == "":
                in_table = False
            fixed_lines.append(line)

    response_text = "\n".join(fixed_lines)

    # Remove excessive blank lines
    response_text = re.sub(r"\n{3,}", "\n\n", response_text)

    # Ensure proper spacing around headers
    response_text = re.sub(r"\n(#{1,3}\s)", r"\n\n\1", response_text)
    response_text = re.sub(r"(#{1,3}\s[^\n]+)\n([^\n#])", r"\1\n\n\2", response_text)

    return response_text.strip()


#####################################################
#### ALL THE DIFFERENT PAGES' ROUTES IN THIS APP ####
#####################################################


# Dummy user model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


users = {"admin": User(1, "admin", "password123")}


@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if not username or not password:
            flash("Please provide both username and password", "danger")
            return render_template("login.html")

        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(
                url_for(
                    "agent_catalog",
                    cat="public_services",
                    dept="ranchocordova",
                    dept_display="City of Rancho Cordova",
                )
            )  # Changed from "dashboard"
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/categories")
@login_required
def categories():
    return render_template("categories.html")


@app.route("/dashboard/<category>")
@login_required
def dashboard(category):
    # Define ALL departments for each category (9 total)
    category_departments = {
        "public_services": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {
                "name": "Rancho Cordova",
                "key": "ranchocordova",
                "icon": "ranchocordova.jpeg",
            },
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {
                "name": "Office of Energy Infrastructure",
                "key": "energy",
                "icon": "energy.jpeg",
            },
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        ],
        "energy": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {
                "name": "Rancho Cordova",
                "key": "ranchocordova",
                "icon": "ranchocordova.jpeg",
            },
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {
                "name": "Office of Energy Infrastructure",
                "key": "energy",
                "icon": "energy.jpeg",
            },
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        ],
        "health": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {
                "name": "Rancho Cordova",
                "key": "ranchocordova",
                "icon": "ranchocordova.jpeg",
            },
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {
                "name": "Office of Energy Infrastructure",
                "key": "energy",
                "icon": "energy.jpeg",
            },
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        ],
    }

    departments = category_departments.get(category, [])
    category_names = {
        "public_services": "Public Services",
        "energy": "Energy",
        "health": "Health",
    }

    return render_template(
        "dashboard.html",
        departments=departments,
        category_name=category_names.get(category, "Unknown"),
        category=category,  # Add this so template can pass it
    )


@app.route("/ftb")
@login_required
def ftb():
    return render_template("ftb.html")


@app.route("/oops")
@login_required
def oops():
    return render_template("oops.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/insights")
@login_required
def insights():
    # Get breadcrumb parameters
    category = request.args.get("cat", "public_services")
    dept = request.args.get("dept", "")
    dept_display = request.args.get("dept_display", "")

    return render_template(
        "insights.html", category=category, dept=dept, dept_display=dept_display
    )


### This routes to the modules page ###
@app.route("/modules/<dept>")
@login_required
def modules(dept):
    icons = {
        "ftb": "ftb.jpeg",
        "dmv": "dmv.jpeg",
        "sanjose": "sanjose.jpeg",
        "edd": "edd.jpeg",
        "fiscal": "fiscal.jpeg",
        "ranchocordova": "ranchocordova.jpeg",
        "calpers": "calpers.jpeg",
        "cdfa": "cdfa.jpeg",
        "energy": "energy.jpeg",
    }

    display_names = {
        "ftb": "FTB",
        "dmv": "DMV",
        "sanjose": "San Jose",
        "edd": "EDD",
        "fiscal": "Fi$cal",
        "ranchocordova": "Rancho Cordova",
        "calpers": "CalPERS",
        "cdfa": "CDFA",
        "energy": "Energy",
    }

    modules_list = [
        {"name": "Agent Catalog", "icon": "qa.png", "route": "qa_page"},
        {"name": "Workflow", "icon": "workflow.png", "route": "oops"},
        {"name": "Transaction", "icon": "transaction.png", "route": "oops"},
        {"name": "Insights", "icon": "insights.png", "route": "insights"},
        {"name": "Data Management", "icon": "datamanagement.png", "route": "oops"},
        {"name": "Voice Agent", "icon": "voiceagent.png", "route": "voice_agent"},
    ]

    company_icon = icons.get(dept, "default.jpeg")
    company_name = display_names.get(dept, "Department")
    category = request.args.get("cat", "public_services")

    return render_template(
        "modules.html",
        company_icon=company_icon,
        company_name=company_name,
        modules=modules_list,
        category=category,
        dept=dept,  # Add this
        dept_display=display_names.get(dept, dept),  # Add this
    )


# Department-specific agent configurations
DEPARTMENT_AGENTS = {
    "ftb": [
        {"name": "Enterprise Q&A:<br>PMO Agent", "route": "qa_projects"},
        {"name": "Enterprise Q&A:<br>Operational Agent", "route": "qa_operations"},
        {"name": "Refund & Filing<br>Assistant", "route": "oops"},
        {"name": "Notice Explainer &<br>Resolution Agent", "route": "oops"},
        {"name": "Payment Plan &<br>Collections Agent", "route": "oops"},
        {"name": "Identity Verification &<br>Fraud Triage Agent", "route": "oops"},
        {"name": "Correspondence Intake,<br>Summarization & Routing", "route": "oops"},
        {"name": "Case Dossier &<br>Audit Prep Agent", "route": "oops"},
    ],
    "ranchocordova": [
        {
            "name": "Energy Efficiency<br>Agent",
            "route": "rancho_energy",
            "icon": "energy_agent_ranchocordovapng",
        },
        {
            "name": "Customer Service<br>Agent",
            "route": "rancho_customer_service",
            "icon": "customer_service_ranchocordova.png",
        },
    ],
    # Add more departments as needed
}


@app.route("/agent_catalog/<cat>/<dept>")
@login_required
def agent_catalog(cat, dept):
    dept_display_names = {
        "ftb": "FTB",
        "ranchocordova": "City of Rancho Cordova",
        "dmv": "DMV",
        "sanjose": "San Jose",
        "edd": "EDD",
        "calpers": "CalPERS",
        "cdfa": "CDFA",
        "energy": "Office of Energy Infrastructure",
        "fiscal": "Fi$cal",
    }

    dept_display = dept_display_names.get(dept, dept.upper())

    # Get agents for this department
    agents = DEPARTMENT_AGENTS.get(dept, [])

    return render_template(
        "agent_catalog.html",
        category=cat,
        dept=dept,
        dept_display=dept_display,
        agents=agents,
    )


@app.route("/rancho_energy")
@login_required
def rancho_energy():
    """Rancho Cordova Energy Efficiency Agent"""
    category = request.args.get("cat", "public_services")
    dept = "ranchocordova"
    dept_display = "City of Rancho Cordova"
    agent_name = "Energy Efficiency Agent"
    return render_template(
        "rancho_agent_chat_enhanced.html",  # ✅ ENHANCED
        category=category,
        dept=dept,
        dept_display=dept_display,
        agent_name=agent_name,
        agent_type="energy",
    )


@app.route("/rancho_customer_service")
@login_required
def rancho_customer_service():
    """Rancho Cordova Customer Service Agent"""
    category = request.args.get("cat", "public_services")
    dept = "ranchocordova"
    dept_display = "City of Rancho Cordova"
    agent_name = "Customer Service Agent"
    return render_template(
        "rancho_agent_chat_enhanced.html",  # ✅ ENHANCED (same template!)
        category=category,
        dept=dept,
        dept_display=dept_display,
        agent_name=agent_name,
        agent_type="customer_service",
    )


@app.route("/rancho_agent_api", methods=["POST"])
@login_required
def rancho_agent_api():
    """API endpoint for Rancho Cordova chatbots with visualization support"""
    try:
        data = request.json
        query = data.get("query", "")
        agent_type = data.get("agent_type", "")

        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})

        # generate_answer now auto-detects agent type and returns visualization
        result = generate_answer(query, agent_type)

        return jsonify(
            {
                "answer": result.get("answer", ""),
                "visualization": result.get("visualization", None),
                "source": "rancho_cordova",
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify(
            {
                "answer": f"I encountered an error: {str(e)}",
                "visualization": None,
                "source": "error",
            }
        ), 500


@app.route("/insights_api", methods=["POST"])
@login_required
def insights_api():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"answer": "Please enter a valid query."})

        # --- Intercept the meta-query BEFORE calling the LLM ---
        if "list" in query.lower() and (
            "tool" in query.lower()
            or "database" in query.lower()
            or "source" in query.lower()
        ):
            mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://0.0.0.0:8080")
            available_tools = mcp_logic.discover_tools(mcp_server_url)

            # Use the new function to format the list
            formatted_list = mcp_logic.list_available_tools(available_tools)

            return jsonify({"answer": formatted_list})

        mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
        available_tools = mcp_logic.discover_tools(mcp_server_url)
        if not available_tools:
            available_tools = {
                "Bigquery_Customer": "Customer data queries",
                "Cloud_SQL_Product": "Product catalog queries",
                "SAP_Hana_Sales": "Sales transaction queries",
                "Oracle_CustomerFeedback": "Customer feedback queries",
                "amazon_redshift_CustomerCallLog": "Customer service call logs",
            }

        parsed = mcp_logic.parse_user_query(query, available_tools)
        if "error" in parsed:
            return jsonify({"answer": parsed["error"]})

        tool = parsed.get("tool")
        sql = parsed.get("sql")

        if not tool or not sql:
            return jsonify(
                {"answer": "Could not parse your query. Please try rephrasing."}
            )

        data = mcp_logic.call_tool_with_sql(tool, sql, mcp_server_url)

        answer = None
        html_table = None
        viz_data = None
        has_visualization = False

        if data.get("rows") and len(data["rows"]) > 0:
            try:
                import pandas as pd

                df = pd.DataFrame(data["rows"])
                answer = mcp_logic.generate_table_description(
                    df, data, "read", tool, query
                )

                # FIX: Remove dual color tone and display all records
                headers = "".join([f"<th>{col}</th>" for col in df.columns])
                rows = "".join(
                    [
                        f"<tr>{''.join([f'<td>{value}</td>' for value in row])}</tr>"
                        for row in df.values  # Removed .head(50) to show all records
                    ]
                )

                html_table = f"""
                    <div class="mt-2 table-responsive">
                        <table class="table table-hover" id="data-table">
                            <thead>
                                <tr>{headers}</tr>
                            </thead>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                """

                if mcp_logic.detect_visualization_intent(query) == "Yes":
                    viz_data = mcp_logic.generate_visualization(
                        data.get("rows"), query, tool
                    )

                    if viz_data:
                        has_visualization = True
                    else:
                        print(
                            "No visualization data found for this query despite intent."
                        )
                        if answer is None:
                            answer = ""
                        answer += "\n\n⚠️ I was unable to generate a visualization for this request."

            except Exception as e:
                print(f"Error processing DataFrame or visualization: {e}")
                import traceback

                traceback.print_exc()
                answer = f"Successfully retrieved {data.get('row_count', 0)} records from the database."
                html_table = f"<div class='alert alert-info'>Data retrieved successfully. Total records: {data.get('row_count', 0)}</div>"
        else:
            answer = mcp_logic.generate_llm_response(data, "read", tool, query)
            html_table = (
                "<div class='alert alert-warning'>No data found for your query.</div>"
            )

        return jsonify(
            {
                "answer": answer or "Query processed successfully.",
                "html_table": html_table,
                "visualization": viz_data,
                "has_visualization": has_visualization,
            }
        )

    except Exception as e:
        print(f"Error in insights_api: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"answer": f"System error: {str(e)}"})


@app.route("/qa_projects")
@login_required
def qa_projects():
    """Project Management QA Chat"""
    category = request.args.get("cat", "public_services")
    dept = request.args.get("dept", "")
    dept_display = request.args.get("dept_display", "")

    return render_template(
        "qa_chat_projects.html", category=category, dept=dept, dept_display=dept_display
    )


@app.route("/qa_operations")
@login_required
def qa_operations():
    """Operations QA Chat"""
    category = request.args.get("cat", "public_services")
    dept = request.args.get("dept", "")
    dept_display = request.args.get("dept_display", "")

    return render_template(
        "qa_chat_operations.html",
        category=category,
        dept=dept,
        dept_display=dept_display,
    )


@app.route("/qa_api", methods=["POST"])
@login_required
def qa_api():
    global project_processor

    try:
        query = request.json.get("query")
        print(f"📝 Step 1: Received query: '{query}'")

        if query == "__INITIALIZE_DISCOVERY__":
            print("🔧 Step 2: Initializing new processor for discovery")
            if project_processor is None:
                project_processor = ProjectDocumentProcessor(
                    MCP_GDRIVE_URL, claude_client
                )

            # Actually trigger the document discovery
            print("📁 Forcing document awareness initialization...")
            if (
                "document_catalog" not in project_processor.session_context
                or not project_processor.session_context.get("document_catalog")
            ):
                discovered = project_processor._initialize_document_awareness()
                total_files = project_processor.session_context.get(
                    "total_documents_available", 0
                )

                return jsonify(
                    {
                        "answer": f"Discovery complete: {total_files} documents cataloged",
                        "status": "ready",
                        "files_discovered": total_files,
                    }
                )
            else:
                total_files = project_processor.session_context.get(
                    "total_documents_available", 0
                )
                return jsonify(
                    {
                        "answer": f"Already initialized: {total_files} documents available",
                        "status": "ready",
                        "files_discovered": total_files,
                    }
                )

        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})

        if project_processor is None:
            project_processor = ProjectDocumentProcessor(MCP_GDRIVE_URL, claude_client)
        else:
            print(
                f"📄 Step 2: Using existing processor - {len(project_processor.session_context.get('files_mentioned', {}))} files"
            )

        # Process query
        print("⚡ Step 3: Processing query...")
        response = project_processor.process_query_iteratively(query)

        # ✅ CLEAN AND FORMAT THE RESPONSE
        response = clean_and_format_response(response)

        print(f"📝 Step 4: Response length: {len(response)} chars")

        # Check if visualization is needed
        should_visualize = detect_visualization_intent(query) == "Yes" or any(
            keyword in query.lower()
            for keyword in ["budget", "cost", "project", "data", "numbers"]
        )
        print(f"🎯 Step 5: Should visualize: {should_visualize}")

        html_visualization = None
        html_table = None
        viz_data = None

        # NEW APPROACH: Use parsed table structure
        if (
            should_visualize
            and hasattr(project_processor, "last_table_data")
            and project_processor.last_table_data
        ):
            table_data = project_processor.last_table_data
            print(
                f"✅ Using parsed table: {len(table_data['headers'])} columns × {table_data['row_count']} rows"
            )
            print(f"   Headers: {table_data['headers']}")

            # Generate HTML table (properly aligned)
            html_table = project_processor.table_to_html(table_data)

            # Extract visualization data WITH context
            viz_data = project_processor.extract_visualization_data(table_data, query)

            if viz_data and len(viz_data["data"]) > 0:
                print(
                    f"📊 Generating visualization with {len(viz_data['numeric_columns'])} numeric columns"
                )
                print(f"   Columns: {viz_data['numeric_columns']}")

                try:
                    html_visualization = mcp_logic.generate_visualization(
                        data=viz_data,
                        user_query=query,
                        tool="Projects_GoogleDrive",
                        table_structure=table_data,  # Pass full table context!
                    )

                    if html_visualization:
                        print(f"✅ Generated visualization successfully")
                    else:
                        print(f"⚠️ Visualization generation returned None")

                except Exception as viz_error:
                    print(f"❌ Visualization error: {viz_error}")
                    import traceback

                    traceback.print_exc()
                    html_visualization = None
            else:
                print(f"⚠️ No numeric data found for visualization")
                print(f"   Table had {len(table_data['rows'])} rows")
        else:
            print(f"ℹ️ No table data available for visualization")
            if should_visualize:
                print(
                    f"   Reason: last_table_data = {getattr(project_processor, 'last_table_data', 'not set')}"
                )

            # Fallback: Try to extract table from response text
            if "|" in response:
                print("   Attempting direct table extraction from response...")
                html_table = extract_and_format_table(response)

        print(
            "DEBUG returning response type:",
            type(response),
            "value preview:",
            str(response)[:200],
        )
        print(f"\n=== FINAL RESPONSE DEBUG ===")
        print(f"Response length: {len(response)}")
        print(f"Response preview: {response[:500]}")
        if viz_data:
            print(f"Viz data keys: {list(viz_data.keys())}")
            for key, value in viz_data.items():
                if isinstance(value, dict):
                    print(
                        f"  {key}: {len(value)} items - Sample: {list(value.items())[:3]}"
                    )
        print(f"=== END DEBUG ===\n")

        if isinstance(response, str):
            response = response.encode("utf-8", errors="ignore").decode(
                "utf-8", "ignore"
            )
            response = response.replace("\x00", "")
            response = response.replace("\r\n", "\n").replace("\r", "\n")

        # Use make_response for explicit UTF-8 encoding
        from flask import make_response

        # html_table is now set above in the visualization section
        # Only generate it here if not already set
        if html_table is None and ("table" in response.lower() or "|" in response):
            html_table = extract_and_format_table(response)

        response_data = {
            "answer": response,
            "html_table": html_table,
            "html_visualization": html_visualization,
            "has_visualization": html_visualization is not None,
            "source": "projects_drive_proper_parsing",  # Changed name
        }

        resp = make_response(jsonify(response_data))
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        error_trace = traceback.format_exc()
        print(f"Stack trace: {error_trace}")

        with open("/tmp/qa_api_errors.log", "a") as f:
            f.write(f"{time.ctime()} - Error: {e}\n")
            f.write(f"Traceback: {error_trace}\n\n")

        return jsonify(
            {
                "answer": f"System error: {str(e)}",
                "html_visualization": None,
                "has_visualization": False,
                "source": "error",
            }
        ), 500  # Include status code for errors


@app.route("/qa_operations_api", methods=["POST"])
@login_required
def qa_operations_endpoint():
    """Operations QA with Google Drive processing + MCP visualizations"""
    global operations_processor

    try:
        query = request.json.get("query")

        if query == "__INITIALIZE_DISCOVERY__":
            print("🔧 Initializing operations discovery...")
            if operations_processor is None:
                operations_processor = OperationsDocumentProcessor(
                    MCP_GDRIVE_URL, claude_client
                )

            # Trigger document discovery
            print("📁 Forcing operations document awareness initialization...")
            if (
                "document_catalog" not in operations_processor.session_context
                or not operations_processor.session_context.get("document_catalog")
            ):
                discovered = operations_processor._initialize_document_awareness()
                total_files = operations_processor.session_context.get(
                    "total_documents_available", 0
                )

                return jsonify(
                    {
                        "answer": f"Discovery complete: {total_files} tax documents cataloged",
                        "status": "ready",
                        "files_discovered": total_files,
                    }
                )
            else:
                total_files = operations_processor.session_context.get(
                    "total_documents_available", 0
                )
                return jsonify(
                    {
                        "answer": f"Already initialized: {total_files} documents available",
                        "status": "ready",
                        "files_discovered": total_files,
                    }
                )

        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})

        print(f"\n=== OPERATIONS QA (DRIVE + MCP VIZ) ===")
        print(f"User query: '{query}'")

        http_server_url = MCP_GDRIVE_URL

        if not claude_client:
            return jsonify({"answer": "Claude API service not available."})

        # Initialize operations processor for Google Drive access
        if operations_processor is None:
            operations_processor = OperationsDocumentProcessor(
                http_server_url, claude_client
            )
            print("New operations session started with OperationsDocumentProcessor")
        else:
            print(
                f"Continuing operations session - {len(operations_processor.session_context['files_mentioned'])} files in context"
            )

        # Process query using Google Drive document processor
        print("Processing operations query with Google Drive access...")
        response = operations_processor.process_query_iteratively(query)

        # Try to extract visualization data from the response
        viz_data = None
        html_table = None
        html_visualization = None

        try:
            if (
                hasattr(operations_processor, "last_table_data")
                and operations_processor.last_table_data
            ):
                print(
                    f"✅ Found parsed table with {len(operations_processor.last_table_data['headers'])} columns"
                )
                viz_data = operations_processor.extract_visualization_data(
                    operations_processor.last_table_data, query
                )
            else:
                print("ℹ️ No table data found")
        except Exception as e:
            print(f"⚠️ Visualization extraction failed: {e}")
            viz_data = None

        # If we have tabular data in the response, try to create an HTML table
        if "table" in response.lower() or "|" in response:
            html_table = extract_and_format_table(response)

        # Generate visualization using MCP logic if data is available
        if viz_data:
            print(f"Found visualization data: {list(viz_data.keys())}")

            html_visualization = mcp_logic.generate_visualization(
                viz_data, query, "Operations_GoogleDrive"
            )

        else:
            print("No operations visualization data found")

        if isinstance(response, str):
            response = response.encode("utf-8", errors="ignore").decode(
                "utf-8", "ignore"
            )
            response = (
                response.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
            )

        # Use make_response for explicit UTF-8 encoding
        from flask import make_response

        response_data = {
            "answer": response,
            "html_table": html_table,
            "html_visualization": html_visualization,
            "has_visualization": html_visualization is not None,
            "source": "projects_drive_with_mcp_viz",
        }

        resp = make_response(jsonify(response_data))
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp

    except Exception as e:
        print(f"Error in operations processing: {e}")
        import traceback

        traceback.print_exc()
        return jsonify(
            {
                "answer": f"System error: {str(e)}",
                "html_table": None,
                "html_visualization": None,
                "has_visualization": False,
                "source": "error",
            }
        ), 500


def extract_and_format_table(response_text):
    """Extract table data from response text and format as HTML"""
    lines = response_text.split("\n")
    table_lines = []

    # Look for lines that contain table-like data (with | separators)
    for line in lines:
        if "|" in line and len(line.split("|")) >= 3:
            # Clean up the line
            cleaned_line = line.strip()
            if cleaned_line and not all(c in "|-= " for c in cleaned_line):
                table_lines.append(cleaned_line)

    if len(table_lines) >= 2:  # Need at least header + 1 data row
        try:
            # Process the first line as headers
            headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]

            # Process remaining lines as data
            rows = []
            for line in table_lines[1:]:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == len(headers):  # Only include properly formatted rows
                    rows.append(cells)

            if rows:
                # Build HTML table
                headers_html = "".join([f"<th>{h}</th>" for h in headers])
                rows_html = "".join(
                    [
                        f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>"
                        for row in rows
                    ]
                )

                return f"""
                    <div class="mt-3 table-responsive">
                        <table class="table table-hover table-striped" id="operations-table">
                            <thead class="table-dark">
                                <tr>{headers_html}</tr>
                            </thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                    </div>
                """
        except Exception as e:
            print(f"Table extraction error: {e}")

    return None


### This routes to the voice agent page ###
@app.route("/voice_agent")
@login_required
def voice_agent():
    """Voice Agent Page"""
    return render_template("voice_agent.html")


#####################################################
#### RELATED TO SESSION CACHE AND FOR INSPECTION ####
#####################################################
# Add a simple cache inspection endpoint
@app.route("/inspect_cache")
@login_required
def inspect_cache():
    cache_info = {
        "cache_size": len(file_content_cache),
        "cached_file_ids": list(file_content_cache.keys()),
    }
    return jsonify(cache_info)


# Enhanced cache clearing and debugging endpoints
@app.route("/clear_cache", methods=["POST"])
@login_required
def clear_cache():
    global file_content_cache, folder_structure_cache
    file_content_cache = {}
    folder_structure_cache = {}
    return jsonify({"status": "All caches cleared"})


@app.route("/cache_status")
@login_required
def cache_status():
    return jsonify(
        {
            "cached_files": len(file_content_cache),
            "cached_folders": len(folder_structure_cache),
            "cache_keys": list(file_content_cache.keys())[
                :10
            ],  # First 10 for debugging
        }
    )


# Add endpoint to view session context
@app.route("/session_context", methods=["GET"])
@login_required
def view_session_context():
    global document_processor

    if document_processor is None:
        return jsonify({"status": "No active session"})

    return jsonify(
        {
            "session_active": True,
            "files_mentioned": len(
                document_processor.session_context["files_mentioned"]
            ),
            "topics_discussed": document_processor.session_context["topics_discussed"],
            "session_summary": document_processor.session_context["session_summary"],
            "conversation_length": len(document_processor.conversation_history),
        }
    )


@app.route("/debug_routes", methods=["GET"])
def debug_routes():
    """Show all available routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    return "<br>".join(sorted(routes))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
