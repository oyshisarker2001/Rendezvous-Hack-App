## [Open Deep Research Agent](https://github.com/Coral-Protocol/Coral-OpenDeepResearch-Agent)

The Open Deep Research agent is an open-source AI assistant that automates in-depth research and report generation via multi-agent workflows, supporting web search, structured reporting, human feedback, and API/LLM integration

![Agentimage](https://github.com/Coral-Protocol/awesome-agents-for-multi-agent-systems/blob/main/images/Coral-OpenDeepResearch-Agent.png)

## Responsibility
The Open Deep Research agent is an open-source research assistant that automates comprehensive report generation using a graph-based workflow or multi-agent architecture. It can perform in-depth web searches, generate structured reports, support human-in-the-loop feedback, and integrate with APIs like Tavily, Linkup, DuckDuckGo, and Azure AI Search, using customizable LLMs for tailored, high-quality research outputs.

## Details
- **Framework**: LangChain
- **Tools used**: OpenDeepResearch Tools, Coral server tools
- **AI model**: GPT-4o
- **Date added**: June 4, 2025
- **Reference**: [Open Deep Research Repo](https://github.com/langchain-ai/open_deep_research)
- **License**: MIT 


## Setup the Agent

### 1. Clone & Install Dependencies


<details>  

```bash
# In a new terminal clone the repository:
git clone https://github.com/Coral-Protocol/Coral-OpenDeepResearch-Agent.git

# Navigate to the project directory:
cd Coral-OpenDeepResearch-Agent

# Download and run the UV installer, setting the installation directory to the current one
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$(pwd) sh

# Create a virtual environment named `.venv` using UV
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# install uv
pip install uv

# Install dependencies from `pyproject.toml` using `uv`:
uv sync
```

</details>
 

### 2. Configure Environment Variables

<details>
 
Get the API Key:
[Linkup](https://app.linkup.so/api-keys) || 
[OpenAI](https://platform.openai.com/api-keys)


```bash
# Create .env file in project root
cp -r .env_sample .env
```
</details>


## Run the Agent

You can run in either of the below modes to get your system running.  

- The Executable Model is part of the Coral Protocol Orchestrator which works with [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio).  
- The Dev Mode allows the Coral Server and all agents to be seaprately running on each terminal without UI support.  

### 1. Executable Mode

Checkout: [How to Build a Multi-Agent System with Awesome Open Source Agents using Coral Protocol](https://github.com/Coral-Protocol/existing-agent-sessions-tutorial-private-temp) and update the file: `coral-server/src/main/resources/application.yaml` with the details below, then run the [Coral Server](https://github.com/Coral-Protocol/coral-server) and [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio). You do not need to set up the `.env` in the project directory for running in this mode; it will be captured through the variables below.

<details>

For Linux or MAC:

```bash
registry:
  # ... your other agents
  opendeepresearch_agent:
    options:
      - name: "OPENAI_API_KEY"
        type: "string"
        description: "Open AI Key"
      - name: "LINKUP_API_KEY"
        type: "string"
        description: "LINKUP API KEY"
      - name: "MODEL_NAME"
        type: "string"
        description: "What model to use (e.g 'gpt-4.1')"
        default: "gpt-4.1"
      - name: "MODEL_PROVIDER"
        type: "string"
        description: "What model provider to use (e.g 'openai', etc)"
        default: "openai"
      - name: "MODEL_MAX_TOKENS"
        type: "string"
        description: "Max tokens to use"
        default: "16000"
      - name: "MODEL_TEMPERATURE"
        type: "string"
        description: "What model temperature to use"
        default: "0.3"

    runtime:
      type: "executable"
      command: ["bash", "-c", "<replace with path to this agent>/run_agent.sh main.py"]
      environment:
        - option: "MODEL_API_KEY"
        - option: "LINKUP_API_KEY"
        - option: "MODEL_NAME"
        - option: "MODEL_PROVIDER"
        - option: "MODEL_MAX_TOKENS"
        - option: "MODEL_TEMPERATURE"

```

For Windows, create a powershell command (run_agent.ps1) and run:

```bash
command: ["powershell","-ExecutionPolicy", "Bypass", "-File", "${PROJECT_DIR}/run_agent.ps1","main.py"]
```

</details>

### 2. Dev Mode

Ensure that the [Coral Server](https://github.com/Coral-Protocol/coral-server) is running on your system and run below command in a separate terminal.

<details>

```bash
# Run the agent using `uv`:
uv run python main.py
```
</details>


## Example

<details>


```bash
# Input:
Write me a report on Model Context Protocol.

#Output:
The research report will be displayed directly in the console output when you run the agent.
```
</details>


## Creator Details
- **Name**: Suman Deb
- **Affiliation**: Coral Protocol
- **Contact**: [Discord](https://discord.com/invite/Xjm892dtt3)
